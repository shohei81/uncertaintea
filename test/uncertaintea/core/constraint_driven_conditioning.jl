# Constraint-driven latent/observation classification (issue #95,
# docs/constraint-driven-conditioning.md). A choice is an observation iff its
# address is in the constraints at inference time; binding is orthogonal. These
# regression tests pin the PR-2 scope: the compiled `logjoint` path derives the
# latent/observation split from the conditioning signature, so it agrees with
# `generate`/`assess`, honors a constraint on a BOUND address (the #95 bug),
# treats an unbound unconstrained choice as a slotted latent, and memoizes the
# resolved plan per signature.

# The #95 reproduction model: `:y` is bound (`y = ({:y} ~ ...)`) yet may be
# observed. Under the old syntactic rule the compiled scoring silently ignored a
# constraint on `:y`; under the conditioning rule it is honored.
@tea static function cdc_chain()
    mu ~ normal(0.0, 1.0)
    y = ({:y} ~ normal(mu, 1.0))
    {:z} ~ normal(y, 1.0)
    return mu
end

@tea static function cdc_gaussian()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# PR-3 value-resolution models. Each threads a possibly-observed `:y` through a
# downstream expression, so the evaluator must resolve `:y` to its constrained
# value (when observed) or its latent value (when not) via one path, in both the
# scoring fold and the dependent-transform walk.

# `:y` observed value feeds a deterministic step that also depends on the latent
# `mu`, so the latent gradient must pick up the observed contribution.
@tea static function cdc_downstream()
    mu ~ normal(0.0, 1.0)
    y = ({:y} ~ normal(mu, 1.0))
    w = y * 2.0 + mu
    {:z} ~ normal(w, 1.0)
    return mu
end

# `:y` observed value feeds a reparam=:noncentered loc/scale (theta = y + z).
# Before PR-3 the transform walk poisoned the slotless `:y` binding and this
# raised, even though the observed value is known. Centered twin for crosscheck.
@tea static function cdc_noncentered_obs()
    mu ~ normal(0.0, 1.0)
    y = ({:y} ~ normal(mu, 1.0))
    theta ~ normal(y, 1.0; reparam=:noncentered)
    {:z} ~ normal(theta, 1.0)
    return mu
end

@tea static function cdc_centered_obs()
    mu ~ normal(0.0, 1.0)
    y = ({:y} ~ normal(mu, 1.0))
    theta ~ normal(y, 1.0)
    {:z} ~ normal(theta, 1.0)
    return mu
end

@testset "constraint_driven_conditioning" begin
    normlogpdf(mean, sd, x) = UncertainTea.logpdf(normal(mean, sd), x)

    @testset "#95 bound observation is honored and agrees with assess" begin
        mu0 = 0.5
        yobs = 100.0
        zobs = 0.0
        cons = choicemap(:y => yobs, :z => zobs)

        # signature {:y, :z}: only `mu` is a latent, so the compiled scoring
        # path takes a single parameter and reads y, z from the constraints.
        layout = UncertainTea._resolve_signature_plan(cdc_chain, cons).plan.parameter_layout
        @test parametervaluecount(layout) == 1
        @test length(layout.slots) == 1
        @test layout.slots[1].binding == :mu

        lj = logjoint(cdc_chain, [mu0], (), cons)
        manual = normlogpdf(0.0, 1.0, mu0) + normlogpdf(mu0, 1.0, yobs) + normlogpdf(yobs, 1.0, zobs)
        assessed = assess(cdc_chain, (), choicemap(:mu => mu0, :y => yobs, :z => zobs))

        @test lj ≈ manual atol = 1e-9
        @test lj ≈ assessed atol = 1e-9

        # The observation is genuinely conditioned on: a different constrained
        # value gives a different density (the #95 bug returned the same value
        # regardless, because the bound `:y` was scored from a free parameter).
        @test logjoint(cdc_chain, [mu0], (), choicemap(:y => yobs, :z => zobs)) !=
              logjoint(cdc_chain, [mu0], (), choicemap(:y => yobs + 5.0, :z => zobs))
    end

    @testset "#95 conjugate posterior (closed form)" begin
        # mu ~ N(0,1), y | mu ~ N(mu,1) observed at yobs. The z term does not
        # depend on mu, so as a function of mu the logjoint is the unnormalized
        # log posterior N(yobs/2, 1/2). Check the shape against the closed form.
        yobs = 3.0
        post_mean = yobs / 2
        post_sd = sqrt(0.5)
        cons = choicemap(:y => yobs, :z => 1.0)

        lj(mu) = logjoint(cdc_chain, [mu], (), cons)
        kernel(mu) = normlogpdf(post_mean, post_sd, mu)
        for mu in (-1.0, 0.0, post_mean, 1.7)
            @test (lj(mu) - lj(post_mean)) ≈ (kernel(mu) - kernel(post_mean)) atol = 1e-9
        end
    end

    @testset "unbound choice left unconstrained is a slotted latent" begin
        # `{:y} ~ normal(mu, 1)` with no constraint: under the conditioning rule
        # `:y` is a latent, gets a parameter slot, and its prior is scored.
        layout = UncertainTea._resolve_signature_plan(cdc_gaussian, choicemap()).plan.parameter_layout
        @test parametervaluecount(layout) == 2
        @test Set(slot.binding for slot in layout.slots) == Set([:mu, Symbol("")])

        mu0 = 0.2
        y0 = 0.9
        lj = logjoint(cdc_gaussian, [mu0, y0], (), choicemap())
        @test lj ≈ normlogpdf(0.0, 1.0, mu0) + normlogpdf(mu0, 1.0, y0) atol = 1e-9
    end

    @testset "constrained vs unconstrained differ by the observation log-density" begin
        mu0 = 0.4
        v = 2.5
        # constrained: `:y` observed -> 1 latent (mu); unconstrained: `:y`
        # latent -> 2 latents (mu, y).
        con_layout = UncertainTea._resolve_signature_plan(cdc_gaussian, choicemap(:y => v)).plan.parameter_layout
        unc_layout = UncertainTea._resolve_signature_plan(cdc_gaussian, choicemap()).plan.parameter_layout
        @test parametervaluecount(unc_layout) - parametervaluecount(con_layout) == 1

        l_con = logjoint(cdc_gaussian, [mu0], (), choicemap(:y => v))
        l_unc = logjoint(cdc_gaussian, [mu0, v], (), choicemap())
        y_term = normlogpdf(mu0, 1.0, v)
        # at matching values the two agree, and the y address contributes exactly
        # its log-density in both roles (observation vs prior draw).
        @test l_con ≈ l_unc atol = 1e-9
        @test (l_con - y_term) ≈ normlogpdf(0.0, 1.0, mu0) atol = 1e-9
    end

    @testset "signature memoization reuses the compiled plan" begin
        @tea static function cdc_memo()
            mu ~ normal(0.0, 1.0)
            {:y} ~ normal(mu, 1.0)
            return mu
        end

        @test isnothing(cdc_memo.signature_cache[])
        # Two runs at the same observed address but different data reuse the plan.
        logjoint(cdc_memo, [0.1], (), choicemap(:y => 1.0))
        first_resolved = UncertainTea._resolve_signature_plan(cdc_memo, choicemap(:y => 1.0))
        logjoint(cdc_memo, [0.1], (), choicemap(:y => 42.0))
        second_resolved = UncertainTea._resolve_signature_plan(cdc_memo, choicemap(:y => 42.0))
        @test first_resolved === second_resolved
        @test length(cdc_memo.signature_cache[]) == 1

        # A different signature (empty vs {:y}) gets its own cached entry.
        logjoint(cdc_memo, [0.1, 0.2], (), choicemap())
        @test length(cdc_memo.signature_cache[]) == 2
    end

    # PR-3: unified value resolution in the evaluator. The bound value of a
    # choice resolves to the constraint value when observed and to the
    # transformed parameter value when latent, through one path -- so a
    # downstream deterministic step / noncentered loc/scale expression sees the
    # right value regardless of classification.
    @testset "observed value used downstream in a deterministic step" begin
        mu0 = 0.4
        yv = 3.0
        zv = 0.5
        cons = choicemap(:y => yv, :z => zv)
        w = yv * 2.0 + mu0

        # scoring resolves the observed `:y` and feeds it to the deterministic
        # `w`; matches the manual density and `assess` on the same choices.
        manual = normlogpdf(0.0, 1.0, mu0) + normlogpdf(mu0, 1.0, yv) + normlogpdf(w, 1.0, zv)
        lj = logjoint(cdc_downstream, [mu0], (), cons)
        assessed = assess(cdc_downstream, (), choicemap(:mu => mu0, :y => yv, :z => zv))
        @test lj ≈ manual atol = 1e-9
        @test lj ≈ assessed atol = 1e-9
        # no dependent transform here, so the unconstrained path agrees exactly.
        @test logjoint_unconstrained(cdc_downstream, [mu0], (), cons) ≈ lj atol = 1e-9

        # the constrained observation genuinely reaches `w`: a different `:y`
        # value shifts the density (via the z term through w).
        @test logjoint(cdc_downstream, [mu0], (), choicemap(:y => yv, :z => zv)) !=
              logjoint(cdc_downstream, [mu0], (), choicemap(:y => yv + 1.0, :z => zv))

        # scalar gradient w.r.t. the latent `mu` is correct even though an
        # observed value feeds the downstream expression (it is a constant
        # w.r.t. the parameters). Analytic + finite-difference crosscheck.
        grad = logjoint_gradient_unconstrained(cdc_downstream, [mu0], (), cons)
        analytic = -mu0 + (yv - mu0) + (zv - w)
        f(m) = logjoint_unconstrained(cdc_downstream, [m], (), cons)
        h = 1e-6
        fd = (f(mu0 + h) - f(mu0 - h)) / (2h)
        @test grad[1] ≈ analytic atol = 1e-9
        @test grad[1] ≈ fd atol = 1e-5
    end

    @testset "constrained vs unconstrained with the address used downstream" begin
        mu0 = 0.4
        yv = 3.0
        zv = 0.5
        # `:y` observed -> latent {mu}; `:y` latent -> latents {mu, y}. With the
        # latent y set to the observed value the two agree, and `y` flows into
        # `w` in both roles (observation and prior draw).
        con_layout =
            UncertainTea._resolve_signature_plan(cdc_downstream, choicemap(:y => yv, :z => zv)).plan.parameter_layout
        unc_layout =
            UncertainTea._resolve_signature_plan(cdc_downstream, choicemap(:z => zv)).plan.parameter_layout
        @test parametervaluecount(con_layout) == 1
        @test parametervaluecount(unc_layout) == 2

        l_con = logjoint(cdc_downstream, [mu0], (), choicemap(:y => yv, :z => zv))
        l_unc = logjoint(cdc_downstream, [mu0, yv], (), choicemap(:z => zv))
        @test l_con ≈ l_unc atol = 1e-9
    end

    @testset "observed value feeds a reparam=:noncentered loc/scale" begin
        mu0 = 0.4
        yv = 3.0
        zv = 0.5
        zstd = 0.1
        cons = choicemap(:y => yv, :z => zv)
        theta = yv + zstd  # theta = location(y) + scale(1) * z

        # Before PR-3 the transform walk poisoned the slotless observed `:y`
        # feeding theta's location and this raised; now it resolves to yv.
        lj_nc = logjoint_unconstrained(cdc_noncentered_obs, [mu0, zstd], (), cons)
        manual =
            normlogpdf(0.0, 1.0, mu0) +
            normlogpdf(mu0, 1.0, yv) +
            normlogpdf(yv, 1.0, theta) +
            normlogpdf(theta, 1.0, zv)
        @test lj_nc ≈ manual atol = 1e-9

        # centered twin at the matching constrained theta gives the same joint
        # (scale 1 -> zero log-abs-det), confirming the noncentered walk used the
        # observed location.
        lj_centered = logjoint_unconstrained(cdc_centered_obs, [mu0, theta], (), cons)
        @test lj_nc ≈ lj_centered atol = 1e-9

        # gradient w.r.t. [mu, z] crosschecks against finite differences with the
        # observed location flowing into theta.
        grad = logjoint_gradient_unconstrained(cdc_noncentered_obs, [mu0, zstd], (), cons)
        g(p) = logjoint_unconstrained(cdc_noncentered_obs, p, (), cons)
        h = 1e-6
        fd = [
            (g([mu0 + h, zstd]) - g([mu0 - h, zstd])) / (2h),
            (g([mu0, zstd + h]) - g([mu0, zstd - h])) / (2h),
        ]
        @test grad ≈ fd atol = 1e-5
    end
end
