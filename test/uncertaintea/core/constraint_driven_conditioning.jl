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
end
