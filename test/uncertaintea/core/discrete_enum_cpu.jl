# Local mean helper (Statistics is not imported by the test harness).
denc_mean(x) = sum(x) / length(x)

# PR-3 of docs/discrete-enumeration.md (issue #13): compiled CPU semantics of
# marginalize=:enumerate. A flagged finite-support latent owns its plan suffix
# and the logjoint logsumexp-combines the per-support-value suffix scores, so
# HMC/NUTS sample only the continuous parameters against the marginal density.

@tea static function denc_indicator_model()
    m1 ~ normal(-2.0, 1.0)
    m2 ~ normal(2.0, 1.0)
    z ~ bernoulli(0.3; marginalize=:enumerate)
    {:y} ~ normal(z * m1 + (1 - z) * m2, 0.5)
    return m1
end

# the same marginal written with the single-site mixture machinery: the
# acceptance oracle from issue #13
@tea static function denc_mixture_model()
    m1 ~ normal(-2.0, 1.0)
    m2 ~ normal(2.0, 1.0)
    {:y} ~ mixture((0.3, 0.7), normal(m1, 0.5), normal(m2, 0.5))
    return m1
end

@tea static function denc_categorical_model()
    mu ~ normal(0.0, 1.0)
    z ~ categorical([0.2, 0.3, 0.5]; marginalize=:enumerate)
    {:y} ~ normal(mu * z, 0.4)
    return mu
end

# two marginalized latents: the suffix recursion gives product enumeration
@tea static function denc_two_latent_model()
    mu ~ normal(0.0, 1.0)
    a ~ bernoulli(0.4; marginalize=:enumerate)
    b ~ bernoulli(0.7; marginalize=:enumerate)
    {:y} ~ normal(mu + a + 2 * b, 0.5)
    return mu
end

# the enumerated latent's own probability depends on an upstream continuous
# latent: d(logsumexp)/dp flows through the bernoulli pmf terms
@tea static function denc_dependent_p_model()
    x ~ beta(2.0, 2.0)
    z ~ bernoulli(x; marginalize=:enumerate)
    {:y} ~ normal(z * 2.0, 1.0)
    return x
end

# a self-referential rebind in the suffix: every enumerated branch must
# re-run from the same pre-branch environment (a leaked `s` from branch one
# would double-increment in branch two)
@tea static function denc_rebind_model()
    mu ~ normal(0.0, 1.0)
    s = 1.0
    z ~ bernoulli(0.4; marginalize=:enumerate)
    s = s + 1.0
    {:y} ~ normal(mu * s + z, 0.5)
    return mu
end

# a zero-mass branch whose suffix is unevaluable (sigma = 0 for z = false):
# the marginalizer must skip it instead of throwing
@tea static function denc_zero_mass_model()
    mu ~ normal(0.0, 1.0)
    z ~ bernoulli(1.0; marginalize=:enumerate)
    {:y} ~ normal(mu, z * 1.0)
    return mu
end

# the enumerated category feeds an integer consumer (binomial trials): the
# backend binding must stay an index slot
@tea static function denc_index_consumer_model()
    p ~ beta(2.0, 2.0)
    z ~ categorical([0.5, 0.5]; marginalize=:enumerate)
    {:y} ~ binomial(z, p)
    return p
end

# the z = 1 branch has p = x, which exceeds 1 (a throwing scorer) for columns
# with large x; those columns condition on z = false and never need the branch
@tea static function denc_partial_branch_model()
    x ~ lognormal(0.0, 1.0)
    z ~ bernoulli(0.5; marginalize=:enumerate)
    {:w} ~ bernoulli(x * z + 0.1 * (1 - z))
    return x
end

# a Pair model-argument address in the suffix: the backend index machinery
# cannot score it, so the per-column workspace path must drop to the compiled
# plan (the same gap exists without the enumerated latent)
@tea static function denc_dynamic_address_model(addr)
    mu ~ normal(0.0, 1.0)
    z ~ bernoulli(0.5; marginalize=:enumerate)
    {addr} ~ normal(mu + z, 1.0)
    return mu
end

# conditioning a marginalized categorical on an out-of-support value binds the
# RAW value into the suffix on the reference path; bernoulli(z * 0.3) stays
# evaluable (total -Inf via the pmf), bernoulli(z) throws
@tea static function denc_cat_out_of_support_model()
    mu ~ normal(0.0, 1.0)
    z ~ categorical([0.5, 0.5]; marginalize=:enumerate)
    {:w} ~ bernoulli(z * 0.3)
    return mu
end

@tea static function denc_cat_out_of_support_throwing_model()
    mu ~ normal(0.0, 1.0)
    z ~ categorical([0.4, 0.6]; marginalize=:enumerate)
    {:w} ~ bernoulli(1.0 * z)
    return mu
end

# six nested bernoulli latents = support product 64 > the backend limit 32
@tea static function denc_support_cap_model()
    mu ~ normal(0.0, 1.0)
    a ~ bernoulli(0.5; marginalize=:enumerate)
    b ~ bernoulli(0.5; marginalize=:enumerate)
    c ~ bernoulli(0.5; marginalize=:enumerate)
    d ~ bernoulli(0.5; marginalize=:enumerate)
    e ~ bernoulli(0.5; marginalize=:enumerate)
    f ~ bernoulli(0.5; marginalize=:enumerate)
    {:y} ~ normal(mu + a + b + c + d + e + f, 0.5)
    return mu
end

@testset "discrete_enum_cpu" begin
    denc_constraints = choicemap((:y, 0.8))

    denc_fd_gradient = function (model, x, constraints)
        g = similar(x)
        for i in eachindex(x)
            h = cbrt(eps(Float64)) * max(1.0, abs(x[i]))
            xp = copy(x)
            xp[i] += h
            xm = copy(x)
            xm[i] -= h
            g[i] =
                (
                    logjoint_unconstrained(model, xp, (), constraints) -
                    logjoint_unconstrained(model, xm, (), constraints)
                ) / (2h)
        end
        return g
    end

    @testset "denc_density_parity" begin
        # the indicator spelling and the mixture spelling are the SAME
        # logsumexp, so density parity is exact, not approximate
        for denc_params in ([-1.5, 1.7], [0.0, 0.0], [2.0, -2.0])
            @test logjoint_unconstrained(denc_indicator_model, denc_params, (), denc_constraints) ==
                  logjoint_unconstrained(denc_mixture_model, denc_params, (), denc_constraints)
        end

        # the batched path is backend-native since PR-4 and must marginalize
        # identically to the per-column reference
        denc_batch = [-1.5 -1.2; 1.7 2.1]
        @test batched_logjoint_unconstrained(denc_indicator_model, denc_batch, (), denc_constraints) ≈ [
            logjoint_unconstrained(denc_indicator_model, denc_batch[:, index], (), denc_constraints) for
            index = 1:2
        ] atol = 1e-12
    end

    @testset "denc_backend_native" begin
        # the flag lowers to the suffix-owning backend step (PR-4); the
        # workspace form guarantees the backend plan scored these values
        @test backend_report(denc_indicator_model).supported == true
        denc_bn_params = [-1.5 -1.2 0.3; 1.7 2.1 -0.4]
        denc_bn_workspace = UncertainTea.BatchedLogjointWorkspace(denc_indicator_model)
        denc_bn_values = UncertainTea._logjoint_with_batched_backend!(
            denc_bn_workspace,
            denc_bn_params,
            (),
            denc_constraints,
        )
        @test denc_bn_values ≈ [
            logjoint_unconstrained(denc_indicator_model, denc_bn_params[:, index], (), denc_constraints)
            for index = 1:3
        ] atol = 1e-12

        # conditioning on 0/1 numerics selects the matching branch (Bool ==
        # Int equality), and a non-boolean numeric like 2 -- which the
        # runtime pmf treats as true while binding the raw value -- routes to
        # the per-column path so the reference semantics survive
        @test batched_logjoint_unconstrained(
            denc_indicator_model,
            denc_bn_params,
            (),
            choicemap((:z, 1), (:y, 0.8)),
        ) ≈ [
            logjoint(denc_indicator_model, denc_bn_params[:, index], (), choicemap((:z, true), (:y, 0.8)))
            for index = 1:3
        ] atol = 1e-12
        @test batched_logjoint_unconstrained(
            denc_indicator_model,
            denc_bn_params,
            (),
            choicemap((:z, 2), (:y, 0.8)),
        ) ≈ [
            logjoint(denc_indicator_model, denc_bn_params[:, index], (), choicemap((:z, 2), (:y, 0.8)))
            for index = 1:3
        ] atol = 1e-12

        # per-column heterogeneous conditioning: column 1 marginalizes,
        # columns 2/3 condition on opposite indicator values
        denc_bn_heterogeneous = [
            choicemap((:y, 0.8)),
            choicemap((:z, true), (:y, 0.8)),
            choicemap((:z, false), (:y, 0.8)),
        ]
        denc_bn_het_values = UncertainTea._logjoint_with_batched_backend!(
            denc_bn_workspace,
            denc_bn_params,
            (),
            denc_bn_heterogeneous,
        )
        @test denc_bn_het_values ≈ [
            logjoint(denc_indicator_model, denc_bn_params[:, index], (), denc_bn_heterogeneous[index])
            for index = 1:3
        ] atol = 1e-12

        # categorical binding consumed as an INDEX (binomial trials) stays an
        # index slot through the backend lowering
        @test backend_report(denc_index_consumer_model).supported == true
        denc_bn_index_params = reshape([0.55, 0.35], 1, 2)
        denc_bn_index_workspace = UncertainTea.BatchedLogjointWorkspace(denc_index_consumer_model)
        @test UncertainTea._logjoint_with_batched_backend!(
            denc_bn_index_workspace,
            denc_bn_index_params,
            (),
            choicemap((:y, 1)),
        ) ≈ [
            logjoint(denc_index_consumer_model, denc_bn_index_params[:, index], (), choicemap((:y, 1)))
            for index = 1:2
        ] atol = 1e-12

        # zero-mass branches are skipped in the batched scorer too
        @test backend_report(denc_zero_mass_model).supported == true
        denc_bn_zero_workspace = UncertainTea.BatchedLogjointWorkspace(denc_zero_mass_model)
        @test UncertainTea._logjoint_with_batched_backend!(
            denc_bn_zero_workspace,
            reshape([0.4], 1, 1),
            (),
            choicemap((:y, 0.9)),
        )[1] ≈ UncertainTea.logpdf(normal(0.0, 1.0), 0.4) + UncertainTea.logpdf(normal(0.4, 1.0), 0.9) atol =
            1e-12

        # gradients ride the flat ForwardDiff tier through the backend value
        # path (analytic marginalize gradients land in PR-5)
        denc_bn_cache = BatchedLogjointGradientCache(denc_indicator_model, denc_bn_params, (), denc_constraints)
        @test isnothing(denc_bn_cache.backend_cache)
        denc_bn_gradient = batched_logjoint_gradient_unconstrained(denc_bn_cache, denc_bn_params)
        for index = 1:3
            @test denc_bn_gradient[:, index] ≈
                  denc_fd_gradient(denc_indicator_model, denc_bn_params[:, index], denc_constraints) atol = 5e-6
        end

        # nested support products beyond the backend limit reject honestly
        denc_bn_cap_report = backend_report(denc_support_cap_model)
        @test denc_bn_cap_report.supported == false
        @test any(issue -> occursin("support product", issue), denc_bn_cap_report.issues)

        # a Pair model-argument address in the suffix drops all the way to
        # the compiled plan (backend index slots cannot hold it); this held
        # neither for plain nor for enumerated models before the workspace
        # fallback learned to retry on the compiled steps
        denc_bn_addr = :a => 1
        denc_bn_addr_params = reshape([0.3, -0.2], 1, 2)
        @test backend_report(denc_dynamic_address_model).supported == true
        @test batched_logjoint_unconstrained(
            denc_dynamic_address_model,
            denc_bn_addr_params,
            (denc_bn_addr,),
            choicemap((denc_bn_addr, 0.4)),
        ) ≈ [
            logjoint_unconstrained(
                denc_dynamic_address_model,
                denc_bn_addr_params[:, index],
                (denc_bn_addr,),
                choicemap((denc_bn_addr, 0.4)),
            ) for index = 1:2
        ] atol = 1e-12

        # conditioning a marginalized categorical on an out-of-support value
        # reproduces the reference semantics through the fallback: the raw
        # value is bound into the suffix, so the pmf-only case totals -Inf
        # and the invalid-parameter case throws like the compiled path
        denc_bn_oos_params = reshape([0.2], 1, 1)
        @test batched_logjoint_unconstrained(
            denc_cat_out_of_support_model,
            denc_bn_oos_params,
            (),
            choicemap((:z, 3), (:w, true)),
        ) == [-Inf]
        @test_throws ArgumentError batched_logjoint_unconstrained(
            denc_cat_out_of_support_throwing_model,
            denc_bn_oos_params,
            (),
            choicemap((:z, 3), (:w, true)),
        )

        # a branch body runs for every column, including columns whose result
        # is ignored -- if such a column makes the suffix throw (here p = x >
        # 1 in the z = 1 branch for a column conditioned on z = false), the
        # scorer routes to the per-column fallback, which stays exact
        denc_bn_partial_params = reshape([log(0.5), log(1.5)], 1, 2)
        denc_bn_partial_constraints = [
            choicemap((:w, true)),
            choicemap((:z, false), (:w, true)),
        ]
        @test backend_report(denc_partial_branch_model).supported == true
        @test batched_logjoint_unconstrained(
            denc_partial_branch_model,
            denc_bn_partial_params,
            (),
            denc_bn_partial_constraints,
        ) ≈ [
            logjoint_unconstrained(
                denc_partial_branch_model,
                denc_bn_partial_params[:, index],
                (),
                denc_bn_partial_constraints[index],
            ) for index = 1:2
        ] atol = 1e-12
    end

    @testset "denc_gradients" begin
        denc_params = [-1.5, 1.7]
        denc_gradient = logjoint_gradient_unconstrained(denc_indicator_model, denc_params, (), denc_constraints)
        @test denc_gradient ≈
              logjoint_gradient_unconstrained(denc_mixture_model, denc_params, (), denc_constraints) atol = 1e-12
        @test denc_gradient ≈ denc_fd_gradient(denc_indicator_model, denc_params, denc_constraints) atol = 5e-6

        denc_dep_params = [0.2]
        @test logjoint_gradient_unconstrained(
            denc_dependent_p_model,
            denc_dep_params,
            (),
            choicemap((:y, 1.5)),
        ) ≈ denc_fd_gradient(denc_dependent_p_model, denc_dep_params, choicemap((:y, 1.5))) atol = 5e-6
    end

    @testset "denc_categorical_and_nested" begin
        denc_cat_params = [0.6]
        denc_cat_terms = [
            log(w) + UncertainTea.logpdf(normal(0.6 * k, 0.4), 1.1) for
            (k, w) in enumerate([0.2, 0.3, 0.5])
        ]
        denc_cat_shift = maximum(denc_cat_terms)
        @test logjoint_unconstrained(denc_categorical_model, denc_cat_params, (), choicemap((:y, 1.1))) ≈
              UncertainTea.logpdf(normal(0.0, 1.0), 0.6) +
              denc_cat_shift +
              log(sum(exp.(denc_cat_terms .- denc_cat_shift))) atol = 1e-12

        denc_two_params = [0.3]
        denc_two_expected =
            UncertainTea.logpdf(normal(0.0, 1.0), 0.3) + log(
                sum(
                    (a == 1 ? 0.4 : 0.6) * (b == 1 ? 0.7 : 0.3) *
                    exp(UncertainTea.logpdf(normal(0.3 + a + 2b, 0.5), 2.2)) for a = 0:1, b = 0:1
                ),
            )
        @test logjoint_unconstrained(denc_two_latent_model, denc_two_params, (), choicemap((:y, 2.2))) ≈
              denc_two_expected atol = 1e-12
        @test logjoint_gradient_unconstrained(
            denc_two_latent_model,
            denc_two_params,
            (),
            choicemap((:y, 2.2)),
        ) ≈ denc_fd_gradient(denc_two_latent_model, denc_two_params, choicemap((:y, 2.2))) atol = 5e-6
    end

    @testset "denc_suffix_rebind_isolation" begin
        denc_rebind_params = [0.7]
        denc_rebind_terms = [
            log(z == 1 ? 0.4 : 0.6) + UncertainTea.logpdf(normal(0.7 * 2.0 + z, 0.5), 1.9) for z = 0:1
        ]
        denc_rebind_shift = maximum(denc_rebind_terms)
        @test logjoint_unconstrained(denc_rebind_model, denc_rebind_params, (), choicemap((:y, 1.9))) ≈
              UncertainTea.logpdf(normal(0.0, 1.0), 0.7) +
              denc_rebind_shift +
              log(sum(exp.(denc_rebind_terms .- denc_rebind_shift))) atol = 1e-12
    end

    @testset "denc_zero_mass_branch" begin
        # bernoulli(1.0) puts zero mass on false, whose branch has sigma = 0;
        # the marginal is just the z = true branch
        denc_zm_params = [0.4]
        @test logjoint_unconstrained(denc_zero_mass_model, denc_zm_params, (), choicemap((:y, 0.9))) ≈
              UncertainTea.logpdf(normal(0.0, 1.0), 0.4) +
              UncertainTea.logpdf(normal(0.4, 1.0), 0.9) atol = 1e-12
    end

    @testset "denc_conditioning" begin
        # a constrained latent short-circuits to the plain joint at that value
        denc_params = [-1.5, 1.7]
        denc_joint = logjoint(denc_indicator_model, denc_params, (), choicemap((:z, true), (:y, 0.8)))
        @test denc_joint ≈
              UncertainTea.logpdf(normal(-2.0, 1.0), -1.5) +
              UncertainTea.logpdf(normal(2.0, 1.0), 1.7) +
              UncertainTea.logpdf(bernoulli(0.3), true) +
              UncertainTea.logpdf(normal(-1.5, 0.5), 0.8) atol = 1e-12
        # ... and matches assess, which always scores the provided joint
        @test denc_joint ≈ assess(
            denc_indicator_model,
            (),
            choicemap((:m1, -1.5), (:m2, 1.7), (:z, true), (:y, 0.8)),
        ) atol = 1e-6
    end

    @testset "denc_nuts_parity" begin
        # issue #13 acceptance: the explicit-indicator spelling recovers the
        # same posterior as the mixture spelling via NUTS. Pool a few seeded
        # chains per spelling and compare the continuous-parameter means.
        denc_ind_m1 = Float64[]
        denc_ind_m2 = Float64[]
        denc_mix_m1 = Float64[]
        denc_mix_m2 = Float64[]
        for denc_seed = 1:3
            denc_ind_chain = nuts(
                denc_indicator_model,
                (),
                denc_constraints;
                num_samples=300,
                num_warmup=200,
                rng=MersenneTwister(700 + denc_seed),
            )
            denc_mix_chain = nuts(
                denc_mixture_model,
                (),
                denc_constraints;
                num_samples=300,
                num_warmup=200,
                rng=MersenneTwister(800 + denc_seed),
            )
            @test all(isfinite, denc_ind_chain.constrained_samples)
            @test all(isfinite, denc_mix_chain.constrained_samples)
            append!(denc_ind_m1, denc_ind_chain.constrained_samples[1, :])
            append!(denc_ind_m2, denc_ind_chain.constrained_samples[2, :])
            append!(denc_mix_m1, denc_mix_chain.constrained_samples[1, :])
            append!(denc_mix_m2, denc_mix_chain.constrained_samples[2, :])
        end
        @test denc_mean(denc_ind_m1) ≈ denc_mean(denc_mix_m1) atol = 0.15
        @test denc_mean(denc_ind_m2) ≈ denc_mean(denc_mix_m2) atol = 0.15
    end
end
