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

        # the batched path rides the per-column fallback (backend rejects the
        # flag until PR-4) and must marginalize identically
        denc_batch = [-1.5 -1.2; 1.7 2.1]
        @test batched_logjoint_unconstrained(denc_indicator_model, denc_batch, (), denc_constraints) ≈ [
            logjoint_unconstrained(denc_indicator_model, denc_batch[:, index], (), denc_constraints) for
            index = 1:2
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
