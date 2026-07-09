# PR 37: pointwise log-likelihood extraction + WAIC + PSIS-LOO model comparison.
# Contract: pointwise_loglikelihood recovers the per-observation logpdf so that the
# row sum plus the prior term equals logjoint; _gpd_fit recovers the shape/scale of a
# known generalized Pareto sample; WAIC and PSIS-LOO agree with each other and with an
# exact (unsmoothed) importance-sampling LOO on a well-behaved conjugate model.

@testset "waic_psis_loo" begin
    mc_logsumexp = function (v)
        m = maximum(v)
        return m + log(sum(exp.(v .- m)))
    end

    @tea static function mc_conjugate_model(n)
        mu ~ normal(0.0f0, 1.0f0)
        for i = 1:n
            {:y => i} ~ normal(mu, 1.0f0)
        end
        return mu
    end

    mc_ys = Float32[0.4f0, -0.7f0, 1.1f0]
    mc_constraints = choicemap((:y => i, mc_ys[i]) for i = 1:3)

    mc_chains = nuts_chains(
        mc_conjugate_model,
        (3,),
        mc_constraints;
        num_chains=4,
        num_samples=400,
        num_warmup=400,
        rng=MersenneTwister(370),
    )

    mc_obs_addresses = observation_addresses(mc_conjugate_model, (3,), mc_constraints)
    @test length(mc_obs_addresses) == 3
    @test mc_obs_addresses == Any[(:y, 1), (:y, 2), (:y, 3)]

    mc_ll = pointwise_loglikelihood(mc_conjugate_model, (3,), mc_constraints, mc_chains)
    @test size(mc_ll, 2) == 3
    @test size(mc_ll, 1) == 4 * 400

    # Pointwise correctness: row-sum of log-likelihoods plus the prior term for mu recovers
    # the full model logjoint for that draw. The pooled draw order matches the column-wise
    # concatenation of each chain's constrained samples.
    mc_pooled = reduce(hcat, [mc_chain.constrained_samples for mc_chain in mc_chains.chains])
    @test size(mc_pooled, 2) == size(mc_ll, 1)
    for mc_row in (1, 5, 137, 400, 801, 1600)
        mc_params = mc_pooled[:, mc_row]
        mc_prior = UncertainTea.logpdf(normal(0.0f0, 1.0f0), mc_params[1])
        mc_expected = logjoint(mc_conjugate_model, mc_params, (3,), mc_constraints)
        @test sum(mc_ll[mc_row, :]) + mc_prior ≈ mc_expected atol=1e-8
    end

    # _gpd_fit recovers the shape/scale of a known generalized Pareto sample.
    mc_gpd_rng = MersenneTwister(20240705)
    mc_true_k = 0.3
    mc_true_sigma = 1.0
    mc_gpd_samples = Float64[
        mc_true_sigma * ((1 - rand(mc_gpd_rng))^(-mc_true_k) - 1) / mc_true_k for _ = 1:2000
    ]
    mc_khat, mc_sigmahat = UncertainTea._gpd_fit(mc_gpd_samples)
    @test abs(mc_khat - mc_true_k) < 0.1
    @test abs(mc_sigmahat - mc_true_sigma) < 0.15

    # WAIC / PSIS-LOO sanity on the conjugate model.
    mc_waic = waic(mc_ll)
    mc_loo = psis_loo(mc_ll)

    mc_S = size(mc_ll, 1)
    mc_lppd = sum(mc_logsumexp(mc_ll[:, i]) - log(mc_S) for i = 1:size(mc_ll, 2))

    @test mc_loo.elpd <= mc_lppd
    @test mc_loo.p_eff > 0
    @test all(mc_loo.pareto_k .< 0.7)
    @test abs(mc_waic.elpd - mc_loo.elpd) < 0.5

    # Exact importance-sampling LOO (no smoothing) as a cross-check.
    for i = 1:size(mc_ll, 2)
        mc_col = mc_ll[:, i]
        mc_lw = -mc_col
        mc_lw_norm = mc_lw .- mc_logsumexp(mc_lw)
        mc_is_elpd_i = mc_logsumexp(mc_col .+ mc_lw_norm)
        @test abs(mc_loo.pointwise[i] - mc_is_elpd_i) < 0.3
    end

    # Convenience wrappers accept model + chains directly.
    mc_loo_conv = loo(mc_conjugate_model, (3,), mc_constraints, mc_chains)
    mc_waic_conv = waic(mc_conjugate_model, (3,), mc_constraints, mc_chains)
    @test mc_loo_conv.elpd ≈ mc_loo.elpd
    @test mc_waic_conv.elpd ≈ mc_waic.elpd

    # show smoke tests.
    @test !isempty(sprint(show, mc_waic))
    @test !isempty(sprint(show, mc_loo))
end
