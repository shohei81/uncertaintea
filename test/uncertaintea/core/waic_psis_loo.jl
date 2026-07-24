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

    # 4x800 draws: the Pareto-k estimator is noisy at smaller draw counts
    # (its tail-sample size scales with sqrt(S)), and a fixed-seed 4x400 run
    # left one observation's khat straddling the 0.7 reliability boundary on
    # some Julia versions after the issue #159 trajectory change (the true k
    # of this well-specified conjugate model is far below the boundary).
    mc_chains = nuts_chains(
        mc_conjugate_model,
        (3,),
        mc_constraints;
        num_chains=4,
        num_samples=800,
        num_warmup=400,
        rng=MersenneTwister(370),
    )

    mc_obs_addresses = observation_addresses(mc_conjugate_model, (3,), mc_constraints)
    @test length(mc_obs_addresses) == 3
    @test mc_obs_addresses == Any[(:y, 1), (:y, 2), (:y, 3)]

    mc_ll = pointwise_loglikelihood(mc_conjugate_model, (3,), mc_constraints, mc_chains)
    @test size(mc_ll, 2) == 3
    @test size(mc_ll, 1) == 4 * 800

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

    # issue #82: PSIS smoothing exponentiates relative to the largest log
    # ratio, so a large common offset must change neither the normalized
    # weights nor the fitted Pareto k (previously exp overflow made k NaN).
    mc_shift_x = collect(range(-2.0, 2.0; length=100))
    mc_w_base, mc_k_base = UncertainTea._psis_smooth(mc_shift_x)
    @test isfinite(mc_k_base)
    for mc_offset in (1000.0, -1000.0, 700.0)
        mc_w_shift, mc_k_shift = UncertainTea._psis_smooth(mc_shift_x .+ mc_offset)
        @test maximum(abs.(mc_w_base .- mc_w_shift)) < 1e-10
        @test mc_k_shift ≈ mc_k_base atol=1e-10
        @test !isnan(mc_k_shift)
    end

    # ... and PSIS-LOO's elpd shifts by exactly the log-likelihood offset.
    mc_shift_ll = reshape(-mc_shift_x, 100, 1)
    mc_loo_shifted = psis_loo(mc_shift_ll .- 1000)
    mc_loo_unshifted = psis_loo(mc_shift_ll)
    @test mc_loo_shifted.elpd - mc_loo_unshifted.elpd ≈ -1000.0 atol=1e-8
    @test mc_loo_shifted.pareto_k ≈ mc_loo_unshifted.pareto_k atol=1e-10
end
