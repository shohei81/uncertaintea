# PR 36: prior/posterior predictive sampling and SMC resampling schemes.
# Feature 1 exercises `predict` on pooled NUTS chains, weighted particle
# results, and the prior. Feature 2 exercises the resampling scheme dispatch
# plus the `log_evidence` accessor and `SMCResult` show.

@testset "predictive_sampling_smc_resampling" begin
    @tea static function pred_conjugate_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 1.0f0)
        return mu
    end

    pred_constraints = choicemap((:y, 0.3f0))

    # Posterior predictive: mu | y ~ N(0.15, 0.5), so y* ~ N(0.15, 1.5).
    pred_chains = nuts_chains(
        pred_conjugate_model,
        (),
        pred_constraints;
        num_chains=4,
        num_samples=400,
        num_warmup=400,
        rng=MersenneTwister(20260704),
    )
    pred_post = predict(pred_conjugate_model, (), pred_chains; rng=MersenneTwister(11))
    pred_post_y = Float64.(pred_post[:y])
    pred_post_mean = sum(pred_post_y) / length(pred_post_y)
    pred_post_var =
        sum((pred_post_y .- pred_post_mean) .^ 2) / (length(pred_post_y) - 1)

    @test length(pred_post) == 4 * 400
    @test isapprox(pred_post_mean, 0.15; atol=0.15)
    @test 1.0 <= pred_post_var <= 2.0
    # Posterior predictive keeps only observation/predictive addresses, not mu.
    @test addresses(pred_post) == [(:y,)]

    # Prior predictive: mu ~ N(0,1), y ~ N(mu,1), so y ~ N(0, 2).
    pred_prior = predict(pred_conjugate_model; num_draws=1000, rng=MersenneTwister(7))
    pred_prior_y = Float64.(pred_prior[:y])
    pred_prior_mean = sum(pred_prior_y) / length(pred_prior_y)

    @test length(pred_prior) == 1000
    @test isapprox(pred_prior_mean, 0.0; atol=0.15)
    # Prior predictive keeps ALL addresses.
    @test (:mu,) in addresses(pred_prior)
    @test (:y,) in addresses(pred_prior)

    # Loop-addressed observations round trip through predict.
    @tea static function pred_loop_model()
        mu ~ normal(0.0f0, 1.0f0)
        for i = 1:3
            {:y => i} ~ normal(mu, 1.0f0)
        end
        return mu
    end

    pred_loop_constraints = choicemap((:y => 1, 0.1f0), (:y => 2, -0.2f0), (:y => 3, 0.4f0))
    pred_loop_chains = nuts_chains(
        pred_loop_model,
        (),
        pred_loop_constraints;
        num_chains=2,
        num_samples=100,
        num_warmup=200,
        rng=MersenneTwister(3),
    )
    pred_loop_post = predict(pred_loop_model, (), pred_loop_chains; num_draws=50, rng=MersenneTwister(5))
    pred_loop_addrs = addresses(pred_loop_post)

    @test length(pred_loop_post) == 50
    @test length(pred_loop_addrs) == 3
    @test (:y, 1) in pred_loop_addrs
    @test (:y, 2) in pred_loop_addrs
    @test (:y, 3) in pred_loop_addrs
    @test all(isfinite, Float64.(pred_loop_post[:y=>1]))

    # Weighted predict from a batched SIR result runs and returns finite values.
    pred_sir = batched_sir(pred_conjugate_model, (), pred_constraints; num_particles=500, rng=MersenneTwister(9))
    pred_sir_post = predict(pred_conjugate_model, (), pred_sir; num_draws=200, rng=MersenneTwister(9))
    @test length(pred_sir_post) == 200
    @test all(isfinite, Float64.(pred_sir_post[:y]))

    # Weighted predict from a batched SMC result.
    pred_smc = batched_smc(pred_conjugate_model, (), pred_constraints; num_particles=200, rng=MersenneTwister(4))
    pred_smc_post = predict(pred_conjugate_model, (), pred_smc; num_draws=100, rng=MersenneTwister(2))
    @test length(pred_smc_post) == 100
    @test all(isfinite, Float64.(pred_smc_post[:y]))

    # --- Feature 2: resampling schemes ---
    resamp_weights = [0.5, 0.3, 0.2, 0.0]
    resamp_n = 1000

    for (resamp_scheme, resamp_tol) in
        ((:systematic, 0.05), (:stratified, 0.05), (:residual, 0.05), (:multinomial, 0.06))
        resamp_idx = UncertainTea._resample_indices(
            resamp_scheme,
            resamp_weights,
            resamp_n,
            MersenneTwister(123),
        )
        @test resamp_idx isa Vector{Int}
        @test length(resamp_idx) == resamp_n
        @test all(index -> 1 <= index <= 4, resamp_idx)
        # No scheme may ever select the zero-weight particle.
        @test !any(==(4), resamp_idx)
        for particle = 1:4
            resamp_share = count(==(particle), resamp_idx) / resamp_n
            @test isapprox(resamp_share, resamp_weights[particle]; atol=resamp_tol)
        end
    end

    # Residual resampling guarantees the deterministic floor for each particle.
    resamp_residual_idx = UncertainTea._resample_indices(
        :residual,
        resamp_weights,
        resamp_n,
        MersenneTwister(77),
    )
    for particle = 1:4
        @test count(==(particle), resamp_residual_idx) >=
              floor(Int, resamp_n * resamp_weights[particle])
    end

    # Unknown scheme is rejected.
    @test_throws ArgumentError UncertainTea._resample_indices(
        :bogus,
        resamp_weights,
        resamp_n,
        MersenneTwister(1),
    )

    # Resampling kwarg routes through the batched entry points.
    resamp_smc_stratified = batched_smc(
        pred_conjugate_model,
        (),
        pred_constraints;
        num_particles=200,
        resampling=:stratified,
        rng=MersenneTwister(4),
    )
    @test isfinite(log_evidence(resamp_smc_stratified))
    @test_throws ArgumentError batched_smc(
        pred_conjugate_model,
        (),
        pred_constraints;
        num_particles=50,
        resampling=:bogus,
        rng=MersenneTwister(4),
    )
    resamp_sir_multinomial = batched_sir(
        pred_conjugate_model,
        (),
        pred_constraints;
        num_particles=200,
        resampling=:multinomial,
        rng=MersenneTwister(4),
    )
    @test length(resamp_sir_multinomial.ancestors) == 200

    # log_evidence accessor and SMCResult show.
    resamp_importance = batched_importance_sampling(
        pred_conjugate_model,
        (),
        pred_constraints;
        num_particles=200,
        rng=MersenneTwister(4),
    )
    @test log_evidence(resamp_importance) == resamp_importance.log_evidence_estimate
    @test log_evidence(pred_smc) == pred_smc.importance.log_evidence_estimate
    @test occursin("SMCResult", sprint(show, pred_smc))
end
