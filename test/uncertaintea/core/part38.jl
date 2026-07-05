    # PR 38: MAP estimation and Laplace approximation.
    # Contract: map_estimate maximizes logjoint_unconstrained via a self-contained
    # L-BFGS with backtracking Armijo line search; laplace_approximation forms the
    # Gaussian covariance inv(-H) at the mode and can draw unconstrained samples.

    # Exact Gaussian: mu ~ N(0,1); y | mu ~ N(mu,1) with y=0.3 has posterior mode
    # mu = y/2 = 0.15 and posterior variance 0.5 in the (already unconstrained)
    # latent space.
    @tea static function map_gaussian_model()
        mu ~ normal(0f0, 1f0)
        {:y} ~ normal(mu, 1f0)
    end

    map_gaussian_constraints = choicemap((:y, 0.3f0))
    map_gaussian_result = map_estimate(map_gaussian_model, (), map_gaussian_constraints)
    @test map_gaussian_result.converged
    @test map_gaussian_result.unconstrained_mode ≈ [0.15] atol=1e-6
    @test map_gaussian_result.constrained_mode ≈ [0.15] atol=1e-6
    @test map_gaussian_result.logjoint ≈ logjoint_unconstrained(
        map_gaussian_model,
        map_gaussian_result.unconstrained_mode,
        (),
        map_gaussian_constraints,
    ) atol=1e-10

    map_gaussian_laplace = laplace_approximation(map_gaussian_model, (), map_gaussian_constraints)
    @test map_gaussian_laplace.covariance ≈ [0.5;;] atol=1e-6
    @test map_gaussian_laplace.map.unconstrained_mode ≈ [0.15] atol=1e-6

    # Transformed latent: s ~ LogNormal(0, 0.5); y | s ~ N(0, s). The latent is
    # sampled in unconstrained (log) space; MAP should converge there and yield a
    # strictly positive constrained mode.
    @tea static function map_lognormal_model()
        s ~ lognormal(0f0, 0.5f0)
        {:y} ~ normal(0f0, s)
    end

    map_lognormal_constraints = choicemap((:y, 1.2f0))
    map_lognormal_result = map_estimate(map_lognormal_model, (), map_lognormal_constraints)
    @test map_lognormal_result.converged
    @test map_lognormal_result.gradient_norm < 1e-6
    @test map_lognormal_result.constrained_mode[1] > 0

    # Multi-parameter convergence from a poor initialization.
    @tea static function map_two_latent_model()
        a ~ normal(0f0, 1f0)
        b ~ normal(0f0, 1f0)
        {:y1} ~ normal(a, 1f0)
        {:y2} ~ normal(b, 1f0)
    end

    map_two_latent_constraints = choicemap((:y1, 2.0f0), (:y2, -1.0f0))
    map_two_latent_result = map_estimate(
        map_two_latent_model,
        (),
        map_two_latent_constraints;
        init=[5.0, -5.0],
    )
    @test map_two_latent_result.converged
    @test map_two_latent_result.gradient_norm < 1e-6
    @test map_two_latent_result.unconstrained_mode ≈ [1.0, -0.5] atol=1e-6

    # Laplace rand: seeded draws from the exact-Gaussian case reproduce the
    # posterior mean (0.15) and variance (0.5).
    map_rand_draws = rand(MersenneTwister(20240705), map_gaussian_laplace, 5000)
    @test size(map_rand_draws) == (1, 5000)
    map_rand_mean = sum(map_rand_draws) / length(map_rand_draws)
    map_rand_var =
        sum((map_rand_draws .- map_rand_mean) .^ 2) / (length(map_rand_draws) - 1)
    @test map_rand_mean ≈ 0.15 atol=0.05
    @test map_rand_var ≈ 0.5 rtol=0.2

    # End-to-end: use the MAP mode as the NUTS initialization and confirm sampling
    # produces finite draws.
    map_nuts_chain = nuts(
        map_gaussian_model,
        (),
        map_gaussian_constraints;
        num_samples=50,
        num_warmup=50,
        initial_params=map_gaussian_result.unconstrained_mode,
        rng=MersenneTwister(7),
    )
    @test all(isfinite, map_nuts_chain.unconstrained_samples)
    @test all(isfinite, map_nuts_chain.constrained_samples)
    @test size(map_nuts_chain.unconstrained_samples, 2) == 50
