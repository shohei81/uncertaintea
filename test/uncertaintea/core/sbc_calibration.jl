# Simulation-based calibration harness (issue #18). Fast seeded smoke runs
# only -- release-grade validation lives in bench/sbc_validation.jl.

@tea static function sbc_conjugate_model()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

@tea static function sbc_scale_model()
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 0.5)
    {:y} ~ normal(mu, sigma)
    return mu
end

@testset "sbc_calibration" begin
    @testset "sbc_uniformity_checker" begin
        # near-uniform ranks over 0:24 -> comfortable p-value
        uniform_ranks = repeat(0:24, 4)
        @test UncertainTea._sbc_uniformity_pvalue(uniform_ranks, 24, 5) > 0.5
        # all mass on one rank -> vanishing p-value
        @test UncertainTea._sbc_uniformity_pvalue(fill(2, 100), 24, 5) < 1e-12
    end

    @testset "sbc_conjugate_passes" begin
        result = sbc(
            sbc_conjugate_model;
            num_simulations=60,
            num_posterior_draws=24,
            num_warmup=40,
            rng=MersenneTwister(42),
        )
        @test size(result.ranks) == (1, 60)
        @test all(0 .<= result.ranks .<= 24)
        @test result.num_posterior_draws == 24
        @test length(result.parameter_names) == 1
        @test occursin("mu", result.parameter_names[1])
        @test result.pvalues[1] > 0.01
        @test !has_warnings(result)
        shown = repr(MIME"text/plain"(), result)
        @test occursin("SBCResult", shown)
        @test occursin("no warnings", shown)
    end

    @testset "sbc_transformed_parameter" begin
        result = sbc(
            sbc_scale_model;
            num_simulations=50,
            num_posterior_draws=20,
            num_warmup=50,
            thin=2,
            rng=MersenneTwister(7),
        )
        @test size(result.ranks) == (2, 50)
        @test all(0 .<= result.ranks .<= 20)
        @test all(result.pvalues .> 0.01)
        @test !has_warnings(result)
    end

    @testset "sbc_detects_broken_sampler" begin
        # no adaptation and an absurd step size freeze the chain, so the true
        # parameter's rank concentrates at the extremes
        result = sbc(
            sbc_conjugate_model;
            num_simulations=60,
            num_posterior_draws=24,
            num_warmup=0,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            step_size=25.0,
            rng=MersenneTwister(42),
        )
        @test result.pvalues[1] < 1e-12
        @test has_warnings(result)
        @test occursin("deviates from uniform", result.warnings[1])
    end

    @testset "sbc_explicit_observations_and_errors" begin
        explicit = sbc(
            sbc_conjugate_model;
            num_simulations=12,
            num_posterior_draws=12,
            num_warmup=30,
            observation_addresses=[(:y,)],
            rng=MersenneTwister(3),
        )
        @test size(explicit.ranks) == (1, 12)

        @test_throws ArgumentError sbc(
            sbc_conjugate_model;
            num_simulations=0,
            num_posterior_draws=8,
            num_warmup=10,
        )
        @test_throws ArgumentError sbc(
            sbc_conjugate_model;
            num_simulations=4,
            num_posterior_draws=8,
            num_warmup=10,
            thin=0,
        )
    end
end
