@testset "Vector Backend Sampler Fast Paths" begin
    @tea static function mvnormal_batched_model()
        state ~ mvnormal([0.0f0, 1.0f0], [1.5f0, 0.8f0])
        return state
    end

    @tea static function dirichlet_batched_model()
        weights ~ dirichlet([2.0f0, 3.0f0, 4.0f0])
        return weights
    end

    mvnormal_trace, _ = generate(mvnormal_batched_model, (), choicemap(); rng=MersenneTwister(190))
    dirichlet_trace, _ = generate(dirichlet_batched_model, (), choicemap(); rng=MersenneTwister(191))

    mvnormal_batch_params = hcat(
        transform_to_unconstrained(mvnormal_trace),
        transform_to_unconstrained(mvnormal_trace) .+ Float64[0.1, -0.05],
    )
    dirichlet_batch_params = hcat(
        transform_to_unconstrained(dirichlet_trace),
        transform_to_unconstrained(dirichlet_trace) .+ Float64[0.08, -0.04],
    )

    mvnormal_hmc_workspace = UncertainTea.BatchedHMCWorkspace(
        mvnormal_batched_model,
        mvnormal_batch_params,
        (),
        choicemap(),
    )
    mvnormal_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        mvnormal_batched_model,
        mvnormal_batch_params,
        (),
        choicemap(),
    )
    dirichlet_hmc_workspace = UncertainTea.BatchedHMCWorkspace(
        dirichlet_batched_model,
        dirichlet_batch_params,
        (),
        choicemap(),
    )
    dirichlet_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        dirichlet_batched_model,
        dirichlet_batch_params,
        (),
        choicemap(),
    )

    @test !isnothing(mvnormal_hmc_workspace.gradient_cache.backend_cache)
    @test !isnothing(mvnormal_nuts_workspace.gradient_cache.backend_cache)
    @test !isnothing(dirichlet_hmc_workspace.gradient_cache.backend_cache)
    @test !isnothing(dirichlet_nuts_workspace.gradient_cache.backend_cache)

    mvnormal_batched_hmc = batched_hmc(
        mvnormal_batched_model,
        (),
        choicemap();
        num_chains=2,
        num_samples=6,
        num_warmup=0,
        step_size=0.05,
        num_leapfrog_steps=2,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        initial_params=mvnormal_batch_params,
        rng=MersenneTwister(192),
    )
    mvnormal_batched_nuts = batched_nuts(
        mvnormal_batched_model,
        (),
        choicemap();
        num_chains=2,
        num_samples=6,
        num_warmup=0,
        step_size=0.05,
        max_tree_depth=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        initial_params=mvnormal_batch_params,
        rng=MersenneTwister(193),
    )
    dirichlet_batched_hmc = batched_hmc(
        dirichlet_batched_model,
        (),
        choicemap();
        num_chains=2,
        num_samples=6,
        num_warmup=0,
        step_size=0.02,
        num_leapfrog_steps=2,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        initial_params=dirichlet_batch_params,
        rng=MersenneTwister(194),
    )
    dirichlet_batched_nuts = batched_nuts(
        dirichlet_batched_model,
        (),
        choicemap();
        num_chains=2,
        num_samples=6,
        num_warmup=0,
        step_size=0.02,
        max_tree_depth=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        initial_params=dirichlet_batch_params,
        rng=MersenneTwister(195),
    )

    @test nchains(mvnormal_batched_hmc) == 2
    @test nchains(mvnormal_batched_nuts) == 2
    @test nchains(dirichlet_batched_hmc) == 2
    @test nchains(dirichlet_batched_nuts) == 2
    @test size(mvnormal_batched_hmc[1].unconstrained_samples) == (2, 6)
    @test size(mvnormal_batched_nuts[1].unconstrained_samples) == (2, 6)
    @test size(dirichlet_batched_hmc[1].unconstrained_samples) == (2, 6)
    @test size(dirichlet_batched_nuts[1].unconstrained_samples) == (2, 6)
    @test size(dirichlet_batched_hmc[1].constrained_samples) == (3, 6)
    @test size(dirichlet_batched_nuts[1].constrained_samples) == (3, 6)
    for chain in dirichlet_batched_hmc
        for sample_index in 1:size(chain.constrained_samples, 2)
            @test all(>(0.0), chain.constrained_samples[:, sample_index])
            @test sum(chain.constrained_samples[:, sample_index]) ≈ 1.0 atol=1e-6
        end
    end
    for chain in dirichlet_batched_nuts
        for sample_index in 1:size(chain.constrained_samples, 2)
            @test all(>(0.0), chain.constrained_samples[:, sample_index])
            @test sum(chain.constrained_samples[:, sample_index]) ≈ 1.0 atol=1e-6
        end
    end
end
