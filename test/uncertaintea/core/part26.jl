@testset "Tempered Batched SMC" begin
    @tea static function mvnormal_smc_model()
        state ~ mvnormal([0.0f0, 1.0f0], [1.5f0, 0.8f0])
        return state
    end

    @tea static function dirichlet_smc_model()
        weights ~ dirichlet([2.0f0, 3.0f0, 4.0f0])
        return weights
    end

    mvnormal_smc = batched_smc(
        mvnormal_smc_model,
        (),
        choicemap();
        num_particles=20,
        proposal_loc=Float64[0.0, 1.0],
        proposal_log_scale=log.([1.5, 0.8]),
        target_ess_ratio=0.9,
        rng=MersenneTwister(210),
    )

    @test mvnormal_smc isa SMCResult
    @test numstages(mvnormal_smc) == 1
    @test mvnormal_smc.stages[1].beta_start == 0.0
    @test mvnormal_smc.stages[1].beta_end ≈ 1.0 atol=1e-8
    @test !mvnormal_smc.stages[1].resampled
    @test mvnormal_smc.importance.log_evidence_estimate ≈ 0.0 atol=1e-8
    @test ess(mvnormal_smc) ≈ 20.0 atol=1e-8
    @test sum(mvnormal_smc.importance.normalized_weights) ≈ 1.0 atol=1e-8
    @test isempty(mvnormal_smc.ancestor_history)

    dirichlet_smc = batched_smc(
        dirichlet_smc_model,
        (),
        choicemap();
        num_particles=24,
        proposal_loc=Float64[-1.2, 0.8],
        proposal_log_scale=fill(0.2, 2),
        target_ess_ratio=0.95,
        rng=MersenneTwister(211),
    )

    @test dirichlet_smc isa SMCResult
    @test numstages(dirichlet_smc) >= 2
    @test dirichlet_smc.importance.evaluation_backend == :backend_native
    @test sum(dirichlet_smc.importance.normalized_weights) ≈ 1.0 atol=1e-8
    @test 0.0 < ess(dirichlet_smc) <= 24.0
    @test length(dirichlet_smc.ancestor_history) == count(stage -> stage.resampled, dirichlet_smc.stages)
    @test first(dirichlet_smc.stages).beta_start == 0.0
    @test last(dirichlet_smc.stages).beta_end ≈ 1.0 atol=1e-6
    for (previous_stage, next_stage) in zip(dirichlet_smc.stages[1:end-1], dirichlet_smc.stages[2:end])
        @test previous_stage.beta_end <= next_stage.beta_start + 1e-12
        @test next_stage.beta_end >= next_stage.beta_start
    end
    for stage in dirichlet_smc.stages[1:end-1]
        @test stage.resampled
        @test stage.effective_sample_size >= 0.95 * 24 - 1e-4
    end
    for ancestors in dirichlet_smc.ancestor_history
        @test length(ancestors) == 24
        @test all(1 .<= ancestors .<= 24)
    end
    for particle_index in 1:size(dirichlet_smc.importance.constrained_particles, 2)
        @test all(>(0.0), dirichlet_smc.importance.constrained_particles[:, particle_index])
        @test sum(dirichlet_smc.importance.constrained_particles[:, particle_index]) ≈ 1.0 atol=1e-6
    end
end
