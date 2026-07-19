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
    @test mvnormal_smc.stages[1].move_kernel == :random_walk
    @test mvnormal_smc.stages[1].move_steps == 0
    @test mvnormal_smc.stages[1].move_acceptance_rate == 0.0
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
        move_steps=2,
        move_scale=0.05,
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
    for (previous_stage, next_stage) in zip(dirichlet_smc.stages[1:(end-1)], dirichlet_smc.stages[2:end])
        @test previous_stage.beta_end <= next_stage.beta_start + 1e-12
        @test next_stage.beta_end >= next_stage.beta_start
    end
    for stage in dirichlet_smc.stages[1:(end-1)]
        @test stage.resampled
        @test stage.move_kernel == :random_walk
        @test stage.effective_sample_size >= 0.95 * 24 - 1e-4
        @test stage.move_steps == 2
        @test 0.0 <= stage.move_acceptance_rate <= 1.0
    end
    @test any(stage.move_acceptance_rate > 0.0 for stage in dirichlet_smc.stages if stage.move_steps > 0)
    for ancestors in dirichlet_smc.ancestor_history
        @test length(ancestors) == 24
        @test all(1 .<= ancestors .<= 24)
    end
    for particle_index = 1:size(dirichlet_smc.importance.constrained_particles, 2)
        @test all(>(0.0), dirichlet_smc.importance.constrained_particles[:, particle_index])
        @test sum(dirichlet_smc.importance.constrained_particles[:, particle_index]) ≈ 1.0 atol=1e-6
    end

    dirichlet_hmc_smc = batched_smc(
        dirichlet_smc_model,
        (),
        choicemap();
        num_particles=16,
        proposal_loc=Float64[-1.0, 0.6],
        proposal_log_scale=fill(0.15, 2),
        target_ess_ratio=0.95,
        move_kernel=:hmc,
        move_steps=1,
        move_step_size=0.02,
        move_num_leapfrog_steps=2,
        move_inverse_mass_matrix=fill(1.0, 2),
        rng=MersenneTwister(212),
    )

    @test dirichlet_hmc_smc isa SMCResult
    @test numstages(dirichlet_hmc_smc) >= 2
    @test last(dirichlet_hmc_smc.stages).beta_end ≈ 1.0 atol=1e-6
    @test length(dirichlet_hmc_smc.ancestor_history) == count(stage -> stage.resampled, dirichlet_hmc_smc.stages)
    for stage in dirichlet_hmc_smc.stages[1:(end-1)]
        @test stage.move_kernel == :hmc
        @test stage.move_steps == 1
        @test 0.0 <= stage.move_acceptance_rate <= 1.0
    end
    @test any(stage.move_acceptance_rate > 0.0 for stage in dirichlet_hmc_smc.stages if stage.move_steps > 0)
    for particle_index = 1:size(dirichlet_hmc_smc.importance.constrained_particles, 2)
        @test all(>(0.0), dirichlet_hmc_smc.importance.constrained_particles[:, particle_index])
        @test sum(dirichlet_hmc_smc.importance.constrained_particles[:, particle_index]) ≈ 1.0 atol=1e-6
    end

    dirichlet_nuts_smc = batched_smc(
        dirichlet_smc_model,
        (),
        choicemap();
        num_particles=16,
        proposal_loc=Float64[-1.0, 0.6],
        proposal_log_scale=fill(0.15, 2),
        target_ess_ratio=0.95,
        move_kernel=:nuts,
        move_steps=1,
        move_step_size=0.02,
        move_max_tree_depth=2,
        move_max_delta_energy=1000.0,
        move_inverse_mass_matrix=fill(1.0, 2),
        rng=MersenneTwister(213),
    )

    @test dirichlet_nuts_smc isa SMCResult
    @test numstages(dirichlet_nuts_smc) >= 2
    @test last(dirichlet_nuts_smc.stages).beta_end ≈ 1.0 atol=1e-6
    @test length(dirichlet_nuts_smc.ancestor_history) == count(stage -> stage.resampled, dirichlet_nuts_smc.stages)
    for stage in dirichlet_nuts_smc.stages[1:(end-1)]
        @test stage.move_kernel == :nuts
        @test stage.move_steps == 1
        @test 0.0 <= stage.move_acceptance_rate <= 1.0
    end
    @test any(stage.move_acceptance_rate > 0.0 for stage in dirichlet_nuts_smc.stages if stage.move_steps > 0)
    for particle_index = 1:size(dirichlet_nuts_smc.importance.constrained_particles, 2)
        @test all(>(0.0), dirichlet_nuts_smc.importance.constrained_particles[:, particle_index])
        @test sum(dirichlet_nuts_smc.importance.constrained_particles[:, particle_index]) ≈ 1.0 atol=1e-6
    end

    dirichlet_nuts_one_step_smc = batched_smc(
        dirichlet_smc_model,
        (),
        choicemap();
        num_particles=12,
        proposal_loc=Float64[-1.0, 0.6],
        proposal_log_scale=fill(0.15, 2),
        target_ess_ratio=0.95,
        move_kernel=:nuts,
        move_steps=1,
        move_step_size=0.02,
        move_max_tree_depth=1,
        move_max_delta_energy=1000.0,
        move_inverse_mass_matrix=fill(1.0, 2),
        rng=MersenneTwister(214),
    )

    @test dirichlet_nuts_one_step_smc isa SMCResult
    @test numstages(dirichlet_nuts_one_step_smc) >= 2
    @test last(dirichlet_nuts_one_step_smc.stages).beta_end ≈ 1.0 atol=1e-6
    @test any(
        stage.move_acceptance_rate > 0.0 for
        stage in dirichlet_nuts_one_step_smc.stages if stage.move_steps > 0
    )
    for stage in dirichlet_nuts_one_step_smc.stages[1:(end-1)]
        @test stage.move_kernel == :nuts
        @test stage.move_steps == 1
        @test 0.0 <= stage.move_acceptance_rate <= 1.0
    end
    for particle_index = 1:size(dirichlet_nuts_one_step_smc.importance.constrained_particles, 2)
        @test all(>(0.0), dirichlet_nuts_one_step_smc.importance.constrained_particles[:, particle_index])
        @test sum(dirichlet_nuts_one_step_smc.importance.constrained_particles[:, particle_index]) ≈ 1.0 atol=1e-6
    end

    @test UncertainTea._tempered_nuts_active_depth(
        [
            UncertainTea.NUTSContinuationState(
                UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
                UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
                UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
                Inf,
                Inf,
                -Inf,
                0.0,
                0,
                0,
                1,
                false,
                false,
            ),
            UncertainTea.NUTSContinuationState(
                UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
                UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
                UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
                Inf,
                Inf,
                -Inf,
                0.0,
                0,
                0,
                2,
                false,
                false,
            ),
            UncertainTea.NUTSContinuationState(
                UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
                UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
                UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
                Inf,
                Inf,
                -Inf,
                0.0,
                0,
                0,
                2,
                false,
                false,
            ),
        ],
        4,
    ) == (2, 2)

    scheduler_workspace = UncertainTea.TemperedNUTSMoveWorkspace(
        dirichlet_smc_model,
        randn(MersenneTwister(2160), 2, 3),
        (),
        choicemap(),
    )
    scheduler_continuations = [
        UncertainTea.NUTSContinuationState(
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            1,
            false,
            false,
        ),
        UncertainTea.NUTSContinuationState(
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            2,
            false,
            false,
        ),
        UncertainTea.NUTSContinuationState(
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            2,
            false,
            false,
        ),
    ]
    @test UncertainTea._select_tempered_nuts_depth_cohort!(
        scheduler_workspace,
        scheduler_continuations,
        4,
    ) == (2, 2)
    @test scheduler_workspace.scheduler.continuation_active == BitVector([true, true, true])
    @test UncertainTea._activate_tempered_nuts_depth_cohort!(
        scheduler_workspace,
        scheduler_continuations,
        2,
    ) == BitVector([false, true, true])
    @test UncertainTea._begin_tempered_nuts_cohort_scheduler!(
        scheduler_workspace,
        scheduler_continuations,
        4,
        MersenneTwister(2161),
    ) == UncertainTea.TemperedNUTSSchedulerExpand
    scheduler_ir = UncertainTea._tempered_nuts_scheduler_ir(scheduler_workspace)
    @test scheduler_ir isa UncertainTea.TemperedNUTSExpandIR
    @test scheduler_ir.active_depth == 2
    @test scheduler_ir.active_depth_count == 2
    @test scheduler_ir.remaining_steps == 4
    @test scheduler_ir.active_particles == BitVector([false, true, true])
    @test length(scheduler_ir.directions) == 3
    scheduler_block = UncertainTea._tempered_nuts_scheduler_block(scheduler_workspace)
    @test scheduler_block isa UncertainTea.TemperedNUTSExpandBlock
    scheduler_descriptor = UncertainTea._tempered_nuts_scheduler_descriptor(scheduler_block)
    @test scheduler_descriptor isa UncertainTea.TemperedNUTSExpandDescriptor
    @test scheduler_descriptor.remaining_steps == 4
    @test scheduler_descriptor.active_particles == BitVector([false, true, true])
    scheduler_continuations[1].turning = true
    scheduler_continuations[2].turning = true
    scheduler_continuations[3].turning = true
    @test UncertainTea._begin_tempered_nuts_cohort_scheduler!(
        scheduler_workspace,
        scheduler_continuations,
        4,
        MersenneTwister(2162),
    ) == UncertainTea.TemperedNUTSSchedulerDone
    @test UncertainTea._tempered_nuts_scheduler_ir(scheduler_workspace) isa UncertainTea.TemperedNUTSDoneIR
    @test UncertainTea._tempered_nuts_scheduler_block(scheduler_workspace) isa UncertainTea.TemperedNUTSDoneBlock
    @test UncertainTea._tempered_nuts_scheduler_descriptor(
        UncertainTea._tempered_nuts_scheduler_block(scheduler_workspace),
    ) isa UncertainTea.TemperedNUTSDoneDescriptor

    workspace_particles = randn(MersenneTwister(216), 2, 8)
    workspace_noise = similar(workspace_particles)
    workspace_logproposal = Vector{Float64}(undef, 8)
    UncertainTea._gaussian_logdensity_from_particles!(
        workspace_logproposal,
        workspace_particles,
        Float64[-1.0, 0.6],
        fill(0.15, 2),
        workspace_noise,
    )
    workspace_logjoint = batched_logjoint_unconstrained(
        dirichlet_smc_model,
        workspace_particles,
        (),
        choicemap(),
    )
    workspace_logratio = workspace_logjoint .- workspace_logproposal
    nuts_workspace = UncertainTea.TemperedNUTSMoveWorkspace(
        dirichlet_smc_model,
        workspace_particles,
        (),
        choicemap(),
    )
    @test size(nuts_workspace.proposal_particles) == (2, 8)
    @test length(nuts_workspace.tree_workspaces) == 8
    first_workspace_acceptance = UncertainTea._batched_nuts_move!(
        nuts_workspace,
        workspace_particles,
        workspace_logjoint,
        workspace_logproposal,
        workspace_logratio,
        dirichlet_smc_model,
        (),
        choicemap(),
        Float64[-1.0, 0.6],
        fill(0.15, 2),
        0.5,
        0.02,
        2,
        1000.0,
        fill(1.0, 2),
        MersenneTwister(217),
    )
    second_workspace_acceptance = UncertainTea._batched_nuts_move!(
        nuts_workspace,
        workspace_particles,
        workspace_logjoint,
        workspace_logproposal,
        workspace_logratio,
        dirichlet_smc_model,
        (),
        choicemap(),
        Float64[-1.0, 0.6],
        fill(0.15, 2),
        0.5,
        0.02,
        2,
        1000.0,
        fill(1.0, 2),
        MersenneTwister(218),
    )
    @test 0.0 <= first_workspace_acceptance <= 1.0
    @test 0.0 <= second_workspace_acceptance <= 1.0

    dirichlet_nuts_deep_smc = batched_smc(
        dirichlet_smc_model,
        (),
        choicemap();
        num_particles=16,
        proposal_loc=Float64[-1.0, 0.6],
        proposal_log_scale=fill(0.15, 2),
        target_ess_ratio=0.95,
        move_kernel=:nuts,
        move_steps=1,
        move_step_size=0.02,
        move_max_tree_depth=3,
        move_max_delta_energy=1000.0,
        move_inverse_mass_matrix=fill(1.0, 2),
        rng=MersenneTwister(215),
    )

    @test dirichlet_nuts_deep_smc isa SMCResult
    @test numstages(dirichlet_nuts_deep_smc) >= 2
    @test last(dirichlet_nuts_deep_smc.stages).beta_end ≈ 1.0 atol=1e-6
    @test any(stage.move_acceptance_rate > 0.0 for stage in dirichlet_nuts_deep_smc.stages if stage.move_steps > 0)
    for stage in dirichlet_nuts_deep_smc.stages[1:(end-1)]
        @test stage.move_kernel == :nuts
        @test stage.move_steps == 1
        @test 0.0 <= stage.move_acceptance_rate <= 1.0
    end

    # Tempered target gradient regression (issue #94): the tempered gradient
    # must combine the freshly computed model gradient, not an undef buffer.
    @tea static function smc_conjugate_nn_model()
        mu ~ normal(0.5, 1.2)
        {:y => 1} ~ normal(mu, 0.7)
        {:y => 2} ~ normal(mu, 0.7)
        {:y => 3} ~ normal(mu, 0.7)
        {:y => 4} ~ normal(mu, 0.7)
        return mu
    end

    nn_y = [1.3, 0.2, 0.9, 1.8]
    nn_constraints =
        choicemap((:y => 1, nn_y[1]), (:y => 2, nn_y[2]), (:y => 3, nn_y[3]), (:y => 4, nn_y[4]))
    nn_particles = reshape([-1.0, 0.0, 1.0, 2.0], 1, 4)
    nn_cache = UncertainTea.BatchedLogjointGradientCache(smc_conjugate_nn_model, nn_particles, (), nn_constraints)
    nn_proposal_loc = [0.0]
    nn_proposal_log_scale = [0.0]
    nn_model_gradient =
        copy(UncertainTea.batched_logjoint_gradient_unconstrained(nn_cache, nn_particles))
    nn_proposal_gradient = similar(nn_model_gradient)
    UncertainTea._gaussian_gradient_from_particles!(
        nn_proposal_gradient,
        nn_particles,
        nn_proposal_loc,
        nn_proposal_log_scale,
    )

    for beta in (1.0, 0.5)
        nn_target = UncertainTea.BatchedTemperedDensityTarget(
            smc_conjugate_nn_model,
            (),
            nn_constraints,
            nn_cache,
            nn_proposal_loc,
            nn_proposal_log_scale,
            beta,
            1,
            4,
        )
        nn_values = zeros(4)
        nn_gradient = zeros(1, 4)
        UncertainTea.batched_target_logdensity_and_gradient!(nn_values, nn_gradient, nn_target, nn_particles)
        @test nn_gradient ≈ beta .* nn_model_gradient .+ (1.0 - beta) .* nn_proposal_gradient atol=1e-12
    end

    # With the true tempered force, a well-tuned beta = 0.5 HMC move kernel on
    # the conjugate model accepts nearly always (was ~0.01 with the undef
    # gradient, issue #94).
    nn_move_rng = MersenneTwister(219)
    nn_move_particles = 0.5 .+ 1.2 .* randn(nn_move_rng, 1, 256)
    nn_move_logjoint = Vector{Float64}(undef, 256)
    copyto!(
        nn_move_logjoint,
        batched_logjoint_unconstrained(smc_conjugate_nn_model, nn_move_particles, (), nn_constraints),
    )
    nn_move_noise = similar(nn_move_particles)
    nn_move_logproposal = Vector{Float64}(undef, 256)
    nn_move_loc = [0.5]
    nn_move_log_scale = [log(1.2)]
    UncertainTea._gaussian_logdensity_from_particles!(
        nn_move_logproposal,
        nn_move_particles,
        nn_move_loc,
        nn_move_log_scale,
        nn_move_noise,
    )
    nn_move_log_ratio = nn_move_logjoint .- nn_move_logproposal
    nn_move_acceptance = UncertainTea._batched_hmc_move!(
        nn_move_particles,
        nn_move_logjoint,
        nn_move_logproposal,
        nn_move_log_ratio,
        smc_conjugate_nn_model,
        (),
        nn_constraints,
        nn_move_loc,
        nn_move_log_scale,
        0.5,
        0.1,
        8,
        [1.0],
        nn_move_rng,
    )
    @test nn_move_acceptance > 0.9

    # Tempered SMC with HMC moves recovers the closed-form log evidence of the
    # conjugate normal-normal model (issue #94 left log evidence noisy and the
    # move kernel paralyzed).
    nn_prior_var = 1.2^2
    nn_obs_var = 0.7^2
    nn_count = length(nn_y)
    nn_mean = sum(nn_y) / nn_count
    nn_ss = sum((nn_y .- nn_mean) .^ 2)
    nn_marginal_var = nn_prior_var + nn_obs_var / nn_count
    nn_true_log_evidence =
        -nn_count / 2 * log(2pi * nn_obs_var) - nn_ss / (2 * nn_obs_var) +
        0.5 * log(2pi * nn_obs_var / nn_count) - 0.5 * log(2pi * nn_marginal_var) -
        (nn_mean - 0.5)^2 / (2 * nn_marginal_var)
    nn_smc = batched_smc(
        smc_conjugate_nn_model,
        (),
        nn_constraints;
        num_particles=512,
        proposal_loc=[0.5],
        proposal_log_scale=log(1.2),
        target_ess_ratio=0.7,
        max_stages=64,
        move_kernel=:hmc,
        move_steps=2,
        move_step_size=0.1,
        move_num_leapfrog_steps=8,
        rng=MersenneTwister(220),
    )
    @test nn_smc.importance.log_evidence_estimate ≈ nn_true_log_evidence atol=0.1
    for stage in nn_smc.stages
        stage.move_steps > 0 || continue
        @test stage.move_acceptance_rate > 0.8
    end

    # batched_smc honors callback_every (issue #91): the callback fires on
    # stage multiples of callback_every plus the final stage.
    smc_callback_events = NamedTuple[]
    smc_callback_result = batched_smc(
        smc_conjugate_nn_model,
        (),
        nn_constraints;
        num_particles=128,
        proposal_loc=[0.0],
        proposal_log_scale=log(3.0),
        target_ess_ratio=0.9,
        max_stages=64,
        callback=info -> push!(smc_callback_events, info),
        callback_every=3,
        rng=MersenneTwister(221),
    )
    smc_callback_stages = numstages(smc_callback_result)
    smc_expected_iterations = sort(unique(vcat(collect(3:3:smc_callback_stages), smc_callback_stages)))
    @test [event.iteration for event in smc_callback_events] == smc_expected_iterations
    @test all(event.phase === :stage for event in smc_callback_events)

    smc_every_stage_events = NamedTuple[]
    batched_smc(
        smc_conjugate_nn_model,
        (),
        nn_constraints;
        num_particles=128,
        proposal_loc=[0.0],
        proposal_log_scale=log(3.0),
        target_ess_ratio=0.9,
        max_stages=64,
        callback=info -> push!(smc_every_stage_events, info),
        callback_every=1,
        rng=MersenneTwister(221),
    )
    @test [event.iteration for event in smc_every_stage_events] == collect(1:smc_callback_stages)
end
