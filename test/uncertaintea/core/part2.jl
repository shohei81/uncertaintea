    @test mod_scale_gradient ≈ hcat([
        logjoint_gradient_unconstrained(mod_scale_model, mod_scale_batch_params[:, index], (), mod_scale_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test !isnothing(mod_scale_gradient_cache.backend_cache)
    @test isnothing(mod_scale_gradient_cache.flat_cache)
    @test isempty(mod_scale_gradient_cache.column_caches)
    mod_scale_combined_values = fill(-1.0, 3)
    mod_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        mod_scale_combined_values,
        mod_scale_gradient_cache,
        mod_scale_batch_params,
    )[2]
    @test mod_scale_combined_values ≈ [
        logjoint_unconstrained(mod_scale_model, mod_scale_batch_params[:, index], (), mod_scale_batch_constraints[index]) for
        index in 1:3
    ] atol=1e-8
    @test mod_scale_combined_gradient === mod_scale_gradient_cache.gradient_buffer
    @test mod_scale_combined_gradient ≈ mod_scale_gradient atol=1e-8
    @test clamp_scale_gradient ≈ hcat([
        logjoint_gradient_unconstrained(clamp_scale_model, clamp_scale_batch_params[:, index], (), clamp_scale_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test !isnothing(clamp_scale_gradient_cache.backend_cache)
    @test isnothing(clamp_scale_gradient_cache.flat_cache)
    @test isempty(clamp_scale_gradient_cache.column_caches)
    clamp_scale_combined_values = fill(-1.0, 3)
    clamp_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        clamp_scale_combined_values,
        clamp_scale_gradient_cache,
        clamp_scale_batch_params,
    )[2]
    @test clamp_scale_combined_values ≈ [
        logjoint_unconstrained(clamp_scale_model, clamp_scale_batch_params[:, index], (), clamp_scale_batch_constraints[index]) for
        index in 1:3
    ] atol=1e-8
    @test clamp_scale_combined_gradient === clamp_scale_gradient_cache.gradient_buffer
    @test clamp_scale_combined_gradient ≈ clamp_scale_gradient atol=1e-8
    positive_workspace = UncertainTea.BatchedLogjointWorkspace(observed_positive_step)
    positive_workspace_values = UncertainTea._logjoint_unconstrained_batched_backend!(
        observed_positive_step,
        positive_workspace,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    )
    positive_workspace_constrained = positive_workspace.batched_constrained_buffer[]
    positive_workspace_logabsdet = positive_workspace.batched_logabsdet_buffer[]
    positive_workspace_observed = positive_workspace.batched_environment[].observed_values
    @test positive_workspace_values ≈ positive_batch_logjoint atol=1e-8
    @test UncertainTea._logjoint_unconstrained_batched_backend!(
        observed_positive_step,
        positive_workspace,
        positive_batch_unconstrained .+ 0.05,
        (),
        positive_batch_constraints,
    ) ≈ [
        logjoint_unconstrained(
            observed_positive_step,
            (positive_batch_unconstrained .+ 0.05)[:, index],
            (),
            positive_batch_constraints[index],
        ) for index in 1:3
    ] atol=1e-8
    @test positive_workspace.batched_constrained_buffer[] === positive_workspace_constrained
    @test positive_workspace.batched_logabsdet_buffer[] === positive_workspace_logabsdet
    @test positive_workspace.batched_environment[].observed_values === positive_workspace_observed
    positive_destination = fill(-1.0, 3)
    @test UncertainTea._batched_logjoint_unconstrained_with_workspace!(
        positive_destination,
        observed_positive_step,
        positive_workspace,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    ) === positive_destination
    @test positive_destination ≈ positive_batch_logjoint atol=1e-8
    positive_reconstrained = similar(positive_step_unconstrained)
    @test UncertainTea._transform_to_constrained!(
        positive_reconstrained,
        observed_positive_step,
        positive_step_unconstrained,
    ) === positive_reconstrained
    @test positive_reconstrained ≈ transform_to_constrained(
        observed_positive_step,
        positive_step_unconstrained,
    ) atol=1e-8
    gaussian_hmc_workspace = UncertainTea.BatchedHMCWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        gaussian_batch_constraints,
        [1.0],
    )
    UncertainTea._sample_batched_momentum!(
        gaussian_hmc_workspace.momentum,
        MersenneTwister(91),
        gaussian_hmc_workspace.sqrt_inverse_mass_matrix,
    )
    gaussian_hmc_current_logjoint = Vector{Float64}(undef, 3)
    UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        gaussian_hmc_current_logjoint,
        gaussian_hmc_workspace.gradient_cache,
        gaussian_batch_params,
    )
    gaussian_hmc_proposal = UncertainTea._batched_leapfrog!(
        gaussian_hmc_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_hmc_workspace.current_gradient,
        [1.0],
        (),
        gaussian_batch_constraints,
        0.1,
        2,
    )
    @test gaussian_hmc_proposal[1] === gaussian_hmc_workspace.proposal_position
    @test gaussian_hmc_proposal[2] === gaussian_hmc_workspace.proposal_momentum
    @test gaussian_hmc_proposal[3] === gaussian_hmc_workspace.proposed_logjoint
    @test gaussian_hmc_proposal[4] === gaussian_hmc_workspace.proposal_gradient
    @test gaussian_hmc_proposal[5] === gaussian_hmc_workspace.valid
    UncertainTea._sample_batched_momentum!(
        gaussian_hmc_workspace.momentum,
        MersenneTwister(92),
        gaussian_hmc_workspace.sqrt_inverse_mass_matrix,
    )
    gaussian_hmc_proposal_replay = UncertainTea._batched_leapfrog!(
        gaussian_hmc_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_hmc_workspace.current_gradient,
        [1.0],
        (),
        gaussian_batch_constraints,
        0.1,
        2,
    )
    @test gaussian_hmc_proposal_replay[1] === gaussian_hmc_proposal[1]
    @test gaussian_hmc_proposal_replay[2] === gaussian_hmc_proposal[2]
    @test gaussian_hmc_proposal_replay[3] === gaussian_hmc_proposal[3]
    @test gaussian_hmc_proposal_replay[4] === gaussian_hmc_proposal[4]
    @test gaussian_hmc_proposal_replay[5] === gaussian_hmc_proposal[5]
    gaussian_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        gaussian_batch_constraints,
    )
    gaussian_shared_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    gaussian_nuts_state = UncertainTea._batched_nuts_state(
        gaussian_nuts_workspace.proposal_position,
        gaussian_nuts_workspace.proposal_momentum,
        gaussian_nuts_workspace.proposed_logjoint,
        gaussian_nuts_workspace.proposal_gradient,
        1,
    )
    @test parent(gaussian_nuts_state.position) === gaussian_nuts_workspace.proposal_position
    @test parent(gaussian_nuts_state.momentum) === gaussian_nuts_workspace.proposal_momentum
    @test parent(gaussian_nuts_state.gradient) === gaussian_nuts_workspace.proposal_gradient
    @test length(gaussian_nuts_workspace.column_tree_workspaces) == 3
    @test gaussian_nuts_workspace.column_tree_workspaces[1] isa UncertainTea.NUTSSubtreeWorkspace
    @test length(gaussian_nuts_workspace.column_tree_workspaces[1].current.position) == 1
    @test length(gaussian_nuts_workspace.column_tree_workspaces[1].left.position) == 1
    @test length(gaussian_nuts_workspace.column_tree_workspaces[1].right.position) == 1
    @test length(gaussian_nuts_workspace.column_tree_workspaces[1].proposal.position) == 1
    @test gaussian_nuts_workspace.column_tree_workspaces[1].summary isa UncertainTea.NUTSSubtreeMetadataState
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].current.position) === gaussian_nuts_workspace.tree_current_position
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].current.momentum) === gaussian_nuts_workspace.tree_current_momentum
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].current.gradient) === gaussian_nuts_workspace.tree_current_gradient
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].next.position) === gaussian_nuts_workspace.tree_next_position
    @test parent(gaussian_nuts_workspace.column_gradient_caches[1].buffer) === gaussian_nuts_workspace.tree_next_gradient
    @test gaussian_shared_nuts_workspace.column_gradient_caches[1].objective ===
        gaussian_shared_nuts_workspace.column_gradient_caches[2].objective
    @test gaussian_shared_nuts_workspace.column_gradient_caches[1].config ===
        gaussian_shared_nuts_workspace.column_gradient_caches[2].config
    @test parent(gaussian_shared_nuts_workspace.column_gradient_caches[1].buffer) === gaussian_shared_nuts_workspace.tree_next_gradient
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].left.position) === gaussian_nuts_workspace.tree_left_position
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].right.position) === gaussian_nuts_workspace.tree_right_position
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].proposal.position) === gaussian_nuts_workspace.tree_proposal_position
    @test length(gaussian_nuts_workspace.column_continuation_states) == 3
    @test gaussian_nuts_workspace.column_continuation_states[1] isa UncertainTea.NUTSContinuationState
    @test length(gaussian_nuts_workspace.column_continuation_states[1].proposal.position) == 1
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].left.position) === gaussian_nuts_workspace.left_position
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].left.momentum) === gaussian_nuts_workspace.left_momentum
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].left.gradient) === gaussian_nuts_workspace.left_gradient
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].right.position) === gaussian_nuts_workspace.right_position
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].right.momentum) === gaussian_nuts_workspace.right_momentum
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].right.gradient) === gaussian_nuts_workspace.right_gradient
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].proposal.position) === gaussian_nuts_workspace.proposal_position
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].proposal.momentum) === gaussian_nuts_workspace.proposal_momentum
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].proposal.gradient) === gaussian_nuts_workspace.proposal_gradient
    @test length(gaussian_nuts_workspace.tree_current_logjoint) == 3
    @test length(gaussian_nuts_workspace.tree_left_logjoint) == 3
    @test length(gaussian_nuts_workspace.tree_right_logjoint) == 3
    @test length(gaussian_nuts_workspace.tree_proposal_logjoint) == 3
    @test length(gaussian_nuts_workspace.left_logjoint) == 3
    @test length(gaussian_nuts_workspace.right_logjoint) == 3
    @test length(gaussian_nuts_workspace.continuation_proposal_logjoint) == 3
    @test length(gaussian_nuts_workspace.continuation_proposed_energy) == 3
    @test length(gaussian_nuts_workspace.continuation_delta_energy) == 3
    @test length(gaussian_nuts_workspace.continuation_accept_prob) == 3
    @test length(gaussian_nuts_workspace.continuation_candidate_log_weight) == 3
    @test length(gaussian_nuts_workspace.continuation_combined_log_weight) == 3
    @test length(gaussian_nuts_workspace.continuation_select_proposal) == 3
    @test length(gaussian_nuts_workspace.subtree_proposed_energy) == 3
    @test length(gaussian_nuts_workspace.subtree_delta_energy) == 3
    @test length(gaussian_nuts_workspace.subtree_proposal_energy) == 3
    @test length(gaussian_nuts_workspace.subtree_proposal_energy_error) == 3
    @test length(gaussian_nuts_workspace.subtree_accept_prob) == 3
    @test length(gaussian_nuts_workspace.subtree_candidate_log_weight) == 3
    @test length(gaussian_nuts_workspace.subtree_combined_log_weight) == 3
    @test length(gaussian_nuts_workspace.subtree_merged_turning) == 3
    @test length(gaussian_nuts_workspace.subtree_copy_left) == 3
    @test length(gaussian_nuts_workspace.subtree_copy_right) == 3
    @test length(gaussian_nuts_workspace.subtree_select_proposal) == 3
    masked_destination = reshape(collect(1.0:6.0), 2, 3)
    masked_source = reshape(collect(7.0:12.0), 2, 3)
    UncertainTea._copy_masked_columns!(masked_destination, masked_source, BitVector([true, false, true]))
    @test masked_destination[:, 1] == masked_source[:, 1]
    @test masked_destination[:, 2] == [3.0, 4.0]
    @test masked_destination[:, 3] == masked_source[:, 3]
    single_chain_mask = UncertainTea._single_chain_mask!(falses(3), 2)
    @test single_chain_mask == BitVector([false, true, false])
    sampled_directions = zeros(Int, 3)
    UncertainTea._sample_batched_nuts_directions!(
        sampled_directions,
        MersenneTwister(105),
        BitVector([true, false, true]),
    )
    @test sampled_directions[1] in (-1, 1)
    @test sampled_directions[2] == 0
    @test sampled_directions[3] in (-1, 1)
    @test UncertainTea._nuts_continuation_active(1, 3, false, false)
    @test !UncertainTea._nuts_continuation_active(3, 3, false, false)
    @test !UncertainTea._nuts_continuation_active(1, 3, true, false)
    @test !UncertainTea._nuts_continuation_active(1, 3, false, true)
    @test UncertainTea._mean_acceptance_stat(3.0, 2) == 1.5
    @test UncertainTea._mean_acceptance_stat(0.0, 0) == 0.0
    moved_destination = falses(3)
    UncertainTea._batched_positions_moved!(
        moved_destination,
        reshape([1.0, 2.0, 3.0], 1, 3),
        reshape([1.0, 4.0, 3.0], 1, 3),
    )
    @test moved_destination == BitVector([false, true, false])
    @test UncertainTea._position_moved([1.0, 2.0], [1.0, 3.0])
    @test !UncertainTea._position_moved([1.0, 2.0], [1.0, 2.0])
    gaussian_copy_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        gaussian_batch_constraints,
    )
    gaussian_copy_nuts_workspace.tree_left_position[:, 1] .= 11.0
    gaussian_copy_nuts_workspace.tree_left_momentum[:, 1] .= 12.0
    gaussian_copy_nuts_workspace.tree_left_gradient[:, 1] .= 13.0
    gaussian_copy_nuts_workspace.tree_left_logjoint[1] = 1.5
    UncertainTea._copy_single_batched_continuation_frontier_from_tree!(
        gaussian_copy_nuts_workspace,
        1,
        -1,
    )
    @test view(gaussian_copy_nuts_workspace.left_position, :, 1) == [11.0]
    @test view(gaussian_copy_nuts_workspace.left_momentum, :, 1) == [12.0]
    @test view(gaussian_copy_nuts_workspace.left_gradient, :, 1) == [13.0]
    @test gaussian_copy_nuts_workspace.left_logjoint[1] == 1.5
    @test gaussian_copy_nuts_workspace.column_continuation_states[1].left.logjoint == 1.5
    gaussian_copy_nuts_workspace.tree_proposal_position[:, 1] .= 21.0
    gaussian_copy_nuts_workspace.tree_proposal_momentum[:, 1] .= 22.0
    gaussian_copy_nuts_workspace.tree_proposal_gradient[:, 1] .= 23.0
    gaussian_copy_nuts_workspace.tree_proposal_logjoint[1] = 2.5
    UncertainTea._copy_single_batched_continuation_proposal_from_tree!(
        gaussian_copy_nuts_workspace,
        1,
    )
    @test view(gaussian_copy_nuts_workspace.proposal_position, :, 1) == [21.0]
    @test view(gaussian_copy_nuts_workspace.proposal_momentum, :, 1) == [22.0]
    @test view(gaussian_copy_nuts_workspace.proposal_gradient, :, 1) == [23.0]
    @test gaussian_copy_nuts_workspace.continuation_proposal_logjoint[1] == 2.5
    @test gaussian_copy_nuts_workspace.column_continuation_states[1].proposal.logjoint == 2.5
    gaussian_copy_nuts_workspace.left_position[:, 1] .= 0.0
    gaussian_copy_nuts_workspace.right_position[:, 1] .= 1.0
    gaussian_copy_nuts_workspace.left_momentum[:, 1] .= 1.0
    gaussian_copy_nuts_workspace.right_momentum[:, 1] .= 1.0
    UncertainTea._update_single_batched_continuation_turning!(
        gaussian_copy_nuts_workspace,
        1,
    )
    @test !gaussian_copy_nuts_workspace.subtree_merged_turning[1]
    @test !any(gaussian_copy_nuts_workspace.subtree_active)
    scalar_tree_workspace = UncertainTea.NUTSSubtreeWorkspace(1)
    scalar_tree_workspace.summary.log_weight = 1.0
    scalar_tree_workspace.summary.accept_stat_sum = 2.0
    scalar_tree_workspace.summary.accept_stat_count = 3
    scalar_tree_workspace.summary.integration_steps = 4
    scalar_tree_workspace.summary.proposal_energy = 5.0
    scalar_tree_workspace.summary.proposal_energy_error = 6.0
    scalar_tree_workspace.summary.turning = true
    scalar_tree_workspace.summary.divergent = true
    UncertainTea._reset_nuts_subtree_summary!(scalar_tree_workspace.summary)
    scalar_tree_summary = UncertainTea._nuts_subtree_summary(scalar_tree_workspace.summary)
    @test scalar_tree_summary.log_weight == -Inf
    @test scalar_tree_summary.accept_stat_sum == 0.0
    @test scalar_tree_summary.accept_stat_count == 0
    @test scalar_tree_summary.integration_steps == 0
    @test !scalar_tree_summary.turning
    @test !scalar_tree_summary.divergent
    @test !isfinite(scalar_tree_workspace.summary.proposal_energy)
    @test !isfinite(scalar_tree_workspace.summary.proposal_energy_error)
    scalar_continuation = UncertainTea.NUTSContinuationState(1)
    UncertainTea._initialize_nuts_continuation!(
        scalar_continuation,
        scalar_tree_workspace.current,
        scalar_tree_workspace.current,
        scalar_tree_workspace.current,
        1.0,
        0.0,
        -Inf,
        0.0,
        0,
        0,
        1,
        false,
        false,
    )
    scalar_tree_workspace.proposal.position .= 7.0
    scalar_tree_workspace.proposal.momentum .= 8.0
    scalar_tree_workspace.proposal.gradient .= 9.0
    scalar_tree_workspace.proposal.logjoint = 0.25
    scalar_tree_workspace.left.position .= 3.0
    scalar_tree_workspace.left.momentum .= 4.0
    scalar_tree_workspace.left.gradient .= 5.0
    scalar_tree_workspace.left.logjoint = -0.5
    scalar_tree_workspace.summary.log_weight = -0.5
    scalar_tree_workspace.summary.accept_stat_sum = 0.75
    scalar_tree_workspace.summary.accept_stat_count = 2
    scalar_tree_workspace.summary.integration_steps = 3
    scalar_tree_workspace.summary.proposal_energy = 1.25
    scalar_tree_workspace.summary.proposal_energy_error = 0.25
    scalar_tree_workspace.summary.turning = false
    scalar_tree_workspace.summary.divergent = false
    UncertainTea._copy_nuts_continuation_frontier_from_tree!(
        scalar_continuation,
        scalar_tree_workspace,
        -1,
    )
    @test UncertainTea._nuts_subtree_start_state(scalar_continuation, -1) === scalar_continuation.left
    @test UncertainTea._nuts_subtree_start_state(scalar_continuation, 1) === scalar_continuation.right
    @test scalar_continuation.left.logjoint == -0.5
    @test scalar_continuation.left.position == [3.0]
    UncertainTea._copy_nuts_continuation_proposal_from_tree!(
        scalar_continuation,
        scalar_tree_workspace,
    )
    @test scalar_continuation.proposal.logjoint == 0.25
    @test scalar_continuation.proposal_energy == 1.25
    @test scalar_continuation.proposal_energy_error == 0.25
    UncertainTea._merge_nuts_subtree_summary!(
        scalar_continuation,
        scalar_tree_workspace,
        -0.5,
    )
    UncertainTea._merge_nuts_continuation_turning!(scalar_continuation, true)
    @test scalar_continuation.integration_steps == 3
    @test scalar_continuation.accept_stat_sum == 0.75
    @test scalar_continuation.accept_stat_count == 2
    @test scalar_continuation.proposal_energy == 1.25
    @test scalar_continuation.proposal_energy_error == 0.25
    @test scalar_continuation.proposal.logjoint == 0.25
    @test scalar_continuation.turning
    turning_destination = falses(3)
    @test UncertainTea._batched_is_turning!(
        turning_destination,
        reshape([0.0, 0.0, 0.0], 1, 3),
        reshape([1.0, -1.0, 2.0], 1, 3),
        reshape([1.0, 1.0, 1.0], 1, 3),
        reshape([1.0, -1.0, -1.0], 1, 3),
        BitVector([true, true, false]),
    ) == BitVector([false, true, false])
    gaussian_nuts_tree_current = gaussian_nuts_workspace.column_tree_workspaces[1].current
    gaussian_nuts_tree_next = gaussian_nuts_workspace.column_tree_workspaces[1].next
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_batch_logjoint,
        gaussian_batch_gradient,
        [1.0],
        (),
        gaussian_batch_constraints,
        0.15,
        1000.0,
        MersenneTwister(93),
    )
    @test gaussian_nuts_workspace.column_tree_workspaces[1].current === gaussian_nuts_tree_current
    @test gaussian_nuts_workspace.column_tree_workspaces[1].next === gaussian_nuts_tree_next
    @test gaussian_nuts_tree_current.position ≈ gaussian_batch_params[:, 1] atol=1e-8
    @test gaussian_nuts_tree_current.logjoint ≈ gaussian_batch_logjoint[1] atol=1e-8
    @test gaussian_nuts_tree_current.momentum ≈ view(gaussian_nuts_workspace.current_momentum, :, 1) atol=1e-8
    @test gaussian_nuts_tree_next.logjoint ≈
        logjoint_unconstrained(gaussian_mean, gaussian_nuts_tree_next.position, (), gaussian_batch_constraints[1]) atol=1e-8
    @test gaussian_nuts_workspace.column_continuation_states[1].left.position ≈ view(gaussian_nuts_workspace.left_position, :, 1) atol=1e-8
    @test gaussian_nuts_workspace.column_continuation_states[1].right.position ≈ view(gaussian_nuts_workspace.right_position, :, 1) atol=1e-8
    @test gaussian_nuts_workspace.column_continuation_states[1].proposal.position ≈ view(gaussian_nuts_workspace.proposal_position, :, 1) atol=1e-8
    @test gaussian_nuts_workspace.left_logjoint[1] ≈ gaussian_nuts_workspace.column_continuation_states[1].left.logjoint atol=1e-8
    @test gaussian_nuts_workspace.right_logjoint[1] ≈ gaussian_nuts_workspace.column_continuation_states[1].right.logjoint atol=1e-8
    @test gaussian_nuts_workspace.continuation_proposal_logjoint[1] ≈ gaussian_nuts_workspace.column_continuation_states[1].proposal.logjoint atol=1e-8
    @test gaussian_nuts_workspace.continuation_proposed_energy[1] ≈
        gaussian_nuts_workspace.column_continuation_states[1].proposal_energy atol=1e-8
    @test gaussian_nuts_workspace.continuation_delta_energy[1] ≈
        gaussian_nuts_workspace.column_continuation_states[1].proposal_energy_error atol=1e-8
    gaussian_nuts_summary = UncertainTea._nuts_proposal_summary(
        gaussian_nuts_workspace.column_continuation_states[1],
        gaussian_batch_params[:, 1],
    )
    @test gaussian_nuts_summary[2] ≈
        gaussian_nuts_workspace.column_continuation_states[1].proposal_energy atol=1e-8
    @test gaussian_nuts_summary[3] ≈
        gaussian_nuts_workspace.column_continuation_states[1].proposal_energy_error atol=1e-8
    gaussian_nuts_workspace.control.step_direction .= [1, -1, 1]
    subtree_active = BitVector([true, true, false])
    UncertainTea._initialize_batched_nuts_subtree_states!(gaussian_nuts_workspace, subtree_active)
    @test gaussian_nuts_workspace.subtree_copy_left == BitVector([false, true, false])
    @test gaussian_nuts_workspace.subtree_copy_right == BitVector([true, false, false])
    @test view(gaussian_nuts_workspace.tree_current_position, :, 1) ≈ view(gaussian_nuts_workspace.right_position, :, 1)
    @test view(gaussian_nuts_workspace.tree_current_position, :, 2) ≈ view(gaussian_nuts_workspace.left_position, :, 2)
    @test view(gaussian_nuts_workspace.tree_proposal_position, :, 1) ≈ view(gaussian_nuts_workspace.right_position, :, 1)
    @test view(gaussian_nuts_workspace.tree_proposal_position, :, 2) ≈ view(gaussian_nuts_workspace.left_position, :, 2)
    @test gaussian_nuts_workspace.tree_current_logjoint[1] ≈ gaussian_nuts_workspace.right_logjoint[1] atol=1e-8
    @test gaussian_nuts_workspace.tree_current_logjoint[2] ≈ gaussian_nuts_workspace.left_logjoint[2] atol=1e-8
    @test gaussian_nuts_workspace.tree_proposal_logjoint[1] ≈ gaussian_nuts_workspace.right_logjoint[1] atol=1e-8
    @test gaussian_nuts_workspace.tree_proposal_logjoint[2] ≈ gaussian_nuts_workspace.left_logjoint[2] atol=1e-8
    @test gaussian_nuts_workspace.control.tree_depths[1] == 1
    @test gaussian_nuts_workspace.control.integration_steps[1] in 0:1
    @test isfinite(gaussian_nuts_workspace.continuation_log_weight[1])
    @test isfinite(gaussian_nuts_workspace.continuation_proposed_energy[1])
    @test isfinite(gaussian_nuts_workspace.continuation_delta_energy[1])
    @test 0.0 <= gaussian_nuts_workspace.continuation_accept_prob[1] <= 1.0
    @test isfinite(gaussian_nuts_workspace.continuation_candidate_log_weight[1])
    @test isfinite(gaussian_nuts_workspace.continuation_combined_log_weight[1])
    @test gaussian_nuts_workspace.continuation_accept_stat_count[1] in 0:1
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_shared_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.05,
        1000.0,
        MersenneTwister(94),
    )
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_shared_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.05,
        2,
        1000.0,
        MersenneTwister(95),
    )
    @test all(depth == 2 for depth in gaussian_shared_nuts_workspace.control.tree_depths)
    @test all(steps >= 2 for steps in gaussian_shared_nuts_workspace.control.integration_steps)
    @test all(isfinite, gaussian_shared_nuts_workspace.continuation_log_weight)
    @test all(isfinite, gaussian_shared_nuts_workspace.continuation_candidate_log_weight)
    @test all(isfinite, gaussian_shared_nuts_workspace.continuation_combined_log_weight)
    @test all(0.0 .<= gaussian_shared_nuts_workspace.continuation_accept_prob .<= 1.0)
    @test all(count >= 1 for count in gaussian_shared_nuts_workspace.continuation_accept_stat_count)
    @test all(isfinite, gaussian_shared_nuts_workspace.subtree_proposed_energy)
    @test all(isfinite, gaussian_shared_nuts_workspace.subtree_delta_energy)
    @test all(0.0 .<= gaussian_shared_nuts_workspace.subtree_accept_prob .<= 1.0)
    @test all(isfinite, gaussian_shared_nuts_workspace.subtree_candidate_log_weight)
    @test all(isfinite, gaussian_shared_nuts_workspace.subtree_combined_log_weight)
    @test gaussian_shared_nuts_workspace.continuation_proposed_energy ≈
        [state.proposal_energy for state in gaussian_shared_nuts_workspace.column_continuation_states] atol=1e-8
    @test gaussian_shared_nuts_workspace.continuation_delta_energy ≈
        [state.proposal_energy_error for state in gaussian_shared_nuts_workspace.column_continuation_states] atol=1e-8
    @test any(gaussian_shared_nuts_workspace.subtree_copy_left .| gaussian_shared_nuts_workspace.subtree_copy_right)
    @test gaussian_shared_nuts_workspace.subtree_merged_turning ==
        UncertainTea._batched_is_turning!(
            falses(length(gaussian_shared_nuts_workspace.subtree_merged_turning)),
            gaussian_shared_nuts_workspace.left_position,
            gaussian_shared_nuts_workspace.right_position,
            gaussian_shared_nuts_workspace.left_momentum,
            gaussian_shared_nuts_workspace.right_momentum,
            trues(length(gaussian_shared_nuts_workspace.subtree_merged_turning)),
        )
    gaussian_single_shared_params = gaussian_batch_params[:, 1:1]
    gaussian_single_shared_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_single_shared_params,
        (),
        choicemap((:y, 0.4)),
    )
    gaussian_single_shared_logjoint = batched_logjoint_unconstrained(
        gaussian_mean,
        gaussian_single_shared_params,
        (),
        choicemap((:y, 0.4)),
    )
    gaussian_single_shared_gradient = batched_logjoint_gradient_unconstrained(
        gaussian_mean,
        gaussian_single_shared_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_single_shared_nuts_workspace,
        gaussian_mean,
        gaussian_single_shared_params,
        gaussian_single_shared_logjoint,
        gaussian_single_shared_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(96),
    )
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_single_shared_nuts_workspace,
        gaussian_mean,
        gaussian_single_shared_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        3,
        1000.0,
        MersenneTwister(97),
    )
    @test gaussian_single_shared_nuts_workspace.control.tree_depths == [2]
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_single_shared_nuts_workspace,
        gaussian_mean,
        gaussian_single_shared_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        3,
        1000.0,
        MersenneTwister(98),
    )
    @test gaussian_single_shared_nuts_workspace.control.tree_depths == [3]
    @test gaussian_single_shared_nuts_workspace.control.integration_steps[1] >= 6
    gaussian_mixed_depth_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_mixed_depth_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(99),
    )
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_mixed_depth_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        4,
        1000.0,
        MersenneTwister(100),
    )
    @test all(depth == 2 for depth in gaussian_mixed_depth_nuts_workspace.control.tree_depths)
    gaussian_mixed_depth_nuts_workspace.control.continuation_turning[1] = true
    @test UncertainTea._batched_nuts_active_depth(
        gaussian_mixed_depth_nuts_workspace,
        4,
    ) == (2, 2)
    @test gaussian_mixed_depth_nuts_workspace.control.scheduler.continuation_active ==
        BitVector([false, true, true])
    @test gaussian_mixed_depth_nuts_workspace.control.scheduler.active_depth == 2
    @test gaussian_mixed_depth_nuts_workspace.control.scheduler.active_depth_count == 2
    @test UncertainTea._activate_batched_nuts_subtree_cohort!(
        gaussian_mixed_depth_nuts_workspace,
    )
    @test gaussian_mixed_depth_nuts_workspace.subtree_active ==
        BitVector([false, true, true])
    prepared_depth = UncertainTea._prepare_batched_nuts_subtree_cohort!(
        gaussian_mixed_depth_nuts_workspace,
        4,
        MersenneTwister(1001),
    )
    @test prepared_depth == 2
    @test gaussian_mixed_depth_nuts_workspace.control.scheduler.continuation_active ==
        BitVector([false, true, true])
    @test gaussian_mixed_depth_nuts_workspace.control.scheduler.active_depth == 2
    @test gaussian_mixed_depth_nuts_workspace.control.scheduler.active_depth_count == 2
    @test gaussian_mixed_depth_nuts_workspace.subtree_active ==
        BitVector([false, true, true])
    @test all(
        gaussian_mixed_depth_nuts_workspace.control.step_direction[index] in (-1, 1)
        for index in 2:3
    )
    @test all(
        isfinite(gaussian_mixed_depth_nuts_workspace.tree_current_logjoint[index])
        for index in 2:3
    )
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_mixed_depth_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        4,
        1000.0,
        MersenneTwister(101),
    )
    @test gaussian_mixed_depth_nuts_workspace.control.tree_depths[1] == 2
    @test gaussian_mixed_depth_nuts_workspace.control.tree_depths[2:3] == [3, 3]
    @test gaussian_mixed_depth_nuts_workspace.subtree_accept_stat_count[1] == 0
    @test gaussian_mixed_depth_nuts_workspace.subtree_candidate_log_weight[1] == -Inf
    @test !gaussian_mixed_depth_nuts_workspace.subtree_copy_left[1]
    @test !gaussian_mixed_depth_nuts_workspace.subtree_copy_right[1]
    @test !gaussian_mixed_depth_nuts_workspace.subtree_select_proposal[1]
    gaussian_mixed_depth_nuts_workspace.control.continuation_turning[1] = false
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_mixed_depth_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        4,
        1000.0,
        MersenneTwister(102),
    )
    @test gaussian_mixed_depth_nuts_workspace.control.tree_depths[1] == 2
    @test gaussian_mixed_depth_nuts_workspace.control.tree_depths[2:3] == [4, 4]
    gaussian_cohort_scheduler_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_cohort_scheduler_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(103),
    )
    cohort_rng = MersenneTwister(104)
    @test gaussian_cohort_scheduler_workspace.control isa UncertainTea.BatchedNUTSControlState
    @test gaussian_cohort_scheduler_workspace.control.scheduler isa
        UncertainTea.BatchedNUTSSchedulerState
    idle_ir = UncertainTea._batched_nuts_control_ir(gaussian_cohort_scheduler_workspace)
    @test idle_ir isa UncertainTea.BatchedNUTSIdleIR
    idle_block = UncertainTea._batched_nuts_control_block(gaussian_cohort_scheduler_workspace)
    @test idle_block isa UncertainTea.BatchedNUTSIdleControlBlock
    idle_descriptor = UncertainTea._batched_nuts_step_descriptor(gaussian_cohort_scheduler_workspace)
    @test idle_descriptor isa UncertainTea.BatchedNUTSIdleStepDescriptor
    idle_state = UncertainTea._batched_nuts_step_state(gaussian_cohort_scheduler_workspace)
    @test idle_state isa UncertainTea.BatchedNUTSIdleStepState
    idle_frame = UncertainTea._batched_nuts_kernel_frame(gaussian_cohort_scheduler_workspace)
    @test idle_frame isa UncertainTea.BatchedNUTSIdleKernelFrame
    idle_program = UncertainTea._batched_nuts_kernel_program(gaussian_cohort_scheduler_workspace)
    @test idle_program isa UncertainTea.BatchedNUTSIdleKernelProgram
    @test UncertainTea._batched_nuts_kernel_ops(idle_program) ==
        (UncertainTea.NUTSKernelReloadControl,)
    @test typeof.(UncertainTea._batched_nuts_kernel_steps(idle_program)) ==
        (UncertainTea.BatchedNUTSReloadControlStep,)
    @test !UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_cohort_scheduler_workspace,
        idle_program,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        cohort_rng,
    )
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
        UncertainTea.NUTSSchedulerIdle
    @test UncertainTea._begin_batched_nuts_subtree_scheduler!(
        gaussian_cohort_scheduler_workspace,
        4,
        cohort_rng,
    )
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
        UncertainTea.NUTSSchedulerExpand
    @test gaussian_cohort_scheduler_workspace.control.scheduler.active_depth == 1
    @test gaussian_cohort_scheduler_workspace.control.scheduler.active_depth_count == 3
    @test gaussian_cohort_scheduler_workspace.control.scheduler.remaining_steps == 2
    @test gaussian_cohort_scheduler_workspace.control.scheduler.continuation_active ==
        BitVector([true, true, true])
    expand_ir = UncertainTea._batched_nuts_control_ir(gaussian_cohort_scheduler_workspace)
    @test expand_ir isa UncertainTea.BatchedNUTSExpandIR
    @test expand_ir.active_depth == 1
    @test expand_ir.active_depth_count == 3
    @test expand_ir.remaining_steps == 2
    @test expand_ir.active_chains == BitVector([true, true, true])
    expand_block = UncertainTea._batched_nuts_control_block(gaussian_cohort_scheduler_workspace)
    @test expand_block isa UncertainTea.BatchedNUTSExpandControlBlock
    @test expand_block.active_chains == expand_ir.active_chains
    @test expand_block.step_direction == expand_ir.step_direction
    expand_descriptor = UncertainTea._batched_nuts_step_descriptor(gaussian_cohort_scheduler_workspace)
    @test expand_descriptor isa UncertainTea.BatchedNUTSExpandStepDescriptor
    @test expand_descriptor.copy_left === gaussian_cohort_scheduler_workspace.subtree_copy_left
    @test expand_descriptor.copy_right === gaussian_cohort_scheduler_workspace.subtree_copy_right
    @test expand_descriptor.select_proposal ===
        gaussian_cohort_scheduler_workspace.subtree_select_proposal
    @test expand_descriptor.turning === gaussian_cohort_scheduler_workspace.subtree_turning
    expand_state = UncertainTea._batched_nuts_step_state(gaussian_cohort_scheduler_workspace)
    @test expand_state isa UncertainTea.BatchedNUTSExpandStepState
    @test expand_state.descriptor.copy_left === gaussian_cohort_scheduler_workspace.subtree_copy_left
    @test expand_state.log_weight === gaussian_cohort_scheduler_workspace.subtree_log_weight
    @test expand_state.proposed_energy ===
        gaussian_cohort_scheduler_workspace.subtree_proposed_energy
    @test expand_state.delta_energy === gaussian_cohort_scheduler_workspace.subtree_delta_energy
    @test expand_state.proposal_energy ===
        gaussian_cohort_scheduler_workspace.subtree_proposal_energy
    @test expand_state.proposal_energy_error ===
        gaussian_cohort_scheduler_workspace.subtree_proposal_energy_error
    @test expand_state.accept_prob === gaussian_cohort_scheduler_workspace.subtree_accept_prob
    @test expand_state.candidate_log_weight ===
        gaussian_cohort_scheduler_workspace.subtree_candidate_log_weight
    @test expand_state.combined_log_weight ===
        gaussian_cohort_scheduler_workspace.subtree_combined_log_weight
    expand_frame = UncertainTea._batched_nuts_kernel_frame(gaussian_cohort_scheduler_workspace)
    @test expand_frame isa UncertainTea.BatchedNUTSExpandKernelFrame
    @test expand_frame.state.log_weight === gaussian_cohort_scheduler_workspace.subtree_log_weight
    @test expand_frame.current_position === gaussian_cohort_scheduler_workspace.tree_current_position
    @test expand_frame.next_position === gaussian_cohort_scheduler_workspace.tree_next_position
    @test expand_frame.proposed_logjoint === gaussian_cohort_scheduler_workspace.proposed_logjoint
    @test expand_frame.left_position === gaussian_cohort_scheduler_workspace.tree_left_position
    @test expand_frame.right_position === gaussian_cohort_scheduler_workspace.tree_right_position
    @test expand_frame.proposal_position === gaussian_cohort_scheduler_workspace.tree_proposal_position
    @test expand_frame.current_energy === gaussian_cohort_scheduler_workspace.current_energy
    @test UncertainTea._batched_nuts_kernel_frame(
        gaussian_cohort_scheduler_workspace,
        expand_state,
    ).state === expand_state
    expand_program = UncertainTea._batched_nuts_kernel_program(gaussian_cohort_scheduler_workspace)
    @test expand_program isa UncertainTea.BatchedNUTSExpandKernelProgram
    @test UncertainTea._batched_nuts_kernel_ops(expand_program) ==
        (
            UncertainTea.NUTSKernelReloadControl,
            UncertainTea.NUTSKernelLeapfrog,
            UncertainTea.NUTSKernelHamiltonian,
            UncertainTea.NUTSKernelAdvance,
            UncertainTea.NUTSKernelTransitionPhase,
        )
    @test typeof.(UncertainTea._batched_nuts_kernel_steps(expand_program)) ==
        (
            UncertainTea.BatchedNUTSReloadControlStep,
            UncertainTea.BatchedNUTSLeapfrogStep,
            UncertainTea.BatchedNUTSHamiltonianStep,
            UncertainTea.BatchedNUTSAdvanceStep,
            UncertainTea.BatchedNUTSTransitionPhaseStep,
        )
    while gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
        UncertainTea.NUTSSchedulerExpand
        @test UncertainTea._step_batched_nuts_subtree_scheduler!(
            gaussian_cohort_scheduler_workspace,
            gaussian_mean,
            [1.0],
            (),
            choicemap((:y, 0.4)),
            0.01,
            1000.0,
            cohort_rng,
        )
    end
    merge_ir = UncertainTea._batched_nuts_control_ir(gaussian_cohort_scheduler_workspace)
    @test merge_ir isa UncertainTea.BatchedNUTSMergeIR
    @test merge_ir.active_depth == 1
    @test merge_ir.active_depth_count == 3
    @test merge_ir.started_chains == BitVector([true, true, true])
    @test merge_ir.merge_active == BitVector([true, true, true])
    merge_block = UncertainTea._batched_nuts_control_block(gaussian_cohort_scheduler_workspace)
    @test merge_block isa UncertainTea.BatchedNUTSMergeControlBlock
    @test merge_block.started_chains == merge_ir.started_chains
    @test merge_block.merge_active == merge_ir.merge_active
    merge_descriptor = UncertainTea._batched_nuts_step_descriptor(gaussian_cohort_scheduler_workspace)
    @test merge_descriptor isa UncertainTea.BatchedNUTSMergeStepDescriptor
    @test merge_descriptor.select_proposal ===
        gaussian_cohort_scheduler_workspace.continuation_select_proposal
    @test merge_descriptor.merged_turning ===
        gaussian_cohort_scheduler_workspace.subtree_merged_turning
