function _advance_batched_nuts_subtree_cohort!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSExpandKernelFrame,
    inverse_mass_matrix::Vector{Float64},
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    state = frame.state
    descriptor = state.descriptor
    any_active = false
    fill!(descriptor.copy_left, false)
    fill!(descriptor.copy_right, false)
    fill!(descriptor.select_proposal, false)
    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] || continue
        tree_workspace = workspace.column_tree_workspaces[chain_index]

        if !workspace.control.step_valid[chain_index]
            workspace.subtree_divergent[chain_index] = true
            workspace.subtree_active[chain_index] = false
            continue
        end

        frame.current_logjoint[chain_index] = frame.proposed_logjoint[chain_index]
        tree_workspace.next.logjoint = frame.proposed_logjoint[chain_index]
        _copyto_nuts_state!(tree_workspace.current, tree_workspace.next)
        workspace.subtree_integration_steps[chain_index] += 1

        if workspace.control.step_direction[chain_index] < 0
            descriptor.copy_left[chain_index] = true
            frame.left_logjoint[chain_index] = frame.current_logjoint[chain_index]
        else
            descriptor.copy_right[chain_index] = true
            frame.right_logjoint[chain_index] = frame.current_logjoint[chain_index]
        end

        delta_energy = state.proposed_energy[chain_index] - frame.current_energy[chain_index]
        state.delta_energy[chain_index] = delta_energy
        if !isfinite(delta_energy) || delta_energy > max_delta_energy
            workspace.subtree_divergent[chain_index] = true
            workspace.subtree_active[chain_index] = false
            continue
        end

        state.accept_prob[chain_index] = min(1.0, exp(min(0.0, -delta_energy)))
        workspace.subtree_accept_stat_sum[chain_index] += state.accept_prob[chain_index]
        workspace.subtree_accept_stat_count[chain_index] += 1
        state.candidate_log_weight[chain_index] = -state.proposed_energy[chain_index]
        state.combined_log_weight[chain_index] = _logaddexp(
            state.log_weight[chain_index],
            state.candidate_log_weight[chain_index],
        )
        if !isfinite(state.log_weight[chain_index]) || log(rand(rng)) <
            state.candidate_log_weight[chain_index] -
            state.combined_log_weight[chain_index]
            descriptor.select_proposal[chain_index] = true
            frame.proposal_logjoint[chain_index] = frame.current_logjoint[chain_index]
            state.proposal_energy[chain_index] = state.proposed_energy[chain_index]
            state.proposal_energy_error[chain_index] = delta_energy
        end
        state.log_weight[chain_index] = state.combined_log_weight[chain_index]
    end

    _copy_masked_nuts_buffers!(
        frame.left_position,
        frame.left_momentum,
        frame.left_gradient,
        frame.current_position,
        frame.current_momentum,
        frame.current_gradient,
        descriptor.copy_left,
    )
    _copy_masked_nuts_buffers!(
        frame.right_position,
        frame.right_momentum,
        frame.right_gradient,
        frame.current_position,
        frame.current_momentum,
        frame.current_gradient,
        descriptor.copy_right,
    )
    _copy_masked_nuts_buffers!(
        frame.proposal_position,
        frame.proposal_momentum,
        frame.proposal_gradient,
        frame.current_position,
        frame.current_momentum,
        frame.current_gradient,
        descriptor.select_proposal,
    )
    _copy_masked_values!(frame.proposal_logjoint, frame.current_logjoint, descriptor.select_proposal)
    _sync_batched_tree_logjoint!(
        workspace,
        workspace.subtree_active .|
        descriptor.copy_left .|
        descriptor.copy_right .|
        descriptor.select_proposal,
    )

    _batched_is_turning!(
        descriptor.turning,
        frame.left_position,
        frame.right_position,
        frame.left_momentum,
        frame.right_momentum,
        workspace.subtree_active,
    )
    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] =
            workspace.subtree_active[chain_index] && !descriptor.turning[chain_index]
        any_active |= workspace.subtree_active[chain_index]
    end
    return any_active
end

function _activate_batched_nuts_subtree_merge_cohort!(
    workspace::BatchedNUTSWorkspace,
)
    ir = _batched_nuts_control_ir(workspace)
    ir isa BatchedNUTSMergeIR || return false
    return _activate_batched_nuts_subtree_merge_cohort!(workspace, ir)
end

function _activate_batched_nuts_subtree_merge_cohort!(
    workspace::BatchedNUTSWorkspace,
    ir::BatchedNUTSMergeIR,
)
    copyto!(workspace.control.scheduler.subtree_started, ir.started_chains)
    fill!(workspace.subtree_active, false)
    any_started = false
    for chain_index in eachindex(workspace.control.tree_depths)
        ir.started_chains[chain_index] || continue

        workspace.control.tree_depths[chain_index] += 1
        if !ir.merge_active[chain_index]
            workspace.control.divergent_step[chain_index] = workspace.subtree_divergent[chain_index]
            continue
        end

        workspace.subtree_active[chain_index] = true
        any_started = true
    end
    return any_started
end

function _activate_batched_nuts_subtree_merge_cohort!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSMergeControlBlock,
)
    copyto!(workspace.control.scheduler.subtree_started, block.started_chains)
    fill!(workspace.subtree_active, false)
    any_started = false
    for chain_index in eachindex(workspace.control.tree_depths)
        block.started_chains[chain_index] || continue

        workspace.control.tree_depths[chain_index] += 1
        if !block.merge_active[chain_index]
            workspace.control.divergent_step[chain_index] = workspace.subtree_divergent[chain_index]
            continue
        end

        workspace.subtree_active[chain_index] = true
        any_started = true
    end
    return any_started
end

function _merge_batched_nuts_subtree_cohort!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSMergeKernelFrame,
    inverse_mass_matrix::Vector{Float64},
    rng::AbstractRNG,
)
    state = frame.state
    descriptor = state.descriptor
    _merge_batched_nuts_continuation_frontiers!(workspace, workspace.subtree_active)
    _batched_is_turning!(
        descriptor.merged_turning,
        frame.left_position,
        frame.right_position,
        frame.left_momentum,
        frame.right_momentum,
        workspace.subtree_active,
    )

    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] || continue
        descriptor.select_proposal[chain_index] = false
        state.candidate_log_weight[chain_index] = -Inf
        state.combined_log_weight[chain_index] =
            frame.continuation_log_weight[chain_index]
        if isfinite(workspace.subtree_log_weight[chain_index])
            state.candidate_log_weight[chain_index] =
                workspace.subtree_log_weight[chain_index]
            state.combined_log_weight[chain_index] = _logaddexp(
                frame.continuation_log_weight[chain_index],
                state.candidate_log_weight[chain_index],
            )
            descriptor.select_proposal[chain_index] =
                log(rand(rng)) < state.candidate_log_weight[chain_index] -
                state.combined_log_weight[chain_index]
            if descriptor.select_proposal[chain_index]
                state.proposal_energy[chain_index] = _hamiltonian(
                    frame.tree_proposal_logjoint[chain_index],
                    view(frame.tree_proposal_momentum, :, chain_index),
                    inverse_mass_matrix,
                )
                state.proposal_energy_error[chain_index] =
                    state.proposal_energy[chain_index] -
                    frame.current_energy[chain_index]
            end
        end
        _merge_batched_subtree_summary!(workspace, chain_index)
    end

    _copy_masked_nuts_buffers!(
        frame.proposal_position,
        frame.proposal_momentum,
        frame.proposal_gradient,
        frame.tree_proposal_position,
        frame.tree_proposal_momentum,
        frame.tree_proposal_gradient,
        descriptor.select_proposal,
    )
    _copy_masked_values!(
        frame.proposed_logjoint,
        frame.continuation_proposal_logjoint,
        descriptor.select_proposal,
    )
    _sync_batched_continuation_logjoint!(
        workspace,
        workspace.subtree_active .| descriptor.select_proposal,
    )
    return workspace
end

function _step_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    program = _batched_nuts_kernel_program(workspace)
    return _step_batched_nuts_subtree_scheduler!(
        workspace,
        program,
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
end

function _step_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    ir::AbstractBatchedNUTSControlIR,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    return _step_batched_nuts_subtree_scheduler!(
        workspace,
        _batched_nuts_step_descriptor(
            workspace,
            _batched_nuts_control_block(ir),
        ),
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
end

function _step_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    block::AbstractBatchedNUTSControlBlock,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    return _step_batched_nuts_subtree_scheduler!(
        workspace,
        _batched_nuts_step_descriptor(workspace, block),
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
end

function _step_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    descriptor::AbstractBatchedNUTSStepDescriptor,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    return _step_batched_nuts_subtree_scheduler!(
        workspace,
        _batched_nuts_step_state(workspace, descriptor),
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
end

function _step_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    state::AbstractBatchedNUTSStepState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    return _step_batched_nuts_subtree_scheduler!(
        workspace,
        _batched_nuts_kernel_frame(workspace, state),
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
end

function _step_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    frame::AbstractBatchedNUTSKernelFrame,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    return _step_batched_nuts_subtree_scheduler!(
        workspace,
        _batched_nuts_kernel_program(workspace, frame),
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
end

function _step_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    program::AbstractBatchedNUTSKernelProgram,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    _execute_batched_nuts_kernel_program!(
        workspace,
        program,
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
    return _batched_nuts_kernel_returns(program)
end

function _execute_batched_nuts_kernel_program!(
    workspace::BatchedNUTSWorkspace,
    program::AbstractBatchedNUTSKernelProgram,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    execution = _batched_nuts_kernel_execution_state()
    for step in _batched_nuts_kernel_steps(program)
        _execute_batched_nuts_kernel_step!(
            workspace,
            program.frame,
            step,
            execution,
            model,
            inverse_mass_matrix,
            args,
            constraints,
            step_size,
            max_delta_energy,
            rng,
        )
    end
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    frame::AbstractBatchedNUTSKernelFrame,
    ::BatchedNUTSReloadControlStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    _batched_nuts_kernel_reload_control!(workspace, frame)
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSExpandKernelFrame,
    ::BatchedNUTSLeapfrogStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    _batched_nuts_kernel_leapfrog!(
        workspace,
        frame,
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSExpandKernelFrame,
    ::BatchedNUTSHamiltonianStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    _batched_nuts_kernel_hamiltonian!(frame, inverse_mass_matrix)
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSExpandKernelFrame,
    ::BatchedNUTSAdvanceStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    execution.any_active = _advance_batched_nuts_subtree_cohort!(
        workspace,
        frame,
        inverse_mass_matrix,
        max_delta_energy,
        rng,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSMergeKernelFrame,
    ::BatchedNUTSActivateMergeStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    execution.any_active = _activate_batched_nuts_subtree_merge_cohort!(
        workspace,
        frame.state.descriptor.block,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSMergeKernelFrame,
    ::BatchedNUTSMergeStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    execution.any_active || return nothing
    _merge_batched_nuts_subtree_cohort!(
        workspace,
        frame,
        inverse_mass_matrix,
        rng,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    frame::AbstractBatchedNUTSKernelFrame,
    ::BatchedNUTSTransitionPhaseStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    _batched_nuts_kernel_transition_phase!(workspace, frame, execution)
    return nothing
end

function _batched_nuts_kernel_reload_control!(
    workspace::BatchedNUTSWorkspace,
    frame::AbstractBatchedNUTSKernelFrame,
)
    _load_batched_nuts_control_block!(workspace, _batched_nuts_frame_control_block(frame))
    return workspace
end

function _batched_nuts_kernel_leapfrog!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSExpandKernelFrame,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
)
    descriptor = frame.state.descriptor
    _batched_nuts_leapfrog_step_to!(
        workspace,
        model,
        frame.next_position,
        frame.next_momentum,
        frame.next_gradient,
        frame.proposed_logjoint,
        frame.current_position,
        frame.current_momentum,
        frame.current_gradient,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        descriptor.block.step_direction,
        descriptor.block.active_chains,
    )
    return frame
end

function _batched_nuts_kernel_hamiltonian!(
    frame::BatchedNUTSExpandKernelFrame,
    inverse_mass_matrix::Vector{Float64},
)
    _batched_hamiltonian!(
        frame.state.proposed_energy,
        frame.proposed_logjoint,
        frame.next_momentum,
        inverse_mass_matrix,
    )
    return frame
end

function _batched_nuts_kernel_transition_phase!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSExpandKernelFrame,
    execution::BatchedNUTSKernelExecutionState,
)
    workspace.control.scheduler.remaining_steps -= 1
    if !execution.any_active || workspace.control.scheduler.remaining_steps == 0
        workspace.control.scheduler.phase = NUTSSchedulerMerge
    end
    return workspace
end

function _batched_nuts_kernel_transition_phase!(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSMergeKernelFrame,
    execution::BatchedNUTSKernelExecutionState,
)
    workspace.control.scheduler.phase = NUTSSchedulerDone
    workspace.control.scheduler.remaining_steps = 0
    return workspace
end

function _batched_nuts_kernel_transition_phase!(
    workspace::BatchedNUTSWorkspace,
    frame::Union{BatchedNUTSIdleKernelFrame,BatchedNUTSDoneKernelFrame},
    execution::BatchedNUTSKernelExecutionState,
)
    return workspace
end

