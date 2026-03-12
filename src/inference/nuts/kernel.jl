function _advance_batched_nuts_subtree_cohort!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
    inverse_mass_matrix::Vector{Float64},
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    any_active = false
    fill!(access.copy_left, false)
    fill!(access.copy_right, false)
    fill!(access.select_proposal, false)
    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] || continue
        tree_workspace = workspace.column_tree_workspaces[chain_index]

        if !workspace.control.step_valid[chain_index]
            workspace.subtree_divergent[chain_index] = true
            workspace.subtree_active[chain_index] = false
            continue
        end

        access.current_logjoint[chain_index] = access.proposed_logjoint[chain_index]
        tree_workspace.next.logjoint = access.proposed_logjoint[chain_index]
        _copyto_nuts_state!(tree_workspace.current, tree_workspace.next)
        workspace.subtree_integration_steps[chain_index] += 1

        if workspace.control.step_direction[chain_index] < 0
            access.copy_left[chain_index] = true
            access.left_logjoint[chain_index] = access.current_logjoint[chain_index]
        else
            access.copy_right[chain_index] = true
            access.right_logjoint[chain_index] = access.current_logjoint[chain_index]
        end

        delta_energy = access.proposed_energy[chain_index] - access.current_energy[chain_index]
        access.delta_energy[chain_index] = delta_energy
        if !isfinite(delta_energy) || delta_energy > max_delta_energy
            workspace.subtree_divergent[chain_index] = true
            workspace.subtree_active[chain_index] = false
            continue
        end

        access.accept_prob[chain_index] = min(1.0, exp(min(0.0, -delta_energy)))
        workspace.subtree_accept_stat_sum[chain_index] += access.accept_prob[chain_index]
        workspace.subtree_accept_stat_count[chain_index] += 1
        access.candidate_log_weight[chain_index] = -access.proposed_energy[chain_index]
        access.combined_log_weight[chain_index] = _logaddexp(
            access.log_weight[chain_index],
            access.candidate_log_weight[chain_index],
        )
        if !isfinite(access.log_weight[chain_index]) || log(rand(rng)) <
            access.candidate_log_weight[chain_index] -
            access.combined_log_weight[chain_index]
            access.select_proposal[chain_index] = true
            access.proposal_logjoint[chain_index] = access.current_logjoint[chain_index]
            access.proposal_energy[chain_index] = access.proposed_energy[chain_index]
            access.proposal_energy_error[chain_index] = delta_energy
        end
        access.log_weight[chain_index] = access.combined_log_weight[chain_index]
    end

    _copy_masked_nuts_buffers!(
        access.left_position,
        access.left_momentum,
        access.left_gradient,
        access.current_position,
        access.current_momentum,
        access.current_gradient,
        access.copy_left,
    )
    _copy_masked_nuts_buffers!(
        access.right_position,
        access.right_momentum,
        access.right_gradient,
        access.current_position,
        access.current_momentum,
        access.current_gradient,
        access.copy_right,
    )
    _copy_masked_nuts_buffers!(
        access.proposal_position,
        access.proposal_momentum,
        access.proposal_gradient,
        access.current_position,
        access.current_momentum,
        access.current_gradient,
        access.select_proposal,
    )
    _copy_masked_values!(
        access.proposal_logjoint,
        access.current_logjoint,
        access.select_proposal,
    )
    _sync_batched_tree_logjoint!(
        workspace,
        workspace.subtree_active .|
        access.copy_left .|
        access.copy_right .|
        access.select_proposal,
    )

    _batched_is_turning!(
        access.turning,
        access.left_position,
        access.right_position,
        access.left_momentum,
        access.right_momentum,
        workspace.subtree_active,
    )
    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] =
            workspace.subtree_active[chain_index] && !access.turning[chain_index]
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
    access::BatchedNUTSMergeKernelAccess,
    inverse_mass_matrix::Vector{Float64},
    rng::AbstractRNG,
)
    _merge_batched_nuts_continuation_frontiers!(workspace, workspace.subtree_active)
    _batched_is_turning!(
        access.merged_turning,
        access.left_position,
        access.right_position,
        access.left_momentum,
        access.right_momentum,
        workspace.subtree_active,
    )

    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] || continue
        access.select_proposal[chain_index] = false
        access.candidate_log_weight[chain_index] = -Inf
        access.combined_log_weight[chain_index] =
            access.continuation_log_weight[chain_index]
        if isfinite(workspace.subtree_log_weight[chain_index])
            access.candidate_log_weight[chain_index] =
                workspace.subtree_log_weight[chain_index]
            access.combined_log_weight[chain_index] = _logaddexp(
                access.continuation_log_weight[chain_index],
                access.candidate_log_weight[chain_index],
            )
            access.select_proposal[chain_index] =
                log(rand(rng)) < access.candidate_log_weight[chain_index] -
                access.combined_log_weight[chain_index]
            if access.select_proposal[chain_index]
                access.proposal_energy[chain_index] = _hamiltonian(
                    access.tree_proposal_logjoint[chain_index],
                    view(access.tree_proposal_momentum, :, chain_index),
                    inverse_mass_matrix,
                )
                access.proposal_energy_error[chain_index] =
                    access.proposal_energy[chain_index] -
                    access.current_energy[chain_index]
            end
        end
        _merge_batched_subtree_summary!(workspace, chain_index)
    end

    _copy_masked_nuts_buffers!(
        access.proposal_position,
        access.proposal_momentum,
        access.proposal_gradient,
        access.tree_proposal_position,
        access.tree_proposal_momentum,
        access.tree_proposal_gradient,
        access.select_proposal,
    )
    _copy_masked_values!(
        access.proposed_logjoint,
        access.continuation_proposal_logjoint,
        access.select_proposal,
    )
    _sync_batched_continuation_logjoint!(
        workspace,
        workspace.subtree_active .| access.select_proposal,
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
    access::AbstractBatchedNUTSKernelAccess,
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
        _batched_nuts_kernel_program(workspace, access),
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
        _batched_nuts_kernel_access(workspace, frame),
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
    launch_plan = _batched_nuts_launch_plan(program)
    for executor_stage in _batched_nuts_launch_executors(launch_plan)
        _execute_batched_nuts_launch_stage!(
            workspace,
            launch_plan,
            executor_stage,
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

function _execute_batched_nuts_launch_stage!(
    workspace::BatchedNUTSWorkspace,
    launch_plan::BatchedNUTSKernelLaunchPlan,
    executor_stage::BatchedNUTSKernelLaunchStageExecutor,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    _execute_batched_nuts_kernel_dataflow!(
        workspace,
        _batched_nuts_launch_stage_dataflow(executor_stage),
        execution,
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
    for barrier in _batched_nuts_launch_barriers_after(executor_stage)
        _execute_batched_nuts_kernel_barrier!(workspace, barrier, execution)
    end
    return nothing
end

function _execute_batched_nuts_kernel_dataflow!(
    workspace::BatchedNUTSWorkspace,
    dataflow::AbstractBatchedNUTSKernelDataflow,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    _execute_batched_nuts_kernel_step!(
        workspace,
        _batched_nuts_kernel_access(dataflow),
        _batched_nuts_kernel_step(dataflow),
        execution,
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
    return nothing
end

function _execute_batched_nuts_kernel_barrier!(
    workspace::BatchedNUTSWorkspace,
    barrier::BatchedNUTSKernelBarrierPlacement,
    execution::BatchedNUTSKernelExecutionState,
)
    return nothing
end

function _execute_batched_nuts_kernel_barrier!(
    workspace::BatchedNUTSWorkspace,
    barrier::BatchedNUTSKernelDeviceBarrierHint,
    execution::BatchedNUTSKernelExecutionState,
)
    return nothing
end

function _execute_batched_nuts_kernel_barrier!(
    workspace::BatchedNUTSWorkspace,
    barrier::BatchedNUTSKernelTargetBarrierHint,
    execution::BatchedNUTSKernelExecutionState,
)
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::AbstractBatchedNUTSKernelAccess,
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
    _batched_nuts_kernel_reload_control!(workspace, access)
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
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
        access,
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
    access::BatchedNUTSExpandKernelAccess,
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
    _batched_nuts_kernel_hamiltonian!(access, inverse_mass_matrix)
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
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
        access,
        inverse_mass_matrix,
        max_delta_energy,
        rng,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSMergeKernelAccess,
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
        access.block,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSMergeKernelAccess,
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
        access,
        inverse_mass_matrix,
        rng,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::AbstractBatchedNUTSKernelAccess,
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
    _batched_nuts_kernel_transition_phase!(workspace, access, execution)
    return nothing
end

function _batched_nuts_kernel_reload_control!(
    workspace::BatchedNUTSWorkspace,
    access::AbstractBatchedNUTSKernelAccess,
)
    _load_batched_nuts_control_block!(workspace, _batched_nuts_access_control_block(access))
    return workspace
end

function _batched_nuts_kernel_leapfrog!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
    model::TeaModel,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
)
    _batched_nuts_leapfrog_step_to!(
        workspace,
        model,
        access.next_position,
        access.next_momentum,
        access.next_gradient,
        access.proposed_logjoint,
        access.current_position,
        access.current_momentum,
        access.current_gradient,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        access.block.step_direction,
        access.block.active_chains,
    )
    return access
end

function _batched_nuts_kernel_hamiltonian!(
    access::BatchedNUTSExpandKernelAccess,
    inverse_mass_matrix::Vector{Float64},
)
    _batched_hamiltonian!(
        access.proposed_energy,
        access.proposed_logjoint,
        access.next_momentum,
        inverse_mass_matrix,
    )
    return access
end

function _batched_nuts_kernel_transition_phase!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
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
    access::BatchedNUTSMergeKernelAccess,
    execution::BatchedNUTSKernelExecutionState,
)
    workspace.control.scheduler.phase = NUTSSchedulerDone
    workspace.control.scheduler.remaining_steps = 0
    return workspace
end

function _batched_nuts_kernel_transition_phase!(
    workspace::BatchedNUTSWorkspace,
    access::Union{BatchedNUTSIdleKernelAccess,BatchedNUTSDoneKernelAccess},
    execution::BatchedNUTSKernelExecutionState,
)
    return workspace
end
