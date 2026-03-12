struct NUTSSubtreeSummary
    log_weight::Float64
    accept_stat_sum::Float64
    accept_stat_count::Int
    integration_steps::Int
    turning::Bool
    divergent::Bool
end

function _reset_nuts_subtree_summary!(
    summary::NUTSSubtreeMetadataState,
)
    summary.log_weight = -Inf
    summary.accept_stat_sum = 0.0
    summary.accept_stat_count = 0
    summary.integration_steps = 0
    summary.proposed_energy = Inf
    summary.delta_energy = Inf
    summary.proposal_energy = Inf
    summary.proposal_energy_error = Inf
    summary.accept_prob = 0.0
    summary.candidate_log_weight = -Inf
    summary.combined_log_weight = -Inf
    summary.turning = false
    summary.divergent = false
    return summary
end

function _nuts_subtree_summary(
    summary::NUTSSubtreeMetadataState,
)
    return NUTSSubtreeSummary(
        summary.log_weight,
        summary.accept_stat_sum,
        summary.accept_stat_count,
        summary.integration_steps,
        summary.turning,
        summary.divergent,
    )
end

function _copy_nuts_state(state::NUTSState)
    return NUTSState(copy(state.position), copy(state.momentum), state.logjoint, copy(state.gradient))
end

function _copyto_nuts_state!(destination::NUTSState, source::NUTSState)
    copyto!(destination.position, source.position)
    copyto!(destination.momentum, source.momentum)
    destination.logjoint = source.logjoint
    copyto!(destination.gradient, source.gradient)
    return destination
end

function _copy_masked_columns!(
    destination::AbstractMatrix,
    source::AbstractMatrix,
    mask::AbstractVector{Bool},
)
    size(destination) == size(source) ||
        throw(DimensionMismatch("expected masked column copy inputs of matching size, got $(size(destination)) and $(size(source))"))
    length(mask) == size(destination, 2) ||
        throw(DimensionMismatch("expected masked column copy mask of length $(size(destination, 2)), got $(length(mask))"))
    for chain_index in eachindex(mask)
        mask[chain_index] || continue
        copyto!(view(destination, :, chain_index), view(source, :, chain_index))
    end
    return destination
end

function _copy_masked_nuts_buffers!(
    destination_position::AbstractMatrix,
    destination_momentum::AbstractMatrix,
    destination_gradient::AbstractMatrix,
    source_position::AbstractMatrix,
    source_momentum::AbstractMatrix,
    source_gradient::AbstractMatrix,
    mask::AbstractVector{Bool},
)
    _copy_masked_columns!(destination_position, source_position, mask)
    _copy_masked_columns!(destination_momentum, source_momentum, mask)
    _copy_masked_columns!(destination_gradient, source_gradient, mask)
    return destination_position, destination_momentum, destination_gradient
end

function _copy_masked_values!(
    destination::AbstractVector,
    source::AbstractVector,
    mask::AbstractVector{Bool},
)
    length(destination) == length(source) ||
        throw(DimensionMismatch("expected masked value copy inputs of matching length, got $(length(destination)) and $(length(source))"))
    length(mask) == length(destination) ||
        throw(DimensionMismatch("expected masked value copy mask of length $(length(destination)), got $(length(mask))"))
    for index in eachindex(mask)
        mask[index] || continue
        destination[index] = source[index]
    end
    return destination
end

function _single_chain_mask!(
    mask::AbstractVector{Bool},
    chain_index::Int,
)
    1 <= chain_index <= length(mask) ||
        throw(BoundsError(mask, chain_index))
    fill!(mask, false)
    mask[chain_index] = true
    return mask
end

function _sync_batched_tree_logjoint!(
    workspace::BatchedNUTSWorkspace,
    mask::AbstractVector{Bool},
)
    for chain_index in eachindex(mask)
        mask[chain_index] || continue
        tree_workspace = workspace.column_tree_workspaces[chain_index]
        tree_workspace.current.logjoint = workspace.tree_current_logjoint[chain_index]
        tree_workspace.next.logjoint = workspace.proposed_logjoint[chain_index]
        tree_workspace.left.logjoint = workspace.tree_left_logjoint[chain_index]
        tree_workspace.right.logjoint = workspace.tree_right_logjoint[chain_index]
        tree_workspace.proposal.logjoint = workspace.tree_proposal_logjoint[chain_index]
    end
    return workspace
end

function _sync_batched_continuation_logjoint!(
    workspace::BatchedNUTSWorkspace,
    mask::AbstractVector{Bool},
)
    for chain_index in eachindex(mask)
        mask[chain_index] || continue
        continuation = workspace.column_continuation_states[chain_index]
        continuation.left.logjoint = workspace.left_logjoint[chain_index]
        continuation.right.logjoint = workspace.right_logjoint[chain_index]
        continuation.proposal.logjoint = workspace.continuation_proposal_logjoint[chain_index]
    end
    return workspace
end

function _copy_single_batched_continuation_frontier_from_tree!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
    direction::Int,
)
    if direction < 0
        mask = _single_chain_mask!(workspace.subtree_copy_left, chain_index)
        fill!(workspace.subtree_copy_right, false)
        _copy_masked_nuts_buffers!(
            workspace.left_position,
            workspace.left_momentum,
            workspace.left_gradient,
            workspace.tree_left_position,
            workspace.tree_left_momentum,
            workspace.tree_left_gradient,
            mask,
        )
        _copy_masked_values!(workspace.left_logjoint, workspace.tree_left_logjoint, mask)
        _sync_batched_continuation_logjoint!(workspace, mask)
    else
        mask = _single_chain_mask!(workspace.subtree_copy_right, chain_index)
        fill!(workspace.subtree_copy_left, false)
        _copy_masked_nuts_buffers!(
            workspace.right_position,
            workspace.right_momentum,
            workspace.right_gradient,
            workspace.tree_right_position,
            workspace.tree_right_momentum,
            workspace.tree_right_gradient,
            mask,
        )
        _copy_masked_values!(workspace.right_logjoint, workspace.tree_right_logjoint, mask)
        _sync_batched_continuation_logjoint!(workspace, mask)
    end
    return workspace
end

function _copy_single_batched_continuation_proposal_from_tree!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
)
    mask = _single_chain_mask!(workspace.subtree_select_proposal, chain_index)
    _copy_masked_nuts_buffers!(
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        mask,
    )
    _copy_masked_values!(
        workspace.continuation_proposal_logjoint,
        workspace.tree_proposal_logjoint,
        mask,
    )
    _sync_batched_continuation_logjoint!(workspace, mask)
    return workspace
end

function _update_single_batched_continuation_turning!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
)
    active = _single_chain_mask!(workspace.subtree_active, chain_index)
    _batched_is_turning!(
        workspace.subtree_merged_turning,
        workspace.left_position,
        workspace.right_position,
        workspace.left_momentum,
        workspace.right_momentum,
        active,
    )
    workspace.subtree_active[chain_index] = false
    return workspace
end

function _reset_batched_nuts_subtree_scratch!(
    workspace::BatchedNUTSWorkspace,
)
    fill!(workspace.subtree_log_weight, -Inf)
    fill!(workspace.subtree_accept_stat_sum, 0.0)
    fill!(workspace.subtree_accept_stat_count, 0)
    fill!(workspace.subtree_integration_steps, 0)
    fill!(workspace.subtree_proposed_energy, Inf)
    fill!(workspace.subtree_delta_energy, Inf)
    fill!(workspace.subtree_proposal_energy, Inf)
    fill!(workspace.subtree_proposal_energy_error, Inf)
    fill!(workspace.subtree_accept_prob, 0.0)
    fill!(workspace.subtree_candidate_log_weight, -Inf)
    fill!(workspace.subtree_combined_log_weight, -Inf)
    fill!(workspace.subtree_turning, false)
    fill!(workspace.subtree_merged_turning, false)
    fill!(workspace.subtree_divergent, false)
    fill!(workspace.subtree_active, false)
    fill!(workspace.control.scheduler.subtree_started, false)
    fill!(workspace.subtree_copy_left, false)
    fill!(workspace.subtree_copy_right, false)
    fill!(workspace.subtree_select_proposal, false)
    workspace.control.scheduler.active_depth = 0
    workspace.control.scheduler.active_depth_count = 0
    workspace.control.scheduler.phase = NUTSSchedulerIdle
    workspace.control.scheduler.remaining_steps = 0
    return workspace
end

function _update_batched_nuts_continuation_active!(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
)
    any_active = false
    for chain_index in eachindex(workspace.control.tree_depths)
        workspace.control.scheduler.continuation_active[chain_index] = _nuts_continuation_active(
            workspace.control.tree_depths[chain_index],
            max_tree_depth,
            workspace.control.divergent_step[chain_index],
            workspace.control.continuation_turning[chain_index],
        )
        any_active |= workspace.control.scheduler.continuation_active[chain_index]
    end
    return any_active
end

function _select_batched_nuts_active_depth!(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
)
    workspace.control.scheduler.active_depth = 0
    workspace.control.scheduler.active_depth_count = 0
    for depth in 1:(max_tree_depth - 1)
        depth_count = 0
        for chain_index in eachindex(workspace.control.tree_depths)
            workspace.control.tree_depths[chain_index] == depth || continue
            workspace.control.scheduler.continuation_active[chain_index] || continue
            depth_count += 1
        end
        if depth_count > workspace.control.scheduler.active_depth_count
            workspace.control.scheduler.active_depth = depth
            workspace.control.scheduler.active_depth_count = depth_count
        end
    end
    return workspace.control.scheduler.active_depth_count > 0
end

function _batched_nuts_active_depth(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
)
    _update_batched_nuts_continuation_active!(workspace, max_tree_depth)
    _select_batched_nuts_active_depth!(workspace, max_tree_depth)
    return workspace.control.scheduler.active_depth, workspace.control.scheduler.active_depth_count
end

function _activate_batched_nuts_subtree_cohort!(
    workspace::BatchedNUTSWorkspace,
)
    fill!(workspace.subtree_active, false)
    any_active = false
    for chain_index in eachindex(workspace.control.tree_depths)
        workspace.control.tree_depths[chain_index] == workspace.control.scheduler.active_depth || continue
        workspace.control.scheduler.continuation_active[chain_index] || continue
        workspace.subtree_active[chain_index] = true
        any_active = true
    end
    return any_active
end

function _prepare_batched_nuts_subtree_cohort!(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
    rng::AbstractRNG,
)
    _reset_batched_nuts_subtree_scratch!(workspace)
    _update_batched_nuts_continuation_active!(workspace, max_tree_depth) || return 0
    _select_batched_nuts_active_depth!(
        workspace,
        max_tree_depth,
    ) || return 0
    _activate_batched_nuts_subtree_cohort!(workspace) || return 0
    _sample_batched_nuts_directions!(workspace.control.step_direction, rng, workspace.subtree_active)
    _initialize_batched_nuts_subtree_states!(workspace, workspace.subtree_active)
    return workspace.control.scheduler.active_depth
end

function _begin_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
    rng::AbstractRNG,
)
    active_depth = _prepare_batched_nuts_subtree_cohort!(
        workspace,
        max_tree_depth,
        rng,
    )
    if active_depth > 0
        workspace.control.scheduler.remaining_steps = 1 << active_depth
        workspace.control.scheduler.phase = NUTSSchedulerExpand
        return true
    end
    workspace.control.scheduler.phase = NUTSSchedulerDone
    workspace.control.scheduler.remaining_steps = 0
    return false
end

function _batched_nuts_merge_masks(
    workspace::BatchedNUTSWorkspace,
)
    num_chains = length(workspace.subtree_active)
    started_chains = falses(num_chains)
    merge_active = falses(num_chains)
    for chain_index in eachindex(workspace.control.tree_depths)
        workspace.control.tree_depths[chain_index] ==
            workspace.control.scheduler.active_depth || continue
        started =
            workspace.subtree_integration_steps[chain_index] > 0 ||
            workspace.subtree_divergent[chain_index]
        started_chains[chain_index] = started
        merge_active[chain_index] =
            started && workspace.subtree_integration_steps[chain_index] > 0
    end
    return started_chains, merge_active
end

function _batched_nuts_control_ir(
    workspace::BatchedNUTSWorkspace,
)
    scheduler = workspace.control.scheduler
    if scheduler.phase == NUTSSchedulerIdle
        return BatchedNUTSIdleIR()
    elseif scheduler.phase == NUTSSchedulerExpand
        return BatchedNUTSExpandIR(
            scheduler.active_depth,
            scheduler.active_depth_count,
            scheduler.remaining_steps,
            copy(workspace.subtree_active),
            copy(workspace.control.step_direction),
        )
    elseif scheduler.phase == NUTSSchedulerMerge
        started_chains, merge_active = _batched_nuts_merge_masks(workspace)
        return BatchedNUTSMergeIR(
            scheduler.active_depth,
            scheduler.active_depth_count,
            started_chains,
            merge_active,
        )
    end
    return BatchedNUTSDoneIR()
end

function _batched_nuts_control_block(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_control_block(_batched_nuts_control_ir(workspace))
end

function _batched_nuts_control_block(
    ir::BatchedNUTSIdleIR,
)
    return BatchedNUTSIdleControlBlock(ir)
end

function _batched_nuts_control_block(
    ir::BatchedNUTSExpandIR,
)
    return BatchedNUTSExpandControlBlock(
        ir,
        copy(ir.active_chains),
        copy(ir.step_direction),
    )
end

function _batched_nuts_control_block(
    ir::BatchedNUTSMergeIR,
)
    return BatchedNUTSMergeControlBlock(
        ir,
        copy(ir.started_chains),
        copy(ir.merge_active),
    )
end

function _batched_nuts_control_block(
    ir::BatchedNUTSDoneIR,
)
    return BatchedNUTSDoneControlBlock(ir)
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_step_descriptor(
        workspace,
        _batched_nuts_control_block(workspace),
    )
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSIdleControlBlock,
)
    return BatchedNUTSIdleStepDescriptor(block)
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSExpandControlBlock,
)
    return BatchedNUTSExpandStepDescriptor(
        block,
        workspace.subtree_copy_left,
        workspace.subtree_copy_right,
        workspace.subtree_select_proposal,
        workspace.subtree_turning,
    )
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSMergeControlBlock,
)
    return BatchedNUTSMergeStepDescriptor(
        block,
        workspace.continuation_select_proposal,
        workspace.subtree_merged_turning,
    )
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSDoneControlBlock,
)
    return BatchedNUTSDoneStepDescriptor(block)
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_step_state(
        workspace,
        _batched_nuts_step_descriptor(workspace),
    )
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
    descriptor::BatchedNUTSIdleStepDescriptor,
)
    return BatchedNUTSIdleStepState(descriptor)
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
    descriptor::BatchedNUTSExpandStepDescriptor,
)
    return BatchedNUTSExpandStepState(
        descriptor,
        workspace.subtree_log_weight,
        workspace.subtree_proposed_energy,
        workspace.subtree_delta_energy,
        workspace.subtree_proposal_energy,
        workspace.subtree_proposal_energy_error,
        workspace.subtree_accept_prob,
        workspace.subtree_candidate_log_weight,
        workspace.subtree_combined_log_weight,
    )
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
    descriptor::BatchedNUTSMergeStepDescriptor,
)
    return BatchedNUTSMergeStepState(
        descriptor,
        workspace.subtree_proposal_energy,
        workspace.subtree_proposal_energy_error,
        workspace.continuation_candidate_log_weight,
        workspace.continuation_combined_log_weight,
    )
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
    descriptor::BatchedNUTSDoneStepDescriptor,
)
    return BatchedNUTSDoneStepState(descriptor)
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_kernel_frame(
        workspace,
        _batched_nuts_step_state(workspace),
    )
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
    state::BatchedNUTSIdleStepState,
)
    return BatchedNUTSIdleKernelFrame(state)
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
    state::BatchedNUTSExpandStepState,
)
    return BatchedNUTSExpandKernelFrame(
        state,
        workspace.tree_current_position,
        workspace.tree_current_momentum,
        workspace.tree_current_gradient,
        workspace.tree_current_logjoint,
        workspace.tree_next_position,
        workspace.tree_next_momentum,
        workspace.tree_next_gradient,
        workspace.proposed_logjoint,
        workspace.tree_left_position,
        workspace.tree_left_momentum,
        workspace.tree_left_gradient,
        workspace.tree_left_logjoint,
        workspace.tree_right_position,
        workspace.tree_right_momentum,
        workspace.tree_right_gradient,
        workspace.tree_right_logjoint,
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        workspace.tree_proposal_logjoint,
        workspace.current_energy,
    )
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
    state::BatchedNUTSMergeStepState,
)
    return BatchedNUTSMergeKernelFrame(
        state,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.left_logjoint,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.right_logjoint,
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        workspace.tree_proposal_logjoint,
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        workspace.proposed_logjoint,
        workspace.continuation_proposal_logjoint,
        workspace.continuation_log_weight,
        workspace.current_energy,
    )
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
    state::BatchedNUTSDoneStepState,
)
    return BatchedNUTSDoneKernelFrame(state)
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_kernel_program(
        workspace,
        _batched_nuts_kernel_frame(workspace),
    )
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSIdleKernelFrame,
)
    return BatchedNUTSIdleKernelProgram(frame, _BATCHED_NUTS_IDLE_KERNEL_OPS)
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSExpandKernelFrame,
)
    return BatchedNUTSExpandKernelProgram(frame, _BATCHED_NUTS_EXPAND_KERNEL_OPS)
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSMergeKernelFrame,
)
    return BatchedNUTSMergeKernelProgram(frame, _BATCHED_NUTS_MERGE_KERNEL_OPS)
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSDoneKernelFrame,
)
    return BatchedNUTSDoneKernelProgram(frame, _BATCHED_NUTS_DONE_KERNEL_OPS)
end

_batched_nuts_kernel_ops(program::AbstractBatchedNUTSKernelProgram) = program.ops
_batched_nuts_kernel_steps(::BatchedNUTSIdleKernelProgram) = _BATCHED_NUTS_IDLE_KERNEL_STEPS
_batched_nuts_kernel_steps(::BatchedNUTSExpandKernelProgram) = _BATCHED_NUTS_EXPAND_KERNEL_STEPS
_batched_nuts_kernel_steps(::BatchedNUTSMergeKernelProgram) = _BATCHED_NUTS_MERGE_KERNEL_STEPS
_batched_nuts_kernel_steps(::BatchedNUTSDoneKernelProgram) = _BATCHED_NUTS_DONE_KERNEL_STEPS

_batched_nuts_kernel_returns(::BatchedNUTSIdleKernelProgram) = false
_batched_nuts_kernel_returns(::BatchedNUTSExpandKernelProgram) = true
_batched_nuts_kernel_returns(::BatchedNUTSMergeKernelProgram) = true
_batched_nuts_kernel_returns(::BatchedNUTSDoneKernelProgram) = false

_batched_nuts_frame_control_block(frame::BatchedNUTSIdleKernelFrame) = frame.state.descriptor.block
_batched_nuts_frame_control_block(frame::BatchedNUTSExpandKernelFrame) = frame.state.descriptor.block
_batched_nuts_frame_control_block(frame::BatchedNUTSMergeKernelFrame) = frame.state.descriptor.block
_batched_nuts_frame_control_block(frame::BatchedNUTSDoneKernelFrame) = frame.state.descriptor.block

function _batched_nuts_kernel_execution_state()
    return BatchedNUTSKernelExecutionState(false)
end

function _load_batched_nuts_control_ir!(
    workspace::BatchedNUTSWorkspace,
    ::BatchedNUTSIdleIR,
)
    workspace.control.scheduler.phase = NUTSSchedulerIdle
    workspace.control.scheduler.remaining_steps = 0
    return workspace
end

function _load_batched_nuts_control_block!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSIdleControlBlock,
)
    return _load_batched_nuts_control_ir!(workspace, block.ir)
end

function _load_batched_nuts_control_ir!(
    workspace::BatchedNUTSWorkspace,
    ir::BatchedNUTSExpandIR,
)
    workspace.control.scheduler.active_depth = ir.active_depth
    workspace.control.scheduler.active_depth_count = ir.active_depth_count
    workspace.control.scheduler.phase = NUTSSchedulerExpand
    workspace.control.scheduler.remaining_steps = ir.remaining_steps
    copyto!(workspace.subtree_active, ir.active_chains)
    copyto!(workspace.control.step_direction, ir.step_direction)
    return workspace
end

function _load_batched_nuts_control_block!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSExpandControlBlock,
)
    _load_batched_nuts_control_ir!(workspace, block.ir)
    copyto!(workspace.subtree_active, block.active_chains)
    copyto!(workspace.control.step_direction, block.step_direction)
    return workspace
end

function _load_batched_nuts_control_ir!(
    workspace::BatchedNUTSWorkspace,
    ir::BatchedNUTSMergeIR,
)
    workspace.control.scheduler.active_depth = ir.active_depth
    workspace.control.scheduler.active_depth_count = ir.active_depth_count
    workspace.control.scheduler.phase = NUTSSchedulerMerge
    workspace.control.scheduler.remaining_steps = 0
    copyto!(workspace.control.scheduler.subtree_started, ir.started_chains)
    return workspace
end

function _load_batched_nuts_control_block!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSMergeControlBlock,
)
    _load_batched_nuts_control_ir!(workspace, block.ir)
    copyto!(workspace.control.scheduler.subtree_started, block.started_chains)
    return workspace
end

function _load_batched_nuts_control_ir!(
    workspace::BatchedNUTSWorkspace,
    ::BatchedNUTSDoneIR,
)
    workspace.control.scheduler.phase = NUTSSchedulerDone
    workspace.control.scheduler.remaining_steps = 0
    return workspace
end

function _load_batched_nuts_control_block!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSDoneControlBlock,
)
    return _load_batched_nuts_control_ir!(workspace, block.ir)
end

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

