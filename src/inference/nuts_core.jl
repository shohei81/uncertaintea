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

function _load_batched_nuts_first_states!(
    workspace::BatchedNUTSWorkspace,
    position::AbstractMatrix,
    current_logjoint::AbstractVector,
    current_gradient::AbstractMatrix,
    active::AbstractVector{Bool},
)
    length(active) == size(position, 2) ||
        throw(DimensionMismatch("expected first-state active mask of length $(size(position, 2)), got $(length(active))"))
    _copy_masked_nuts_buffers!(
        workspace.tree_current_position,
        workspace.tree_current_momentum,
        workspace.tree_current_gradient,
        position,
        workspace.current_momentum,
        current_gradient,
        active,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_next_position,
        workspace.tree_next_momentum,
        workspace.tree_next_gradient,
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        active,
    )
    _copy_masked_values!(workspace.tree_current_logjoint, current_logjoint, active)
    _sync_batched_tree_logjoint!(workspace, active)
    return workspace
end

function _load_nuts_state!(
    destination::NUTSState,
    position::AbstractVector,
    momentum::AbstractVector,
    logjoint::Real,
    gradient::AbstractVector,
)
    copyto!(destination.position, position)
    copyto!(destination.momentum, momentum)
    destination.logjoint = Float64(logjoint)
    copyto!(destination.gradient, gradient)
    return destination
end

function _initialize_nuts_continuation!(
    continuation::NUTSContinuationState,
    left::NUTSState,
    right::NUTSState,
    proposal::NUTSState,
    proposal_energy::Float64,
    proposal_energy_error::Float64,
    log_weight::Float64,
    accept_stat_sum::Float64,
    accept_stat_count::Int,
    integration_steps::Int,
    tree_depth::Int,
    turning::Bool,
    divergent::Bool,
)
    _copyto_nuts_state!(continuation.left, left)
    _copyto_nuts_state!(continuation.right, right)
    _copyto_nuts_state!(continuation.proposal, proposal)
    continuation.proposal_energy = proposal_energy
    continuation.proposal_energy_error = proposal_energy_error
    continuation.log_weight = log_weight
    continuation.accept_stat_sum = accept_stat_sum
    continuation.accept_stat_count = accept_stat_count
    continuation.integration_steps = integration_steps
    continuation.tree_depth = tree_depth
    continuation.turning = turning
    continuation.divergent = divergent
    return continuation
end

function _initialize_nuts_first_step!(
    continuation::NUTSContinuationState,
    current_state::NUTSState,
    proposed_state::NUTSState,
    valid::Bool,
    direction::Int,
    initial_hamiltonian::Float64,
    inverse_mass_matrix::Vector{Float64},
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    _initialize_nuts_continuation!(
        continuation,
        current_state,
        current_state,
        current_state,
        initial_hamiltonian,
        0.0,
        -initial_hamiltonian,
        0.0,
        0,
        0,
        1,
        false,
        false,
    )

    valid || return (false, true)

    if direction < 0
        _copyto_nuts_state!(continuation.left, proposed_state)
    else
        _copyto_nuts_state!(continuation.right, proposed_state)
    end

    proposed_hamiltonian = _hamiltonian(proposed_state.logjoint, proposed_state.momentum, inverse_mass_matrix)
    delta_energy = proposed_hamiltonian - initial_hamiltonian
    continuation.integration_steps = 1
    if !isfinite(delta_energy) || delta_energy > max_delta_energy
        continuation.divergent = true
        return (false, true)
    end

    continuation.accept_stat_sum = min(1.0, exp(min(0.0, -delta_energy)))
    continuation.accept_stat_count = 1
    candidate_log_weight = -proposed_hamiltonian
    combined_log_weight = _logaddexp(continuation.log_weight, candidate_log_weight)
    moved = log(rand(rng)) < candidate_log_weight - combined_log_weight
    if moved
        _copyto_nuts_state!(continuation.proposal, proposed_state)
        continuation.proposal_energy = proposed_hamiltonian
        continuation.proposal_energy_error = delta_energy
    end
    continuation.log_weight = combined_log_weight
    continuation.turning = _is_turning(
        continuation.left.position,
        continuation.right.position,
        continuation.left.momentum,
        continuation.right.momentum,
    )
    return (moved, false)
end

function _initialize_nuts_first_trajectory!(
    continuation::NUTSContinuationState,
    tree_workspace::NUTSSubtreeWorkspace,
    valid::Bool,
    direction::Int,
    initial_hamiltonian::Float64,
    inverse_mass_matrix::Vector{Float64},
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    return _initialize_nuts_first_step!(
        continuation,
        tree_workspace.current,
        tree_workspace.next,
        valid,
        direction,
        initial_hamiltonian,
        inverse_mass_matrix,
        max_delta_energy,
        rng,
    )
end

function _initialize_batched_nuts_first_step!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
    current_state::NUTSState,
    proposed_state::NUTSState,
    valid::Bool,
    direction::Int,
    initial_hamiltonian::Float64,
    inverse_mass_matrix::Vector{Float64},
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    continuation = workspace.column_continuation_states[chain_index]
    _copyto_nuts_state!(continuation.left, current_state)
    _copyto_nuts_state!(continuation.right, current_state)
    _copyto_nuts_state!(continuation.proposal, current_state)
    continuation.proposal_energy = initial_hamiltonian
    continuation.proposal_energy_error = 0.0
    workspace.left_logjoint[chain_index] = current_state.logjoint
    workspace.right_logjoint[chain_index] = current_state.logjoint
    workspace.continuation_proposal_logjoint[chain_index] = current_state.logjoint
    workspace.continuation_log_weight[chain_index] = -initial_hamiltonian
    workspace.continuation_accept_stat_sum[chain_index] = 0.0
    workspace.continuation_accept_stat_count[chain_index] = 0
    workspace.continuation_proposed_energy[chain_index] = initial_hamiltonian
    workspace.continuation_delta_energy[chain_index] = 0.0
    workspace.continuation_accept_prob[chain_index] = 0.0
    workspace.continuation_candidate_log_weight[chain_index] = -Inf
    workspace.continuation_combined_log_weight[chain_index] = -Inf
    workspace.continuation_select_proposal[chain_index] = false
    workspace.control.integration_steps[chain_index] = 0
    workspace.control.tree_depths[chain_index] = 1
    workspace.control.continuation_turning[chain_index] = false
    workspace.control.divergent_step[chain_index] = false

    valid || begin
        workspace.control.divergent_step[chain_index] = true
        return (false, true)
    end

    if direction < 0
        _copyto_nuts_state!(continuation.left, proposed_state)
        workspace.left_logjoint[chain_index] = proposed_state.logjoint
    else
        _copyto_nuts_state!(continuation.right, proposed_state)
        workspace.right_logjoint[chain_index] = proposed_state.logjoint
    end

    workspace.continuation_proposed_energy[chain_index] =
        _hamiltonian(proposed_state.logjoint, proposed_state.momentum, inverse_mass_matrix)
    delta_energy = workspace.continuation_proposed_energy[chain_index] - initial_hamiltonian
    workspace.continuation_delta_energy[chain_index] = delta_energy
    workspace.control.integration_steps[chain_index] = 1
    if !isfinite(delta_energy) || delta_energy > max_delta_energy
        workspace.control.divergent_step[chain_index] = true
        return (false, true)
    end

    workspace.continuation_accept_prob[chain_index] = min(1.0, exp(min(0.0, -delta_energy)))
    workspace.continuation_accept_stat_sum[chain_index] = workspace.continuation_accept_prob[chain_index]
    workspace.continuation_accept_stat_count[chain_index] = 1
    workspace.continuation_candidate_log_weight[chain_index] = -workspace.continuation_proposed_energy[chain_index]
    workspace.continuation_combined_log_weight[chain_index] = _logaddexp(
        workspace.continuation_log_weight[chain_index],
        workspace.continuation_candidate_log_weight[chain_index],
    )
    workspace.continuation_select_proposal[chain_index] =
        log(rand(rng)) < workspace.continuation_candidate_log_weight[chain_index] -
        workspace.continuation_combined_log_weight[chain_index]
    moved = workspace.continuation_select_proposal[chain_index]
    if moved
        _copyto_nuts_state!(continuation.proposal, proposed_state)
        continuation.proposal_energy = workspace.continuation_proposed_energy[chain_index]
        continuation.proposal_energy_error = delta_energy
        workspace.continuation_proposal_logjoint[chain_index] = proposed_state.logjoint
    end
    workspace.continuation_log_weight[chain_index] = workspace.continuation_combined_log_weight[chain_index]
    workspace.control.continuation_turning[chain_index] = _is_turning(
        continuation.left.position,
        continuation.right.position,
        continuation.left.momentum,
        continuation.right.momentum,
    )
    sync_mask = falses(length(workspace.left_logjoint))
    sync_mask[chain_index] = true
    _sync_batched_continuation_logjoint!(workspace, sync_mask)
    return (moved, false)
end

function NUTSSubtreeWorkspace(num_params::Int)
    state() = NUTSState(zeros(num_params), zeros(num_params), 0.0, zeros(num_params))
    summary = NUTSSubtreeMetadataState(-Inf, 0.0, 0, 0, Inf, Inf, Inf, Inf, 0.0, -Inf, -Inf, false, false)
    return NUTSSubtreeWorkspace(state(), state(), state(), state(), state(), summary)
end

function NUTSContinuationState(num_params::Int)
    state() = NUTSState(zeros(num_params), zeros(num_params), 0.0, zeros(num_params))
    return NUTSContinuationState(state(), state(), state(), Inf, Inf, -Inf, 0.0, 0, 0, 0, false, false)
end

function _logaddexp(x::Float64, y::Float64)
    if x == -Inf
        return y
    elseif y == -Inf
        return x
    end
    high = max(x, y)
    return high + log1p(exp(min(x, y) - high))
end

function _leapfrog_step!(
    destination::NUTSState,
    model::TeaModel,
    state::NUTSState,
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
)
    q = destination.position
    p = destination.momentum
    gradient = destination.gradient
    copyto!(q, state.position)
    copyto!(p, state.momentum)
    p .+= (step_size / 2) .* state.gradient
    q .+= step_size .* (inverse_mass_matrix .* p)
    proposed_logjoint = logjoint_unconstrained(model, q, args, constraints)
    isfinite(proposed_logjoint) || return false
    proposed_gradient = _logjoint_gradient!(gradient_cache, q)
    all(isfinite, proposed_gradient) || return false
    copyto!(gradient, proposed_gradient)
    p .+= (step_size / 2) .* gradient
    destination.logjoint = proposed_logjoint
    return true
end

function _is_turning(
    left_position::AbstractVector,
    right_position::AbstractVector,
    left_momentum::AbstractVector,
    right_momentum::AbstractVector,
)
    delta = right_position .- left_position
    return dot(delta, left_momentum) <= 0 || dot(delta, right_momentum) <= 0
end

function _batched_is_turning!(
    destination::AbstractVector{Bool},
    left_position::AbstractMatrix,
    right_position::AbstractMatrix,
    left_momentum::AbstractMatrix,
    right_momentum::AbstractMatrix,
    active::AbstractVector{Bool},
)
    num_chains = size(left_position, 2)
    size(left_position) == size(right_position) == size(left_momentum) == size(right_momentum) ||
        throw(DimensionMismatch("batched turning check requires matching position and momentum matrices"))
    length(destination) == num_chains ||
        throw(DimensionMismatch("expected turning destination of length $num_chains, got $(length(destination))"))
    length(active) == num_chains ||
        throw(DimensionMismatch("expected active mask of length $num_chains, got $(length(active))"))

    for chain_index in 1:num_chains
        if !active[chain_index]
            destination[chain_index] = false
            continue
        end
        left_dot = 0.0
        right_dot = 0.0
        for parameter_index in axes(left_position, 1)
            delta = right_position[parameter_index, chain_index] - left_position[parameter_index, chain_index]
            left_dot += delta * left_momentum[parameter_index, chain_index]
            right_dot += delta * right_momentum[parameter_index, chain_index]
        end
        destination[chain_index] = left_dot <= 0 || right_dot <= 0
    end
    return destination
end

function _merge_batched_nuts_continuation_frontiers!(
    workspace::BatchedNUTSWorkspace,
    active::AbstractVector{Bool},
)
    length(active) == size(workspace.left_position, 2) ||
        throw(DimensionMismatch("expected continuation-frontier active mask of length $(size(workspace.left_position, 2)), got $(length(active))"))
    fill!(workspace.subtree_copy_left, false)
    fill!(workspace.subtree_copy_right, false)
    for chain_index in eachindex(active)
        active[chain_index] || continue
        if workspace.control.step_direction[chain_index] < 0
            workspace.subtree_copy_left[chain_index] = true
            workspace.left_logjoint[chain_index] = workspace.tree_left_logjoint[chain_index]
        else
            workspace.subtree_copy_right[chain_index] = true
            workspace.right_logjoint[chain_index] = workspace.tree_right_logjoint[chain_index]
        end
    end
    _copy_masked_nuts_buffers!(
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.tree_left_position,
        workspace.tree_left_momentum,
        workspace.tree_left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.tree_right_position,
        workspace.tree_right_momentum,
        workspace.tree_right_gradient,
        workspace.subtree_copy_right,
    )
    _sync_batched_continuation_logjoint!(workspace, active)
    return workspace
end

function _initialize_batched_nuts_subtree_states!(
    workspace::BatchedNUTSWorkspace,
    active::AbstractVector{Bool},
)
    length(active) == size(workspace.left_position, 2) ||
        throw(DimensionMismatch("expected subtree-state active mask of length $(size(workspace.left_position, 2)), got $(length(active))"))
    fill!(workspace.subtree_copy_left, false)
    fill!(workspace.subtree_copy_right, false)
    fill!(workspace.subtree_select_proposal, false)
    for chain_index in eachindex(active)
        active[chain_index] || continue
        if workspace.control.step_direction[chain_index] < 0
            workspace.subtree_copy_left[chain_index] = true
            start_logjoint = workspace.left_logjoint[chain_index]
        else
            workspace.subtree_copy_right[chain_index] = true
            start_logjoint = workspace.right_logjoint[chain_index]
        end
        workspace.tree_current_logjoint[chain_index] = start_logjoint
        workspace.tree_left_logjoint[chain_index] = start_logjoint
        workspace.tree_right_logjoint[chain_index] = start_logjoint
        workspace.tree_proposal_logjoint[chain_index] = start_logjoint
    end
    _copy_masked_nuts_buffers!(
        workspace.tree_current_position,
        workspace.tree_current_momentum,
        workspace.tree_current_gradient,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_current_position,
        workspace.tree_current_momentum,
        workspace.tree_current_gradient,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.subtree_copy_right,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_left_position,
        workspace.tree_left_momentum,
        workspace.tree_left_gradient,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_left_position,
        workspace.tree_left_momentum,
        workspace.tree_left_gradient,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.subtree_copy_right,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_right_position,
        workspace.tree_right_momentum,
        workspace.tree_right_gradient,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_right_position,
        workspace.tree_right_momentum,
        workspace.tree_right_gradient,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.subtree_copy_right,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.subtree_copy_right,
    )
    _copy_masked_values!(workspace.tree_current_logjoint, workspace.left_logjoint, workspace.subtree_copy_left)
    _copy_masked_values!(workspace.tree_current_logjoint, workspace.right_logjoint, workspace.subtree_copy_right)
    _copy_masked_values!(workspace.tree_left_logjoint, workspace.left_logjoint, workspace.subtree_copy_left)
    _copy_masked_values!(workspace.tree_left_logjoint, workspace.right_logjoint, workspace.subtree_copy_right)
    _copy_masked_values!(workspace.tree_right_logjoint, workspace.left_logjoint, workspace.subtree_copy_left)
    _copy_masked_values!(workspace.tree_right_logjoint, workspace.right_logjoint, workspace.subtree_copy_right)
    _copy_masked_values!(workspace.tree_proposal_logjoint, workspace.left_logjoint, workspace.subtree_copy_left)
    _copy_masked_values!(workspace.tree_proposal_logjoint, workspace.right_logjoint, workspace.subtree_copy_right)
    _sync_batched_tree_logjoint!(workspace, active)
    return workspace
end

function _merge_batched_subtree_summary!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
)
    continuation = workspace.column_continuation_states[chain_index]
    workspace.control.integration_steps[chain_index] += workspace.subtree_integration_steps[chain_index]
    workspace.continuation_accept_stat_sum[chain_index] += workspace.subtree_accept_stat_sum[chain_index]
    workspace.continuation_accept_stat_count[chain_index] += workspace.subtree_accept_stat_count[chain_index]
    if isfinite(workspace.subtree_log_weight[chain_index])
        if workspace.continuation_select_proposal[chain_index]
            continuation.proposal_energy = workspace.subtree_proposal_energy[chain_index]
            continuation.proposal_energy_error =
                workspace.subtree_proposal_energy_error[chain_index]
            workspace.continuation_proposal_logjoint[chain_index] =
                workspace.tree_proposal_logjoint[chain_index]
            workspace.continuation_proposed_energy[chain_index] =
                continuation.proposal_energy
            workspace.continuation_delta_energy[chain_index] =
                continuation.proposal_energy_error
        end
        workspace.continuation_log_weight[chain_index] = workspace.continuation_combined_log_weight[chain_index]
    end

    workspace.control.divergent_step[chain_index] = workspace.subtree_divergent[chain_index]
    workspace.control.continuation_turning[chain_index] =
        workspace.subtree_turning[chain_index] ||
        workspace.subtree_merged_turning[chain_index]
    return workspace
end

function _copy_nuts_continuation_frontier_from_tree!(
    continuation::NUTSContinuationState,
    subtree_workspace::NUTSSubtreeWorkspace,
    direction::Int,
)
    if direction < 0
        _copyto_nuts_state!(continuation.left, subtree_workspace.left)
    else
        _copyto_nuts_state!(continuation.right, subtree_workspace.right)
    end
    return continuation
end

function _copy_nuts_continuation_proposal_from_tree!(
    continuation::NUTSContinuationState,
    subtree_workspace::NUTSSubtreeWorkspace,
)
    _copyto_nuts_state!(continuation.proposal, subtree_workspace.proposal)
    continuation.proposal_energy = subtree_workspace.summary.proposal_energy
    continuation.proposal_energy_error = subtree_workspace.summary.proposal_energy_error
    return continuation
end

function _merge_nuts_continuation_turning!(
    continuation::NUTSContinuationState,
    subtree_turning::Bool,
)
    continuation.turning =
        subtree_turning || _is_turning(
            continuation.left.position,
            continuation.right.position,
            continuation.left.momentum,
            continuation.right.momentum,
        )
    return continuation
end

function _merge_nuts_subtree_summary!(
    continuation::NUTSContinuationState,
    subtree_workspace::NUTSSubtreeWorkspace,
    combined_log_weight::Float64,
)
    summary = subtree_workspace.summary
    continuation.integration_steps += summary.integration_steps
    continuation.accept_stat_sum += summary.accept_stat_sum
    continuation.accept_stat_count += summary.accept_stat_count
    if isfinite(summary.log_weight)
        continuation.log_weight = combined_log_weight
    end

    continuation.divergent = summary.divergent
    return continuation
end

function _build_nuts_subtree(
    subtree_workspace::NUTSSubtreeWorkspace,
    model::TeaModel,
    start_state::NUTSState,
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    direction::Int,
    depth::Int,
    initial_hamiltonian::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    current = subtree_workspace.current
    next = subtree_workspace.next
    left = subtree_workspace.left
    right = subtree_workspace.right
    proposal = subtree_workspace.proposal
    summary = subtree_workspace.summary
    _copyto_nuts_state!(current, start_state)
    _copyto_nuts_state!(left, start_state)
    _copyto_nuts_state!(right, start_state)
    _copyto_nuts_state!(proposal, start_state)
    _reset_nuts_subtree_summary!(summary)

    for _ in 1:(1 << depth)
        if !_leapfrog_step!(
            next,
            model,
            current,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            direction * step_size,
        )
            summary.divergent = true
            break
        end
        current, next = next, current
        summary.integration_steps += 1
        if direction < 0
            _copyto_nuts_state!(left, current)
        else
            _copyto_nuts_state!(right, current)
        end

        summary.proposed_energy = _hamiltonian(
            current.logjoint,
            current.momentum,
            inverse_mass_matrix,
        )
        summary.delta_energy = summary.proposed_energy - initial_hamiltonian
        if !isfinite(summary.delta_energy) || summary.delta_energy > max_delta_energy
            summary.divergent = true
            break
        end

        summary.accept_prob = min(1.0, exp(min(0.0, -summary.delta_energy)))
        summary.accept_stat_sum += summary.accept_prob
        summary.accept_stat_count += 1
        summary.candidate_log_weight = -summary.proposed_energy
        summary.combined_log_weight = _logaddexp(
            summary.log_weight,
            summary.candidate_log_weight,
        )
        if !isfinite(summary.log_weight) || log(rand(rng)) <
            summary.candidate_log_weight - summary.combined_log_weight
            _copyto_nuts_state!(proposal, current)
            summary.proposal_energy = summary.proposed_energy
            summary.proposal_energy_error = summary.delta_energy
        end
        summary.log_weight = summary.combined_log_weight

        summary.turning = _is_turning(
            left.position,
            right.position,
            left.momentum,
            right.momentum,
        )
        summary.turning && break
    end

    return _nuts_subtree_summary(summary)
end

function _continue_nuts_proposal!(
    continuation::NUTSContinuationState,
    model::TeaModel,
    initial_hamiltonian::Float64,
    gradient_cache::LogjointGradientCache,
    tree_workspace::NUTSSubtreeWorkspace,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    while _nuts_continuation_active(
        continuation.tree_depth,
        max_tree_depth,
        continuation.divergent,
        continuation.turning,
    )
        direction = _sample_nuts_direction(rng)
        subtree = _build_nuts_subtree(
            tree_workspace,
            model,
            _nuts_subtree_start_state(continuation, direction),
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            step_size,
            direction,
            continuation.tree_depth,
            initial_hamiltonian,
            max_delta_energy,
            rng,
        )
        continuation.tree_depth += 1

        if subtree.integration_steps == 0
            continuation.divergent = subtree.divergent
            break
        end

        _copy_nuts_continuation_frontier_from_tree!(
            continuation,
            tree_workspace,
            direction,
        )

        combined_log_weight = continuation.log_weight
        if isfinite(tree_workspace.summary.log_weight)
            combined_log_weight = _logaddexp(
                continuation.log_weight,
                tree_workspace.summary.log_weight,
            )
            if log(rand(rng)) < tree_workspace.summary.log_weight - combined_log_weight
                _copy_nuts_continuation_proposal_from_tree!(
                    continuation,
                    tree_workspace,
                )
            end
        end

        _merge_nuts_subtree_summary!(
            continuation,
            tree_workspace,
            combined_log_weight,
        )
        _merge_nuts_continuation_turning!(continuation, tree_workspace.summary.turning)
    end
    return continuation
end

function _continue_batched_nuts_proposal!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
    model::TeaModel,
    initial_hamiltonian::Float64,
    gradient_cache::LogjointGradientCache,
    tree_workspace::NUTSSubtreeWorkspace,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    continuation = workspace.column_continuation_states[chain_index]
    while _nuts_continuation_active(
        workspace.control.tree_depths[chain_index],
        max_tree_depth,
        workspace.control.divergent_step[chain_index],
        workspace.control.continuation_turning[chain_index],
    )
        direction = _sample_nuts_direction(rng)
        subtree = _build_nuts_subtree(
            tree_workspace,
            model,
            _nuts_subtree_start_state(continuation, direction),
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            step_size,
            direction,
            workspace.control.tree_depths[chain_index],
            initial_hamiltonian,
            max_delta_energy,
            rng,
        )
        workspace.control.tree_depths[chain_index] += 1
        workspace.subtree_log_weight[chain_index] = subtree.log_weight
        workspace.subtree_accept_stat_sum[chain_index] = subtree.accept_stat_sum
        workspace.subtree_accept_stat_count[chain_index] = subtree.accept_stat_count
        workspace.subtree_integration_steps[chain_index] = subtree.integration_steps
        workspace.subtree_turning[chain_index] = subtree.turning
        workspace.subtree_divergent[chain_index] = subtree.divergent
        workspace.subtree_merged_turning[chain_index] = false
        workspace.subtree_proposal_energy[chain_index] = Inf
        workspace.subtree_proposal_energy_error[chain_index] = Inf

        if subtree.integration_steps == 0
            workspace.control.divergent_step[chain_index] = workspace.subtree_divergent[chain_index]
            break
        end

        _copy_single_batched_continuation_frontier_from_tree!(
            workspace,
            chain_index,
            direction,
        )

        workspace.continuation_select_proposal[chain_index] = false
        workspace.continuation_candidate_log_weight[chain_index] = -Inf
        workspace.continuation_combined_log_weight[chain_index] =
            workspace.continuation_log_weight[chain_index]
        if isfinite(workspace.subtree_log_weight[chain_index])
            workspace.subtree_proposal_energy[chain_index] = _hamiltonian(
                tree_workspace.proposal.logjoint,
                tree_workspace.proposal.momentum,
                inverse_mass_matrix,
            )
            workspace.subtree_proposal_energy_error[chain_index] =
                workspace.subtree_proposal_energy[chain_index] - initial_hamiltonian
            workspace.continuation_candidate_log_weight[chain_index] =
                workspace.subtree_log_weight[chain_index]
            workspace.continuation_combined_log_weight[chain_index] = _logaddexp(
                workspace.continuation_log_weight[chain_index],
                workspace.continuation_candidate_log_weight[chain_index],
            )
            workspace.continuation_select_proposal[chain_index] =
                log(rand(rng)) < workspace.continuation_candidate_log_weight[chain_index] -
                workspace.continuation_combined_log_weight[chain_index]
            workspace.tree_proposal_logjoint[chain_index] = tree_workspace.proposal.logjoint
            if workspace.continuation_select_proposal[chain_index]
                _copy_single_batched_continuation_proposal_from_tree!(
                    workspace,
                    chain_index,
                )
            end
        end
        _update_single_batched_continuation_turning!(workspace, chain_index)
        _merge_batched_subtree_summary!(workspace, chain_index)
    end
    return workspace
end

function _continue_batched_nuts_batched_subtree!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    max_tree_depth > 1 || return false
    _begin_batched_nuts_subtree_scheduler!(
        workspace,
        max_tree_depth,
        rng,
    ) || return false
    while _step_batched_nuts_subtree_scheduler!(
        workspace,
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
    end

    return true
end

function _nuts_proposal(
    model::TeaModel,
    position::AbstractVector{Float64},
    current_logjoint::Float64,
    current_gradient::AbstractVector{Float64},
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    tree_workspace = NUTSSubtreeWorkspace(length(position))
    continuation = NUTSContinuationState(length(position))
    initial_momentum = _sample_momentum(rng, inverse_mass_matrix)
    initial_state = _load_nuts_state!(
        tree_workspace.current,
        position,
        initial_momentum,
        current_logjoint,
        current_gradient,
    )
    initial_hamiltonian = _hamiltonian(initial_state.logjoint, initial_state.momentum, inverse_mass_matrix)
    direction = _sample_nuts_direction(rng)
    first_step_valid = _leapfrog_step!(
        tree_workspace.next,
        model,
        initial_state,
        gradient_cache,
        inverse_mass_matrix,
        args,
        constraints,
        direction * step_size,
    )
    _initialize_nuts_first_trajectory!(
        continuation,
        tree_workspace,
        first_step_valid,
        direction,
        initial_hamiltonian,
        inverse_mass_matrix,
        max_delta_energy,
        rng,
    )
    _continue_nuts_proposal!(
        continuation,
        model,
        initial_hamiltonian,
        gradient_cache,
        tree_workspace,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_tree_depth,
        max_delta_energy,
        rng,
    )
    accept_stat, proposed_energy, energy_error, moved = _nuts_proposal_summary(
        continuation,
        position,
    )
    return continuation.proposal, accept_stat, continuation.tree_depth, continuation.integration_steps, proposed_energy, energy_error, continuation.divergent, moved
end

function _batched_nuts_proposals!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    current_logjoint::AbstractVector{Float64},
    current_gradient::AbstractMatrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    num_chains = size(position, 2)
    _initialize_batched_nuts_continuations!(
        workspace,
        model,
        position,
        current_logjoint,
        current_gradient,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_delta_energy,
        rng,
    )
    while _continue_batched_nuts_batched_subtree!(
        workspace,
        model,
        position,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_tree_depth,
        max_delta_energy,
        rng,
    )
    end
    for chain_index in 1:num_chains
        _continue_batched_nuts_proposal!(
            workspace,
            chain_index,
            model,
            workspace.current_energy[chain_index],
            workspace.column_gradient_caches[chain_index],
            workspace.column_tree_workspaces[chain_index],
            inverse_mass_matrix,
            _batched_args(args, chain_index),
            _batched_constraints(constraints, chain_index),
            step_size,
            max_tree_depth,
            max_delta_energy,
            rng,
        )
    end
    _finalize_batched_nuts_proposals!(workspace, position)
    return workspace
end

function _batched_nuts_state(
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    logjoint::AbstractVector{Float64},
    gradient::AbstractMatrix{Float64},
    chain_index::Int,
)
    return NUTSState(
        view(position, :, chain_index),
        view(momentum, :, chain_index),
        logjoint[chain_index],
        view(gradient, :, chain_index),
    )
end

function _initialize_batched_nuts_continuations!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    current_logjoint::AbstractVector{Float64},
    current_gradient::AbstractMatrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    num_chains = size(position, 2)
    _sample_batched_momentum!(workspace.current_momentum, rng, sqrt.(Float64.(inverse_mass_matrix)))
    _batched_hamiltonian!(workspace.current_energy, current_logjoint, workspace.current_momentum, inverse_mass_matrix)
    fill!(workspace.control.accepted_step, true)
    _sample_batched_nuts_directions!(workspace.control.step_direction, rng, workspace.control.accepted_step)
    _batched_nuts_leapfrog_step!(
        workspace,
        model,
        position,
        workspace.current_momentum,
        current_gradient,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        workspace.control.step_direction,
        workspace.control.accepted_step,
    )
    _batched_hamiltonian!(workspace.proposed_energy, workspace.proposed_logjoint, workspace.proposal_momentum, inverse_mass_matrix)

    fill!(workspace.continuation_log_weight, -Inf)
    fill!(workspace.continuation_accept_stat_sum, 0.0)
    fill!(workspace.continuation_accept_stat_count, 0)
    fill!(workspace.accept_prob, 0.0)
    fill!(workspace.control.accepted_step, false)
    fill!(workspace.control.divergent_step, false)
    fill!(workspace.control.continuation_turning, false)
    fill!(workspace.control.tree_depths, 1)
    fill!(workspace.control.integration_steps, 0)
    copyto!(workspace.tree_current_logjoint, current_logjoint)
    copyto!(workspace.left_logjoint, current_logjoint)
    copyto!(workspace.right_logjoint, current_logjoint)
    copyto!(workspace.continuation_proposal_logjoint, current_logjoint)
    copyto!(workspace.continuation_proposed_energy, workspace.current_energy)
    fill!(workspace.continuation_delta_energy, 0.0)
    _load_batched_nuts_first_states!(workspace, position, current_logjoint, current_gradient, trues(num_chains))

    for chain_index in 1:num_chains
        tree_workspace = workspace.column_tree_workspaces[chain_index]
        moved, divergent = _initialize_batched_nuts_first_step!(
            workspace,
            chain_index,
            tree_workspace.current,
            tree_workspace.next,
            workspace.control.step_valid[chain_index],
            workspace.control.step_direction[chain_index],
            workspace.current_energy[chain_index],
            inverse_mass_matrix,
            max_delta_energy,
            rng,
        )
        workspace.control.divergent_step[chain_index] = divergent
        workspace.control.accepted_step[chain_index] = moved
    end
    fill!(workspace.subtree_active, false)
    for chain_index in 1:num_chains
        workspace.subtree_active[chain_index] = workspace.control.divergent_step[chain_index] || !workspace.control.accepted_step[chain_index]
    end
    _copy_masked_nuts_buffers!(
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        position,
        workspace.current_momentum,
        current_gradient,
        workspace.subtree_active,
    )
    _copy_masked_values!(workspace.proposed_logjoint, current_logjoint, workspace.subtree_active)
    _copy_masked_values!(workspace.continuation_proposal_logjoint, current_logjoint, workspace.subtree_active)
    return workspace
end

function _batched_nuts_leapfrog_step!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    gradient::AbstractMatrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    direction::AbstractVector{Int},
    active::AbstractVector{Bool},
)
    return _batched_nuts_leapfrog_step_to!(
        workspace,
        model,
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        workspace.proposed_logjoint,
        position,
        momentum,
        gradient,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        direction,
        active,
    )
end

function _batched_nuts_leapfrog_step_to!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    destination_position::AbstractMatrix{Float64},
    destination_momentum::AbstractMatrix{Float64},
    destination_gradient::AbstractMatrix{Float64},
    destination_logjoint::AbstractVector{Float64},
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    gradient::AbstractMatrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    direction::AbstractVector{Int},
    active::AbstractVector{Bool},
)
    q = destination_position
    p = destination_momentum
    proposed_gradient = destination_gradient
    valid = workspace.control.step_valid
    num_chains = size(position, 2)
    size(position) == size(momentum) == size(gradient) ||
        throw(DimensionMismatch("batched NUTS leapfrog requires matching position, momentum, and gradient matrices"))
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(proposed_gradient) == size(position) ||
        throw(DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(proposed_gradient))"))
    length(direction) == num_chains ||
        throw(DimensionMismatch("expected $num_chains batched NUTS directions, got $(length(direction))"))
    length(active) == num_chains ||
        throw(DimensionMismatch("expected $num_chains batched NUTS activity flags, got $(length(active))"))

    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, false)
    for chain_index in 1:num_chains
        active[chain_index] || continue
        valid[chain_index] = true
        signed_step = direction[chain_index] * step_size
        p[:, chain_index] .+= (signed_step / 2) .* gradient[:, chain_index]
        q[:, chain_index] .+= signed_step .* (inverse_mass_matrix .* p[:, chain_index])
    end

    proposed_logjoint, new_gradient = _batched_logjoint_and_gradient_unconstrained!(
        destination_logjoint,
        workspace.gradient_cache,
        q,
    )
    copyto!(proposed_gradient, new_gradient)
    for chain_index in 1:num_chains
        valid[chain_index] || continue
        if !isfinite(proposed_logjoint[chain_index]) || !all(isfinite, view(proposed_gradient, :, chain_index))
            valid[chain_index] = false
        else
            signed_step = direction[chain_index] * step_size
            p[:, chain_index] .+= (signed_step / 2) .* proposed_gradient[:, chain_index]
        end
    end

    return q, p, workspace.proposed_logjoint, proposed_gradient, valid
end

function _dual_averaging_state(step_size::Float64, target_accept::Float64)
    log_step_size = log(step_size)
    return DualAveragingState(
        target_accept,
        0.05,
        10.0,
        0.75,
        log(10 * step_size),
        log_step_size,
        log_step_size,
        0.0,
        0,
    )
end

function _update_step_size!(state::DualAveragingState, accept_prob::Float64)
    state.iteration += 1
    eta = 1 / (state.iteration + state.t0)
    state.hbar = (1 - eta) * state.hbar + eta * (state.target_accept - accept_prob)
    state.log_step_size = state.mu - sqrt(state.iteration) / state.gamma * state.hbar
    eta_bar = state.iteration^(-state.kappa)
    state.log_step_size_avg = eta_bar * state.log_step_size + (1 - eta_bar) * state.log_step_size_avg
    return exp(state.log_step_size)
end

function _final_step_size(state::DualAveragingState)
    return exp(state.log_step_size_avg)
end

function _running_variance_state(num_params::Int, window_length::Int=_RUNNING_VARIANCE_CLIP_START + 16)
    window_length > 0 || throw(ArgumentError("running variance state requires window_length > 0"))
    return RunningVarianceState(
        zeros(num_params),
        zeros(num_params),
        zeros(num_params),
        window_length,
        0,
        0.0,
        0.0,
    )
end

function _warmup_window_length(schedule::WarmupSchedule, window_index::Int)
    1 <= window_index <= length(schedule.slow_window_ends) ||
        throw(BoundsError(schedule.slow_window_ends, window_index))
    window_start = window_index == 1 ? schedule.initial_buffer + 1 : schedule.slow_window_ends[window_index - 1] + 1
    return schedule.slow_window_ends[window_index] - window_start + 1
end

function _warmup_window_start(schedule::WarmupSchedule, window_index::Int)
    1 <= window_index <= length(schedule.slow_window_ends) ||
        throw(BoundsError(schedule.slow_window_ends, window_index))
    return window_index == 1 ? schedule.initial_buffer + 1 : schedule.slow_window_ends[window_index - 1] + 1
end

function _running_variance_window_progress(state::RunningVarianceState)
    if state.window_length <= _RUNNING_VARIANCE_CLIP_START
        return 1.0
    end
    return min(
        max(state.count - _RUNNING_VARIANCE_CLIP_START, 0) /
        (state.window_length - _RUNNING_VARIANCE_CLIP_START),
        1.0,
    )
end

function _running_variance_clip_scale(state::RunningVarianceState)
    state.count <= _RUNNING_VARIANCE_CLIP_START && return _RUNNING_VARIANCE_CLIP_SCALE_EARLY
    progress = _running_variance_window_progress(state)
    return _RUNNING_VARIANCE_CLIP_SCALE_EARLY +
        (_RUNNING_VARIANCE_CLIP_SCALE_LATE - _RUNNING_VARIANCE_CLIP_SCALE_EARLY) * progress
end

function _running_variance_sample!(
    state::RunningVarianceState,
    sample::AbstractVector,
)
    clipped_sample = state.clipped_sample
    if state.count < _RUNNING_VARIANCE_CLIP_START
        copyto!(clipped_sample, sample)
        return clipped_sample
    end

    clip_scale = _running_variance_clip_scale(state)
    @inbounds for index in eachindex(clipped_sample, sample, state.mean, state.m2)
        variance = state.m2[index] / max(state.count - 1, 1)
        bound = clip_scale * sqrt(max(variance, _RUNNING_VARIANCE_FLOOR))
        delta = sample[index] - state.mean[index]
        clipped_sample[index] = state.mean[index] + clamp(delta, -bound, bound)
    end
    return clipped_sample
end

function _warmup_schedule(num_warmup::Int)
    num_warmup < 0 && throw(ArgumentError("warmup schedule requires num_warmup >= 0"))
    num_warmup == 0 && return WarmupSchedule(0, Int[], 0)

    if num_warmup < 20
        initial_buffer = min(5, max(num_warmup - 1, 0))
        if initial_buffer == num_warmup
            return WarmupSchedule(num_warmup, Int[], 0)
        end
        return WarmupSchedule(initial_buffer, [num_warmup], 0)
    end

    initial_buffer = min(max(5, fld(num_warmup, 10)), num_warmup)
    terminal_buffer = min(max(5, fld(num_warmup, 10)), max(num_warmup - initial_buffer, 0))
    slow_budget = num_warmup - initial_buffer - terminal_buffer
    if slow_budget <= 0
        return WarmupSchedule(num_warmup, Int[], 0)
    end

    window_ends = Int[]
    next_window_size = min(25, slow_budget)
    window_start = initial_buffer + 1
    remaining = slow_budget
    while remaining > 0
        window_size = remaining <= fld(3 * next_window_size, 2) ? remaining : next_window_size
        window_end = window_start + window_size - 1
        push!(window_ends, window_end)
        remaining -= window_size
        window_start = window_end + 1
        next_window_size *= 2
    end

    return WarmupSchedule(initial_buffer, window_ends, terminal_buffer)
end

function _running_variance_effective_count(state::RunningVarianceState)
    state.weight_square_sum <= 0 && return 0.0
    return state.weight_sum^2 / state.weight_square_sum
end

function _mass_adaptation_window_summary(
    schedule::WarmupSchedule,
    window_index::Int,
    state::RunningVarianceState,
    inverse_mass_matrix::AbstractVector,
    updated::Bool,
)
    mass_matrix = 1 ./ inverse_mass_matrix
    pooled_samples = state.count
    return HMCMassAdaptationWindowSummary(
        window_index,
        _warmup_window_start(schedule, window_index),
        schedule.slow_window_ends[window_index],
        state.window_length,
        pooled_samples,
        state.weight_sum,
        _running_variance_effective_count(state),
        pooled_samples == 0 ? 0.0 : state.weight_sum / pooled_samples,
        _RUNNING_VARIANCE_CLIP_SCALE_EARLY,
        _running_variance_clip_scale(state),
        updated,
        sum(mass_matrix) / length(mass_matrix),
        minimum(mass_matrix),
        maximum(mass_matrix),
    )
end

function _update_running_variance!(
    state::RunningVarianceState,
    sample::AbstractVector,
    weight::Real,
)
    weight_value = Float64(weight)
    0 <= weight_value <= 1 || throw(ArgumentError("running variance weight must lie in [0, 1], got $weight"))
    iszero(weight_value) && return nothing

    update_sample = _running_variance_sample!(state, sample)
    state.count += 1
    new_weight_sum = state.weight_sum + weight_value
    @inbounds for index in eachindex(update_sample, state.mean, state.m2)
        delta = update_sample[index] - state.mean[index]
        state.mean[index] += (weight_value / new_weight_sum) * delta
        delta2 = update_sample[index] - state.mean[index]
        state.m2[index] += weight_value * delta * delta2
    end
    state.weight_sum = new_weight_sum
    state.weight_square_sum += weight_value^2
    return nothing
end

function _update_running_variance!(state::RunningVarianceState, sample::AbstractVector)
    _update_running_variance!(state, sample, 1.0)
    return nothing
end

function _update_running_variance!(
    state::RunningVarianceState,
    samples::AbstractMatrix,
    weights::AbstractVector,
)
    size(samples, 2) == length(weights) ||
        throw(DimensionMismatch("expected $(size(samples, 2)) running variance weights, got $(length(weights))"))
    for column_index in axes(samples, 2)
        _update_running_variance!(state, view(samples, :, column_index), weights[column_index])
    end
    return nothing
end

function _inverse_mass_matrix(state::RunningVarianceState, regularization::Float64)
    effective_count = _running_variance_effective_count(state)
    if effective_count < 2
        return ones(length(state.mean))
    end

    variance_denom = state.weight_sum - state.weight_square_sum / state.weight_sum
    variance_denom <= 0 && return ones(length(state.mean))
    variance = state.m2 ./ variance_denom
    shrinkage = effective_count / (effective_count + 5.0)
    regularized_variance = shrinkage .* variance .+ (1 - shrinkage)
    return 1 ./ max.(regularized_variance, regularization)
end

function _mass_adaptation_weight(
    state::RunningVarianceState,
    accepted_step::Bool,
    accept_prob::Real,
    divergent_step::Bool,
)
    divergent_step && return 0.0
    accepted_step && return 1.0
    if !isfinite(accept_prob)
        return 0.0
    end
    progress = _running_variance_window_progress(state)
    rejection_weight = clamp(Float64(accept_prob), 0.0, 1.0)
    return _RUNNING_VARIANCE_REJECTION_WEIGHT_EARLY +
        (rejection_weight - _RUNNING_VARIANCE_REJECTION_WEIGHT_EARLY) * progress
end

function _mass_adaptation_weights!(
    state::RunningVarianceState,
    destination::AbstractVector,
    accepted_step::AbstractVector,
    accept_prob::AbstractVector,
    divergent_step::AbstractVector,
)
    length(destination) == length(accepted_step) == length(accept_prob) == length(divergent_step) ||
        throw(DimensionMismatch("mass adaptation weights require matching vector lengths"))
    @inbounds for index in eachindex(destination, accepted_step, accept_prob, divergent_step)
        destination[index] = _mass_adaptation_weight(
            state,
            accepted_step[index],
            accept_prob[index],
            divergent_step[index],
        )
    end
    return destination
end

function _acceptance_probability(log_accept_ratio::Float64)
    if !isfinite(log_accept_ratio)
        return 0.0
    elseif log_accept_ratio >= 0
        return 1.0
    end
    return exp(log_accept_ratio)
end

function _proposal_log_accept_ratio(
    current_logjoint::Float64,
    current_momentum::AbstractVector,
    proposal,
    inverse_mass_matrix::AbstractVector,
)
    isnothing(proposal) && return -Inf
    _, proposed_momentum, proposed_logjoint = proposal
    current_hamiltonian = _hamiltonian(current_logjoint, current_momentum, inverse_mass_matrix)
    proposed_hamiltonian = _hamiltonian(proposed_logjoint, proposed_momentum, inverse_mass_matrix)
    return current_hamiltonian - proposed_hamiltonian
end

function _proposal_diagnostics(
    current_logjoint::Float64,
    current_momentum::AbstractVector,
    proposal,
    inverse_mass_matrix::AbstractVector,
    divergence_threshold::Float64,
)
    current_hamiltonian = _hamiltonian(current_logjoint, current_momentum, inverse_mass_matrix)
    if isnothing(proposal)
        return current_hamiltonian, Inf, true
    end

    _, proposed_momentum, proposed_logjoint = proposal
    proposed_hamiltonian = _hamiltonian(proposed_logjoint, proposed_momentum, inverse_mass_matrix)
    energy_error = proposed_hamiltonian - current_hamiltonian
    divergent = !isfinite(energy_error) || abs(energy_error) > divergence_threshold
    return proposed_hamiltonian, energy_error, divergent
end

function _find_reasonable_batched_step_size(
    workspace::BatchedHMCWorkspace,
    model::TeaModel,
    position::Matrix{Float64},
    current_logjoint::AbstractVector,
    current_gradient::Matrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    reasonable_step_size = step_size
    min_step_size = 1e-8
    max_step_size = 1e3
    target_accept = 0.5

    for _ in 0:20
        _update_sqrt_inverse_mass_matrix!(workspace.sqrt_inverse_mass_matrix, inverse_mass_matrix)
        _sample_batched_momentum!(workspace.momentum, rng, workspace.sqrt_inverse_mass_matrix)
        _, proposal_momentum, proposed_logjoint, _, valid = _batched_leapfrog!(
            workspace,
            model,
            position,
            current_gradient,
            inverse_mass_matrix,
            args,
            constraints,
            reasonable_step_size,
            1,
        )

        current_hamiltonian = _batched_hamiltonian!(
            workspace.current_hamiltonian,
            current_logjoint,
            workspace.momentum,
            inverse_mass_matrix,
        )
        proposed_hamiltonian = workspace.proposed_hamiltonian
        copyto!(proposed_hamiltonian, current_hamiltonian)
        log_accept_ratio = workspace.log_accept_ratio
        fill!(log_accept_ratio, -Inf)
        divergent_step = workspace.divergent_step
        fill!(divergent_step, true)
        for chain_index in eachindex(current_logjoint)
            if valid[chain_index]
                proposed_hamiltonian[chain_index] = _hamiltonian(
                    proposed_logjoint[chain_index],
                    view(proposal_momentum, :, chain_index),
                    inverse_mass_matrix,
                )
                log_accept_ratio[chain_index] =
                    current_hamiltonian[chain_index] - proposed_hamiltonian[chain_index]
                energy_error = proposed_hamiltonian[chain_index] - current_hamiltonian[chain_index]
                divergent_step[chain_index] =
                    !isfinite(energy_error) || abs(energy_error) > divergence_threshold
            end
        end

        accept_prob = _batched_acceptance_probability!(workspace.accept_prob, log_accept_ratio)
        mean_accept_prob = _mean_batched_adaptation_probability(accept_prob, divergent_step)
        direction = mean_accept_prob > target_accept ? 1.0 : -1.0
        next_step_size = reasonable_step_size * (2.0 ^ direction)
        if next_step_size < min_step_size || next_step_size > max_step_size
            break
        end

        if (direction > 0 && mean_accept_prob <= target_accept) ||
           (direction < 0 && mean_accept_prob >= target_accept)
            break
        end
        reasonable_step_size = next_step_size
    end

    return clamp(reasonable_step_size, min_step_size, max_step_size)
end

function _find_reasonable_step_size(
    model::TeaModel,
    position::Vector{Float64},
    current_logjoint::Float64,
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    rng::AbstractRNG,
)
    reasonable_step_size = step_size
    min_step_size = 1e-8
    max_step_size = 1e3
    log_target = log(0.5)
    momentum = _sample_momentum(rng, inverse_mass_matrix)
    proposal = _leapfrog(
        model,
        position,
        momentum,
        gradient_cache,
        inverse_mass_matrix,
        args,
        constraints,
        reasonable_step_size,
        1,
    )
    log_accept_ratio = _proposal_log_accept_ratio(current_logjoint, momentum, proposal, inverse_mass_matrix)
    direction = log_accept_ratio > log_target ? 1.0 : -1.0

    for _ in 1:20
        next_step_size = reasonable_step_size * (2.0 ^ direction)
        if next_step_size < min_step_size || next_step_size > max_step_size
            break
        end

        reasonable_step_size = next_step_size
        momentum = _sample_momentum(rng, inverse_mass_matrix)
        proposal = _leapfrog(
            model,
            position,
            momentum,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            reasonable_step_size,
            1,
        )
        log_accept_ratio = _proposal_log_accept_ratio(current_logjoint, momentum, proposal, inverse_mass_matrix)
        if (direction > 0 && log_accept_ratio <= log_target) ||
           (direction < 0 && log_accept_ratio >= log_target)
            break
        end
    end

    return clamp(reasonable_step_size, min_step_size, max_step_size)
end

function _chain_initial_params(initial_params, chain_index::Int, num_params::Int, num_chains::Int)
    if isnothing(initial_params)
        return nothing
    elseif initial_params isa AbstractMatrix
        size(initial_params) == (num_params, num_chains) ||
            throw(DimensionMismatch("expected initial_params matrix of size ($num_params, $num_chains), got $(size(initial_params))"))
        return collect(Float64, view(initial_params, :, chain_index))
    elseif initial_params isa AbstractVector && !isempty(initial_params) && first(initial_params) isa AbstractVector
        length(initial_params) == num_chains ||
            throw(DimensionMismatch("expected $num_chains initial parameter vectors, got $(length(initial_params))"))
        chain_params = initial_params[chain_index]
        length(chain_params) == num_params ||
            throw(DimensionMismatch("expected $num_params initial parameters for chain $chain_index, got $(length(chain_params))"))
        return Float64[value for value in chain_params]
    elseif initial_params isa AbstractVector
        length(initial_params) == num_params ||
            throw(DimensionMismatch("expected $num_params initial parameters, got $(length(initial_params))"))
        return Float64[value for value in initial_params]
    end

    throw(ArgumentError("unsupported initial_params container for multi-chain HMC"))
end

