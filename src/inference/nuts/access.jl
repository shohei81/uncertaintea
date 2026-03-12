struct BatchedNUTSIdleKernelAccess <: AbstractBatchedNUTSKernelAccess
    block::BatchedNUTSIdleControlBlock
end

struct BatchedNUTSExpandKernelAccess <: AbstractBatchedNUTSKernelAccess
    block::BatchedNUTSExpandControlBlock
    copy_left::BitVector
    copy_right::BitVector
    select_proposal::BitVector
    turning::BitVector
    log_weight::Vector{Float64}
    proposed_energy::Vector{Float64}
    delta_energy::Vector{Float64}
    proposal_energy::Vector{Float64}
    proposal_energy_error::Vector{Float64}
    accept_prob::Vector{Float64}
    candidate_log_weight::Vector{Float64}
    combined_log_weight::Vector{Float64}
    current_position::Matrix{Float64}
    current_momentum::Matrix{Float64}
    current_gradient::Matrix{Float64}
    current_logjoint::Vector{Float64}
    next_position::Matrix{Float64}
    next_momentum::Matrix{Float64}
    next_gradient::Matrix{Float64}
    proposed_logjoint::Vector{Float64}
    left_position::Matrix{Float64}
    left_momentum::Matrix{Float64}
    left_gradient::Matrix{Float64}
    left_logjoint::Vector{Float64}
    right_position::Matrix{Float64}
    right_momentum::Matrix{Float64}
    right_gradient::Matrix{Float64}
    right_logjoint::Vector{Float64}
    proposal_position::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    proposal_logjoint::Vector{Float64}
    current_energy::Vector{Float64}
end

struct BatchedNUTSMergeKernelAccess <: AbstractBatchedNUTSKernelAccess
    block::BatchedNUTSMergeControlBlock
    select_proposal::BitVector
    merged_turning::BitVector
    proposal_energy::Vector{Float64}
    proposal_energy_error::Vector{Float64}
    candidate_log_weight::Vector{Float64}
    combined_log_weight::Vector{Float64}
    left_position::Matrix{Float64}
    left_momentum::Matrix{Float64}
    left_gradient::Matrix{Float64}
    left_logjoint::Vector{Float64}
    right_position::Matrix{Float64}
    right_momentum::Matrix{Float64}
    right_gradient::Matrix{Float64}
    right_logjoint::Vector{Float64}
    tree_proposal_position::Matrix{Float64}
    tree_proposal_momentum::Matrix{Float64}
    tree_proposal_gradient::Matrix{Float64}
    tree_proposal_logjoint::Vector{Float64}
    proposal_position::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    proposed_logjoint::Vector{Float64}
    continuation_proposal_logjoint::Vector{Float64}
    continuation_log_weight::Vector{Float64}
    current_energy::Vector{Float64}
end

struct BatchedNUTSDoneKernelAccess <: AbstractBatchedNUTSKernelAccess
    block::BatchedNUTSDoneControlBlock
end

_batched_nuts_access_control_block(access::BatchedNUTSIdleKernelAccess) = access.block
_batched_nuts_access_control_block(access::BatchedNUTSExpandKernelAccess) = access.block
_batched_nuts_access_control_block(access::BatchedNUTSMergeKernelAccess) = access.block
_batched_nuts_access_control_block(access::BatchedNUTSDoneKernelAccess) = access.block

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSIdleKernelAccess,
)
    return BatchedNUTSIdleKernelProgram(access, _BATCHED_NUTS_IDLE_KERNEL_OPS)
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
)
    return BatchedNUTSExpandKernelProgram(access, _BATCHED_NUTS_EXPAND_KERNEL_OPS)
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSMergeKernelAccess,
)
    return BatchedNUTSMergeKernelProgram(access, _BATCHED_NUTS_MERGE_KERNEL_OPS)
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSDoneKernelAccess,
)
    return BatchedNUTSDoneKernelProgram(access, _BATCHED_NUTS_DONE_KERNEL_OPS)
end

function _batched_nuts_kernel_access(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_kernel_access(
        workspace,
        _batched_nuts_kernel_frame(workspace),
    )
end

function _batched_nuts_kernel_access(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSIdleKernelFrame,
)
    return BatchedNUTSIdleKernelAccess(frame.state.descriptor.block)
end

function _batched_nuts_kernel_access(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSExpandKernelFrame,
)
    state = frame.state
    descriptor = state.descriptor
    return BatchedNUTSExpandKernelAccess(
        descriptor.block,
        descriptor.copy_left,
        descriptor.copy_right,
        descriptor.select_proposal,
        descriptor.turning,
        state.log_weight,
        state.proposed_energy,
        state.delta_energy,
        state.proposal_energy,
        state.proposal_energy_error,
        state.accept_prob,
        state.candidate_log_weight,
        state.combined_log_weight,
        frame.current_position,
        frame.current_momentum,
        frame.current_gradient,
        frame.current_logjoint,
        frame.next_position,
        frame.next_momentum,
        frame.next_gradient,
        frame.proposed_logjoint,
        frame.left_position,
        frame.left_momentum,
        frame.left_gradient,
        frame.left_logjoint,
        frame.right_position,
        frame.right_momentum,
        frame.right_gradient,
        frame.right_logjoint,
        frame.proposal_position,
        frame.proposal_momentum,
        frame.proposal_gradient,
        frame.proposal_logjoint,
        frame.current_energy,
    )
end

function _batched_nuts_kernel_access(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSMergeKernelFrame,
)
    state = frame.state
    descriptor = state.descriptor
    return BatchedNUTSMergeKernelAccess(
        descriptor.block,
        descriptor.select_proposal,
        descriptor.merged_turning,
        state.proposal_energy,
        state.proposal_energy_error,
        state.candidate_log_weight,
        state.combined_log_weight,
        frame.left_position,
        frame.left_momentum,
        frame.left_gradient,
        frame.left_logjoint,
        frame.right_position,
        frame.right_momentum,
        frame.right_gradient,
        frame.right_logjoint,
        frame.tree_proposal_position,
        frame.tree_proposal_momentum,
        frame.tree_proposal_gradient,
        frame.tree_proposal_logjoint,
        frame.proposal_position,
        frame.proposal_momentum,
        frame.proposal_gradient,
        frame.proposed_logjoint,
        frame.continuation_proposal_logjoint,
        frame.continuation_log_weight,
        frame.current_energy,
    )
end

function _batched_nuts_kernel_access(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSDoneKernelFrame,
)
    return BatchedNUTSDoneKernelAccess(frame.state.descriptor.block)
end

_batched_nuts_kernel_access(program::AbstractBatchedNUTSKernelProgram) = program.access
