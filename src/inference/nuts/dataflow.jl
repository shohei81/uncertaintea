@enum BatchedNUTSKernelBuffer::UInt8 begin
    NUTSKernelBufferControlBlock = 0
    NUTSKernelBufferSchedulerState = 1
    NUTSKernelBufferControlState = 2
    NUTSKernelBufferDescriptorScratch = 3
    NUTSKernelBufferTreeCurrentState = 4
    NUTSKernelBufferTreeNextState = 5
    NUTSKernelBufferTreeFrontierState = 6
    NUTSKernelBufferTreeProposalState = 7
    NUTSKernelBufferTreeEnergy = 8
    NUTSKernelBufferSubtreeSummary = 9
    NUTSKernelBufferContinuationFrontierState = 10
    NUTSKernelBufferContinuationProposalState = 11
    NUTSKernelBufferContinuationSummary = 12
end

abstract type AbstractBatchedNUTSKernelDataflow end

struct BatchedNUTSKernelDataflow{
    S<:AbstractBatchedNUTSKernelStep,
    A<:AbstractBatchedNUTSKernelAccess,
    R,
    W,
} <: AbstractBatchedNUTSKernelDataflow
    access::A
    step::S
    reads::NTuple{R,BatchedNUTSKernelBuffer}
    writes::NTuple{W,BatchedNUTSKernelBuffer}
end

_batched_nuts_kernel_step(dataflow::BatchedNUTSKernelDataflow) = dataflow.step
_batched_nuts_kernel_access(dataflow::BatchedNUTSKernelDataflow) = dataflow.access
_batched_nuts_kernel_reads(dataflow::BatchedNUTSKernelDataflow) = dataflow.reads
_batched_nuts_kernel_writes(dataflow::BatchedNUTSKernelDataflow) = dataflow.writes

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSIdleKernelAccess,
    step::BatchedNUTSReloadControlStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (NUTSKernelBufferControlBlock,),
        (NUTSKernelBufferSchedulerState, NUTSKernelBufferControlState),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSExpandKernelAccess,
    step::BatchedNUTSReloadControlStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (NUTSKernelBufferControlBlock,),
        (NUTSKernelBufferSchedulerState, NUTSKernelBufferControlState),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSExpandKernelAccess,
    step::BatchedNUTSLeapfrogStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (NUTSKernelBufferControlState, NUTSKernelBufferTreeCurrentState),
        (NUTSKernelBufferControlState, NUTSKernelBufferTreeNextState),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSExpandKernelAccess,
    step::BatchedNUTSHamiltonianStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (NUTSKernelBufferTreeNextState,),
        (NUTSKernelBufferTreeEnergy,),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSExpandKernelAccess,
    step::BatchedNUTSAdvanceStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (
            NUTSKernelBufferControlState,
            NUTSKernelBufferDescriptorScratch,
            NUTSKernelBufferTreeCurrentState,
            NUTSKernelBufferTreeNextState,
            NUTSKernelBufferTreeEnergy,
            NUTSKernelBufferSubtreeSummary,
        ),
        (
            NUTSKernelBufferControlState,
            NUTSKernelBufferDescriptorScratch,
            NUTSKernelBufferTreeCurrentState,
            NUTSKernelBufferTreeFrontierState,
            NUTSKernelBufferTreeProposalState,
            NUTSKernelBufferSubtreeSummary,
        ),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSExpandKernelAccess,
    step::BatchedNUTSTransitionPhaseStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (NUTSKernelBufferSchedulerState,),
        (NUTSKernelBufferSchedulerState,),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSMergeKernelAccess,
    step::BatchedNUTSReloadControlStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (NUTSKernelBufferControlBlock,),
        (NUTSKernelBufferSchedulerState, NUTSKernelBufferControlState),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSMergeKernelAccess,
    step::BatchedNUTSActivateMergeStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (NUTSKernelBufferControlBlock, NUTSKernelBufferSubtreeSummary),
        (NUTSKernelBufferSchedulerState, NUTSKernelBufferControlState),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSMergeKernelAccess,
    step::BatchedNUTSMergeStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (
            NUTSKernelBufferDescriptorScratch,
            NUTSKernelBufferTreeFrontierState,
            NUTSKernelBufferTreeProposalState,
            NUTSKernelBufferSubtreeSummary,
            NUTSKernelBufferContinuationFrontierState,
            NUTSKernelBufferContinuationProposalState,
            NUTSKernelBufferContinuationSummary,
            NUTSKernelBufferTreeEnergy,
        ),
        (
            NUTSKernelBufferDescriptorScratch,
            NUTSKernelBufferControlState,
            NUTSKernelBufferContinuationFrontierState,
            NUTSKernelBufferContinuationProposalState,
            NUTSKernelBufferContinuationSummary,
        ),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSMergeKernelAccess,
    step::BatchedNUTSTransitionPhaseStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (NUTSKernelBufferSchedulerState,),
        (NUTSKernelBufferSchedulerState,),
    )
end

function _batched_nuts_kernel_dataflow(
    access::BatchedNUTSDoneKernelAccess,
    step::BatchedNUTSReloadControlStep,
)
    return BatchedNUTSKernelDataflow(
        access,
        step,
        (NUTSKernelBufferControlBlock,),
        (NUTSKernelBufferSchedulerState, NUTSKernelBufferControlState),
    )
end

function _batched_nuts_kernel_dataflows(
    program::AbstractBatchedNUTSKernelProgram,
)
    access = _batched_nuts_kernel_access(program)
    return map(step -> _batched_nuts_kernel_dataflow(access, step), _batched_nuts_kernel_steps(program))
end
