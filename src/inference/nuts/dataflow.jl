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

@enum BatchedNUTSKernelAliasClass::UInt8 begin
    NUTSKernelAliasControlBlock = 0
    NUTSKernelAliasSchedulerState = 1
    NUTSKernelAliasControlState = 2
    NUTSKernelAliasDescriptorScratch = 3
    NUTSKernelAliasTreeState = 4
    NUTSKernelAliasTreeEnergy = 5
    NUTSKernelAliasSubtreeSummary = 6
    NUTSKernelAliasContinuationState = 7
    NUTSKernelAliasContinuationSummary = 8
end

_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferControlBlock}) = NUTSKernelAliasControlBlock
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferSchedulerState}) = NUTSKernelAliasSchedulerState
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferControlState}) = NUTSKernelAliasControlState
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferDescriptorScratch}) = NUTSKernelAliasDescriptorScratch
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferTreeCurrentState}) = NUTSKernelAliasTreeState
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferTreeNextState}) = NUTSKernelAliasTreeState
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferTreeFrontierState}) = NUTSKernelAliasTreeState
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferTreeProposalState}) = NUTSKernelAliasTreeState
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferTreeEnergy}) = NUTSKernelAliasTreeEnergy
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferSubtreeSummary}) = NUTSKernelAliasSubtreeSummary
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferContinuationFrontierState}) = NUTSKernelAliasContinuationState
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferContinuationProposalState}) = NUTSKernelAliasContinuationState
_batched_nuts_kernel_alias_class(::Val{NUTSKernelBufferContinuationSummary}) = NUTSKernelAliasContinuationSummary

function _batched_nuts_kernel_alias_class(buffer::BatchedNUTSKernelBuffer)
    return _batched_nuts_kernel_alias_class(Val(buffer))
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
_batched_nuts_kernel_read_aliases(dataflow::BatchedNUTSKernelDataflow) =
    map(_batched_nuts_kernel_alias_class, dataflow.reads)
_batched_nuts_kernel_write_aliases(dataflow::BatchedNUTSKernelDataflow) =
    map(_batched_nuts_kernel_alias_class, dataflow.writes)

@enum BatchedNUTSKernelDependencyKind::UInt8 begin
    NUTSKernelFlowDependency = 0
    NUTSKernelAntiDependency = 1
    NUTSKernelOutputDependency = 2
end

struct BatchedNUTSKernelDependency
    producer_step::Int
    consumer_step::Int
    kind::BatchedNUTSKernelDependencyKind
    buffer::BatchedNUTSKernelBuffer
    alias_class::BatchedNUTSKernelAliasClass
end

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
            NUTSKernelBufferControlState,
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

const _BATCHED_NUTS_IDLE_KERNEL_DEPENDENCIES = ()

const _BATCHED_NUTS_EXPAND_KERNEL_DEPENDENCIES = (
    BatchedNUTSKernelDependency(
        1,
        2,
        NUTSKernelFlowDependency,
        NUTSKernelBufferControlState,
        NUTSKernelAliasControlState,
    ),
    BatchedNUTSKernelDependency(
        1,
        5,
        NUTSKernelFlowDependency,
        NUTSKernelBufferSchedulerState,
        NUTSKernelAliasSchedulerState,
    ),
    BatchedNUTSKernelDependency(
        2,
        3,
        NUTSKernelFlowDependency,
        NUTSKernelBufferTreeNextState,
        NUTSKernelAliasTreeState,
    ),
    BatchedNUTSKernelDependency(
        2,
        4,
        NUTSKernelFlowDependency,
        NUTSKernelBufferControlState,
        NUTSKernelAliasControlState,
    ),
    BatchedNUTSKernelDependency(
        2,
        4,
        NUTSKernelFlowDependency,
        NUTSKernelBufferTreeNextState,
        NUTSKernelAliasTreeState,
    ),
    BatchedNUTSKernelDependency(
        3,
        4,
        NUTSKernelFlowDependency,
        NUTSKernelBufferTreeEnergy,
        NUTSKernelAliasTreeEnergy,
    ),
)

const _BATCHED_NUTS_MERGE_KERNEL_DEPENDENCIES = (
    BatchedNUTSKernelDependency(
        2,
        3,
        NUTSKernelFlowDependency,
        NUTSKernelBufferControlState,
        NUTSKernelAliasControlState,
    ),
    BatchedNUTSKernelDependency(
        2,
        4,
        NUTSKernelFlowDependency,
        NUTSKernelBufferSchedulerState,
        NUTSKernelAliasSchedulerState,
    ),
)

const _BATCHED_NUTS_DONE_KERNEL_DEPENDENCIES = ()

_batched_nuts_kernel_dependencies(::BatchedNUTSIdleKernelProgram) = _BATCHED_NUTS_IDLE_KERNEL_DEPENDENCIES
_batched_nuts_kernel_dependencies(::BatchedNUTSExpandKernelProgram) = _BATCHED_NUTS_EXPAND_KERNEL_DEPENDENCIES
_batched_nuts_kernel_dependencies(::BatchedNUTSMergeKernelProgram) = _BATCHED_NUTS_MERGE_KERNEL_DEPENDENCIES
_batched_nuts_kernel_dependencies(::BatchedNUTSDoneKernelProgram) = _BATCHED_NUTS_DONE_KERNEL_DEPENDENCIES
