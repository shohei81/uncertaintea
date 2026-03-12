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

struct BatchedNUTSKernelBufferLifecycle
    buffer::BatchedNUTSKernelBuffer
    alias_class::BatchedNUTSKernelAliasClass
    first_stage::Int
    last_stage::Int
    first_read_stage::Int
    last_read_stage::Int
    first_write_stage::Int
    last_write_stage::Int
end

struct BatchedNUTSKernelScheduleStage{D<:AbstractBatchedNUTSKernelDataflow,T}
    index::Int
    dataflow::D
    dependencies::T
end

struct BatchedNUTSKernelSchedule{S,L}
    stages::S
    lifecycles::L
end

struct BatchedNUTSKernelResourceGroup{B}
    alias_class::BatchedNUTSKernelAliasClass
    buffers::B
    first_stage::Int
    last_stage::Int
end

@enum BatchedNUTSKernelBarrierKind::UInt8 begin
    NUTSKernelDependencyBarrier = 0
end

struct BatchedNUTSKernelBarrierPlacement{A,B}
    after_stage::Int
    kind::BatchedNUTSKernelBarrierKind
    alias_classes::A
    buffers::B
end

struct BatchedNUTSKernelResourcePlan{S,R,B}
    schedule::S
    resources::R
    barriers::B
end

_batched_nuts_kernel_stage_dataflow(stage::BatchedNUTSKernelScheduleStage) = stage.dataflow
_batched_nuts_kernel_stage_dependencies(stage::BatchedNUTSKernelScheduleStage) = stage.dependencies
_batched_nuts_kernel_schedule_stages(schedule::BatchedNUTSKernelSchedule) = schedule.stages
_batched_nuts_kernel_schedule_lifecycles(schedule::BatchedNUTSKernelSchedule) = schedule.lifecycles
_batched_nuts_kernel_resource_groups(plan::BatchedNUTSKernelResourcePlan) = plan.resources
_batched_nuts_kernel_barriers(plan::BatchedNUTSKernelResourcePlan) = plan.barriers

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

function _batched_nuts_kernel_stage_dependencies(
    dependencies::Tuple,
    stage_index::Int,
)
    selected = BatchedNUTSKernelDependency[]
    for dependency in dependencies
        dependency.consumer_step == stage_index || continue
        push!(selected, dependency)
    end
    return Tuple(selected)
end

function _batched_nuts_kernel_buffer_lifecycles(
    dataflows::Tuple,
)
    seen = Set{BatchedNUTSKernelBuffer}()
    ordered_buffers = BatchedNUTSKernelBuffer[]
    for dataflow in dataflows
        for buffer in _batched_nuts_kernel_reads(dataflow)
            if !(buffer in seen)
                push!(ordered_buffers, buffer)
                push!(seen, buffer)
            end
        end
        for buffer in _batched_nuts_kernel_writes(dataflow)
            if !(buffer in seen)
                push!(ordered_buffers, buffer)
                push!(seen, buffer)
            end
        end
    end

    lifecycles = BatchedNUTSKernelBufferLifecycle[]
    for buffer in ordered_buffers
        first_stage = 0
        last_stage = 0
        first_read_stage = 0
        last_read_stage = 0
        first_write_stage = 0
        last_write_stage = 0
        for stage_index in eachindex(dataflows)
            reads = _batched_nuts_kernel_reads(dataflows[stage_index])
            writes = _batched_nuts_kernel_writes(dataflows[stage_index])
            touched = false
            if buffer in reads
                first_read_stage == 0 && (first_read_stage = stage_index)
                last_read_stage = stage_index
                touched = true
            end
            if buffer in writes
                first_write_stage == 0 && (first_write_stage = stage_index)
                last_write_stage = stage_index
                touched = true
            end
            if touched
                first_stage == 0 && (first_stage = stage_index)
                last_stage = stage_index
            end
        end
        push!(
            lifecycles,
            BatchedNUTSKernelBufferLifecycle(
                buffer,
                _batched_nuts_kernel_alias_class(buffer),
                first_stage,
                last_stage,
                first_read_stage,
                last_read_stage,
                first_write_stage,
                last_write_stage,
            ),
        )
    end
    return Tuple(lifecycles)
end

function _batched_nuts_kernel_schedule(
    program::AbstractBatchedNUTSKernelProgram,
)
    dataflows = _batched_nuts_kernel_dataflows(program)
    dependencies = _batched_nuts_kernel_dependencies(program)
    stages = ntuple(
        index -> BatchedNUTSKernelScheduleStage(
            index,
            dataflows[index],
            _batched_nuts_kernel_stage_dependencies(dependencies, index),
        ),
        length(dataflows),
    )
    return BatchedNUTSKernelSchedule(
        stages,
        _batched_nuts_kernel_buffer_lifecycles(dataflows),
    )
end

function _batched_nuts_kernel_resource_groups(
    lifecycles::Tuple,
)
    ordered_aliases = BatchedNUTSKernelAliasClass[]
    seen = Set{BatchedNUTSKernelAliasClass}()
    for lifecycle in lifecycles
        alias_class = lifecycle.alias_class
        if !(alias_class in seen)
            push!(ordered_aliases, alias_class)
            push!(seen, alias_class)
        end
    end

    resources = BatchedNUTSKernelResourceGroup[]
    for alias_class in ordered_aliases
        buffers = BatchedNUTSKernelBuffer[]
        first_stage = typemax(Int)
        last_stage = 0
        for lifecycle in lifecycles
            lifecycle.alias_class == alias_class || continue
            push!(buffers, lifecycle.buffer)
            first_stage = min(first_stage, lifecycle.first_stage)
            last_stage = max(last_stage, lifecycle.last_stage)
        end
        push!(
            resources,
            BatchedNUTSKernelResourceGroup(
                alias_class,
                Tuple(buffers),
                first_stage == typemax(Int) ? 0 : first_stage,
                last_stage,
            ),
        )
    end
    return Tuple(resources)
end

function _batched_nuts_kernel_barriers(
    schedule::BatchedNUTSKernelSchedule,
    dependencies::Tuple,
)
    barriers = BatchedNUTSKernelBarrierPlacement[]
    num_stages = length(schedule.stages)
    for stage_index in 1:num_stages
        alias_classes = BatchedNUTSKernelAliasClass[]
        buffers = BatchedNUTSKernelBuffer[]
        for dependency in dependencies
            dependency.producer_step == stage_index || continue
            push!(alias_classes, dependency.alias_class)
            push!(buffers, dependency.buffer)
        end
        isempty(alias_classes) && continue
        push!(
            barriers,
            BatchedNUTSKernelBarrierPlacement(
                stage_index,
                NUTSKernelDependencyBarrier,
                Tuple(unique(alias_classes)),
                Tuple(unique(buffers)),
            ),
        )
    end
    return Tuple(barriers)
end

function _batched_nuts_kernel_resource_plan(
    program::AbstractBatchedNUTSKernelProgram,
)
    schedule = _batched_nuts_kernel_schedule(program)
    dependencies = _batched_nuts_kernel_dependencies(program)
    return BatchedNUTSKernelResourcePlan(
        schedule,
        _batched_nuts_kernel_resource_groups(_batched_nuts_kernel_schedule_lifecycles(schedule)),
        _batched_nuts_kernel_barriers(schedule, dependencies),
    )
end

function _batched_nuts_kernel_barriers_after(
    plan::BatchedNUTSKernelResourcePlan,
    stage_index::Int,
)
    placements = BatchedNUTSKernelBarrierPlacement[]
    for barrier in plan.barriers
        barrier.after_stage == stage_index || continue
        push!(placements, barrier)
    end
    return Tuple(placements)
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
