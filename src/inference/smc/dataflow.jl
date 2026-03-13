@enum TemperedSMCNUTSKernelBuffer::UInt8 begin
    TemperedSMCBufferControlBlock = 0
    TemperedSMCBufferSchedulerState = 1
    TemperedSMCBufferControlState = 2
    TemperedSMCBufferDescriptorScratch = 3
    TemperedSMCBufferCohortState = 4
    TemperedSMCBufferCohortSummary = 5
end

@enum TemperedSMCNUTSKernelAliasClass::UInt8 begin
    TemperedSMCAliasControlBlock = 0
    TemperedSMCAliasSchedulerState = 1
    TemperedSMCAliasControlState = 2
    TemperedSMCAliasDescriptorScratch = 3
    TemperedSMCAliasCohortState = 4
    TemperedSMCAliasCohortSummary = 5
end

@enum TemperedSMCNUTSKernelStepKind::UInt8 begin
    TemperedSMCReloadControlStep = 0
    TemperedSMCExpandCohortStep = 1
    TemperedSMCMergeCohortStep = 2
    TemperedSMCTransitionPhaseStep = 3
end

_tempered_smc_nuts_kernel_alias_class(::Val{TemperedSMCBufferControlBlock}) = TemperedSMCAliasControlBlock
_tempered_smc_nuts_kernel_alias_class(::Val{TemperedSMCBufferSchedulerState}) = TemperedSMCAliasSchedulerState
_tempered_smc_nuts_kernel_alias_class(::Val{TemperedSMCBufferControlState}) = TemperedSMCAliasControlState
_tempered_smc_nuts_kernel_alias_class(::Val{TemperedSMCBufferDescriptorScratch}) = TemperedSMCAliasDescriptorScratch
_tempered_smc_nuts_kernel_alias_class(::Val{TemperedSMCBufferCohortState}) = TemperedSMCAliasCohortState
_tempered_smc_nuts_kernel_alias_class(::Val{TemperedSMCBufferCohortSummary}) = TemperedSMCAliasCohortSummary

function _tempered_smc_nuts_kernel_alias_class(buffer::TemperedSMCNUTSKernelBuffer)
    return _tempered_smc_nuts_kernel_alias_class(Val(buffer))
end

struct TemperedSMCNUTSKernelDataflow{D,R,W}
    descriptor::D
    step::TemperedSMCNUTSKernelStepKind
    reads::NTuple{R,TemperedSMCNUTSKernelBuffer}
    writes::NTuple{W,TemperedSMCNUTSKernelBuffer}
end

_tempered_smc_nuts_kernel_descriptor(dataflow::TemperedSMCNUTSKernelDataflow) = dataflow.descriptor
_tempered_smc_nuts_kernel_step(dataflow::TemperedSMCNUTSKernelDataflow) = dataflow.step
_tempered_smc_nuts_kernel_reads(dataflow::TemperedSMCNUTSKernelDataflow) = dataflow.reads
_tempered_smc_nuts_kernel_writes(dataflow::TemperedSMCNUTSKernelDataflow) = dataflow.writes
_tempered_smc_nuts_kernel_read_aliases(dataflow::TemperedSMCNUTSKernelDataflow) =
    map(_tempered_smc_nuts_kernel_alias_class, dataflow.reads)
_tempered_smc_nuts_kernel_write_aliases(dataflow::TemperedSMCNUTSKernelDataflow) =
    map(_tempered_smc_nuts_kernel_alias_class, dataflow.writes)

@enum TemperedSMCNUTSKernelDependencyKind::UInt8 begin
    TemperedSMCKernelFlowDependency = 0
    TemperedSMCKernelAntiDependency = 1
    TemperedSMCKernelOutputDependency = 2
end

struct TemperedSMCNUTSKernelDependency
    producer_step::Int
    consumer_step::Int
    kind::TemperedSMCNUTSKernelDependencyKind
    buffer::TemperedSMCNUTSKernelBuffer
    alias_class::TemperedSMCNUTSKernelAliasClass
end

struct TemperedSMCNUTSKernelBufferLifecycle
    buffer::TemperedSMCNUTSKernelBuffer
    alias_class::TemperedSMCNUTSKernelAliasClass
    first_stage::Int
    last_stage::Int
    first_read_stage::Int
    last_read_stage::Int
    first_write_stage::Int
    last_write_stage::Int
end

struct TemperedSMCNUTSKernelScheduleStage{D,T}
    index::Int
    dataflow::D
    dependencies::T
end

struct TemperedSMCNUTSKernelSchedule{S,L}
    stages::S
    lifecycles::L
end

struct TemperedSMCNUTSKernelResourceGroup{B}
    alias_class::TemperedSMCNUTSKernelAliasClass
    buffers::B
    first_stage::Int
    last_stage::Int
end

@enum TemperedSMCNUTSKernelBarrierKind::UInt8 begin
    TemperedSMCKernelDependencyBarrier = 0
end

struct TemperedSMCNUTSKernelBarrierPlacement{A,B}
    after_stage::Int
    kind::TemperedSMCNUTSKernelBarrierKind
    alias_classes::A
    buffers::B
end

struct TemperedSMCNUTSKernelResourcePlan{S,R,B}
    schedule::S
    resources::R
    barriers::B
end

_tempered_smc_nuts_kernel_stage_dataflow(stage::TemperedSMCNUTSKernelScheduleStage) = stage.dataflow
_tempered_smc_nuts_kernel_stage_dependencies(stage::TemperedSMCNUTSKernelScheduleStage) = stage.dependencies
_tempered_smc_nuts_kernel_schedule_stages(schedule::TemperedSMCNUTSKernelSchedule) = schedule.stages
_tempered_smc_nuts_kernel_schedule_lifecycles(schedule::TemperedSMCNUTSKernelSchedule) = schedule.lifecycles
_tempered_smc_nuts_kernel_resource_groups(plan::TemperedSMCNUTSKernelResourcePlan) = plan.resources
_tempered_smc_nuts_kernel_barriers(plan::TemperedSMCNUTSKernelResourcePlan) = plan.barriers

function _tempered_smc_nuts_kernel_dataflows(
    descriptor::TemperedNUTSIdleDescriptor,
)
    return (
        TemperedSMCNUTSKernelDataflow(
            descriptor,
            TemperedSMCReloadControlStep,
            (TemperedSMCBufferControlBlock,),
            (TemperedSMCBufferSchedulerState, TemperedSMCBufferControlState),
        ),
    )
end

function _tempered_smc_nuts_kernel_dataflows(
    descriptor::TemperedNUTSExpandDescriptor,
)
    return (
        TemperedSMCNUTSKernelDataflow(
            descriptor,
            TemperedSMCReloadControlStep,
            (TemperedSMCBufferControlBlock,),
            (TemperedSMCBufferSchedulerState, TemperedSMCBufferControlState),
        ),
        TemperedSMCNUTSKernelDataflow(
            descriptor,
            TemperedSMCExpandCohortStep,
            (
                TemperedSMCBufferSchedulerState,
                TemperedSMCBufferControlState,
                TemperedSMCBufferCohortState,
            ),
            (
                TemperedSMCBufferControlState,
                TemperedSMCBufferDescriptorScratch,
                TemperedSMCBufferCohortState,
                TemperedSMCBufferCohortSummary,
            ),
        ),
        TemperedSMCNUTSKernelDataflow(
            descriptor,
            TemperedSMCTransitionPhaseStep,
            (
                TemperedSMCBufferSchedulerState,
                TemperedSMCBufferControlState,
                TemperedSMCBufferCohortSummary,
            ),
            (TemperedSMCBufferSchedulerState, TemperedSMCBufferControlState),
        ),
    )
end

function _tempered_smc_nuts_kernel_dataflows(
    descriptor::TemperedNUTSMergeDescriptor,
)
    return (
        TemperedSMCNUTSKernelDataflow(
            descriptor,
            TemperedSMCReloadControlStep,
            (TemperedSMCBufferControlBlock,),
            (TemperedSMCBufferSchedulerState, TemperedSMCBufferControlState),
        ),
        TemperedSMCNUTSKernelDataflow(
            descriptor,
            TemperedSMCMergeCohortStep,
            (
                TemperedSMCBufferSchedulerState,
                TemperedSMCBufferControlState,
                TemperedSMCBufferDescriptorScratch,
                TemperedSMCBufferCohortState,
                TemperedSMCBufferCohortSummary,
            ),
            (
                TemperedSMCBufferSchedulerState,
                TemperedSMCBufferControlState,
                TemperedSMCBufferDescriptorScratch,
                TemperedSMCBufferCohortSummary,
            ),
        ),
        TemperedSMCNUTSKernelDataflow(
            descriptor,
            TemperedSMCTransitionPhaseStep,
            (
                TemperedSMCBufferSchedulerState,
                TemperedSMCBufferControlState,
                TemperedSMCBufferCohortSummary,
            ),
            (TemperedSMCBufferSchedulerState, TemperedSMCBufferControlState),
        ),
    )
end

function _tempered_smc_nuts_kernel_dataflows(
    descriptor::TemperedNUTSDoneDescriptor,
)
    return (
        TemperedSMCNUTSKernelDataflow(
            descriptor,
            TemperedSMCReloadControlStep,
            (TemperedSMCBufferControlBlock,),
            (TemperedSMCBufferSchedulerState, TemperedSMCBufferControlState),
        ),
    )
end

const _TEMPERED_SMC_NUTS_IDLE_KERNEL_DEPENDENCIES = ()

const _TEMPERED_SMC_NUTS_EXPAND_KERNEL_DEPENDENCIES = (
    TemperedSMCNUTSKernelDependency(
        1,
        2,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferSchedulerState,
        TemperedSMCAliasSchedulerState,
    ),
    TemperedSMCNUTSKernelDependency(
        1,
        2,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferControlState,
        TemperedSMCAliasControlState,
    ),
    TemperedSMCNUTSKernelDependency(
        1,
        3,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferSchedulerState,
        TemperedSMCAliasSchedulerState,
    ),
    TemperedSMCNUTSKernelDependency(
        2,
        3,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferControlState,
        TemperedSMCAliasControlState,
    ),
    TemperedSMCNUTSKernelDependency(
        2,
        3,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferCohortSummary,
        TemperedSMCAliasCohortSummary,
    ),
)

const _TEMPERED_SMC_NUTS_MERGE_KERNEL_DEPENDENCIES = (
    TemperedSMCNUTSKernelDependency(
        1,
        2,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferSchedulerState,
        TemperedSMCAliasSchedulerState,
    ),
    TemperedSMCNUTSKernelDependency(
        1,
        2,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferControlState,
        TemperedSMCAliasControlState,
    ),
    TemperedSMCNUTSKernelDependency(
        2,
        3,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferSchedulerState,
        TemperedSMCAliasSchedulerState,
    ),
    TemperedSMCNUTSKernelDependency(
        2,
        3,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferControlState,
        TemperedSMCAliasControlState,
    ),
    TemperedSMCNUTSKernelDependency(
        2,
        3,
        TemperedSMCKernelFlowDependency,
        TemperedSMCBufferCohortSummary,
        TemperedSMCAliasCohortSummary,
    ),
)

const _TEMPERED_SMC_NUTS_DONE_KERNEL_DEPENDENCIES = ()

_tempered_smc_nuts_kernel_dependencies(::TemperedNUTSIdleDescriptor) =
    _TEMPERED_SMC_NUTS_IDLE_KERNEL_DEPENDENCIES
_tempered_smc_nuts_kernel_dependencies(::TemperedNUTSExpandDescriptor) =
    _TEMPERED_SMC_NUTS_EXPAND_KERNEL_DEPENDENCIES
_tempered_smc_nuts_kernel_dependencies(::TemperedNUTSMergeDescriptor) =
    _TEMPERED_SMC_NUTS_MERGE_KERNEL_DEPENDENCIES
_tempered_smc_nuts_kernel_dependencies(::TemperedNUTSDoneDescriptor) =
    _TEMPERED_SMC_NUTS_DONE_KERNEL_DEPENDENCIES

function _tempered_smc_nuts_kernel_stage_dependencies(
    dependencies::Tuple,
    stage_index::Int,
)
    selected = TemperedSMCNUTSKernelDependency[]
    for dependency in dependencies
        dependency.consumer_step == stage_index || continue
        push!(selected, dependency)
    end
    return Tuple(selected)
end

function _tempered_smc_nuts_kernel_buffer_lifecycles(
    dataflows::Tuple,
)
    seen = Set{TemperedSMCNUTSKernelBuffer}()
    ordered_buffers = TemperedSMCNUTSKernelBuffer[]
    for dataflow in dataflows
        for buffer in _tempered_smc_nuts_kernel_reads(dataflow)
            if !(buffer in seen)
                push!(ordered_buffers, buffer)
                push!(seen, buffer)
            end
        end
        for buffer in _tempered_smc_nuts_kernel_writes(dataflow)
            if !(buffer in seen)
                push!(ordered_buffers, buffer)
                push!(seen, buffer)
            end
        end
    end

    lifecycles = TemperedSMCNUTSKernelBufferLifecycle[]
    for buffer in ordered_buffers
        first_stage = 0
        last_stage = 0
        first_read_stage = 0
        last_read_stage = 0
        first_write_stage = 0
        last_write_stage = 0
        for stage_index in eachindex(dataflows)
            reads = _tempered_smc_nuts_kernel_reads(dataflows[stage_index])
            writes = _tempered_smc_nuts_kernel_writes(dataflows[stage_index])
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
            TemperedSMCNUTSKernelBufferLifecycle(
                buffer,
                _tempered_smc_nuts_kernel_alias_class(buffer),
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

function _tempered_smc_nuts_kernel_schedule(
    descriptor::AbstractTemperedNUTSSchedulerDescriptor,
)
    dataflows = _tempered_smc_nuts_kernel_dataflows(descriptor)
    dependencies = _tempered_smc_nuts_kernel_dependencies(descriptor)
    stages = ntuple(
        index -> TemperedSMCNUTSKernelScheduleStage(
            index,
            dataflows[index],
            _tempered_smc_nuts_kernel_stage_dependencies(dependencies, index),
        ),
        length(dataflows),
    )
    return TemperedSMCNUTSKernelSchedule(
        stages,
        _tempered_smc_nuts_kernel_buffer_lifecycles(dataflows),
    )
end

function _tempered_smc_nuts_kernel_resource_groups(
    lifecycles::Tuple,
)
    ordered_aliases = TemperedSMCNUTSKernelAliasClass[]
    seen = Set{TemperedSMCNUTSKernelAliasClass}()
    for lifecycle in lifecycles
        alias_class = lifecycle.alias_class
        if !(alias_class in seen)
            push!(ordered_aliases, alias_class)
            push!(seen, alias_class)
        end
    end

    resources = TemperedSMCNUTSKernelResourceGroup[]
    for alias_class in ordered_aliases
        buffers = TemperedSMCNUTSKernelBuffer[]
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
            TemperedSMCNUTSKernelResourceGroup(
                alias_class,
                Tuple(buffers),
                first_stage == typemax(Int) ? 0 : first_stage,
                last_stage,
            ),
        )
    end
    return Tuple(resources)
end

function _tempered_smc_nuts_kernel_barriers(
    schedule::TemperedSMCNUTSKernelSchedule,
    dependencies::Tuple,
)
    barriers = TemperedSMCNUTSKernelBarrierPlacement[]
    num_stages = length(schedule.stages)
    for stage_index in 1:num_stages
        alias_classes = TemperedSMCNUTSKernelAliasClass[]
        buffers = TemperedSMCNUTSKernelBuffer[]
        for dependency in dependencies
            dependency.producer_step == stage_index || continue
            push!(alias_classes, dependency.alias_class)
            push!(buffers, dependency.buffer)
        end
        isempty(alias_classes) && continue
        push!(
            barriers,
            TemperedSMCNUTSKernelBarrierPlacement(
                stage_index,
                TemperedSMCKernelDependencyBarrier,
                Tuple(unique(alias_classes)),
                Tuple(unique(buffers)),
            ),
        )
    end
    return Tuple(barriers)
end

function _tempered_smc_nuts_kernel_resource_plan(
    descriptor::AbstractTemperedNUTSSchedulerDescriptor,
)
    schedule = _tempered_smc_nuts_kernel_schedule(descriptor)
    dependencies = _tempered_smc_nuts_kernel_dependencies(descriptor)
    return TemperedSMCNUTSKernelResourcePlan(
        schedule,
        _tempered_smc_nuts_kernel_resource_groups(
            _tempered_smc_nuts_kernel_schedule_lifecycles(schedule),
        ),
        _tempered_smc_nuts_kernel_barriers(schedule, dependencies),
    )
end

function _tempered_smc_nuts_kernel_barriers_after(
    plan::TemperedSMCNUTSKernelResourcePlan,
    stage_index::Int,
)
    placements = TemperedSMCNUTSKernelBarrierPlacement[]
    for barrier in plan.barriers
        barrier.after_stage == stage_index || continue
        push!(placements, barrier)
    end
    return Tuple(placements)
end
