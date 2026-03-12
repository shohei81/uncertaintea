@enum BatchedNUTSKernelTargetAllocation::UInt8 begin
    NUTSKernelTargetAllocateConstant = 0
    NUTSKernelTargetAllocateDevice = 1
    NUTSKernelTargetAllocateShared = 2
end

@enum BatchedNUTSKernelTargetBarrierStrategy::UInt8 begin
    NUTSKernelTargetSequentialBarrier = 0
    NUTSKernelTargetMetalThreadgroupBarrier = 1
    NUTSKernelTargetCUDAThreadBlockBarrier = 2
end

struct BatchedNUTSKernelTargetBufferBinding
    slot::BatchedNUTSKernelDeviceBufferSlot
    allocation::BatchedNUTSKernelTargetAllocation
end

struct BatchedNUTSKernelTargetBarrierHint
    barrier::BatchedNUTSKernelDeviceBarrierHint
    strategy::BatchedNUTSKernelTargetBarrierStrategy
end

struct BatchedNUTSKernelTargetStage{S,R,W,B}
    device_stage::S
    read_bindings::R
    write_bindings::W
    barriers_after::B
end

struct BatchedNUTSKernelTargetPlan{B,S}
    target::Symbol
    device_plan::B
    bindings::Tuple
    stages::S
end

_batched_nuts_target_bindings(plan::BatchedNUTSKernelTargetPlan) = plan.bindings
_batched_nuts_target_stages(plan::BatchedNUTSKernelTargetPlan) = plan.stages
_batched_nuts_target_device_plan(plan::BatchedNUTSKernelTargetPlan) = plan.device_plan
_batched_nuts_target_device_stage(stage::BatchedNUTSKernelTargetStage) = stage.device_stage
_batched_nuts_target_read_bindings(stage::BatchedNUTSKernelTargetStage) = stage.read_bindings
_batched_nuts_target_write_bindings(stage::BatchedNUTSKernelTargetStage) = stage.write_bindings
_batched_nuts_target_barriers_after(stage::BatchedNUTSKernelTargetStage) = stage.barriers_after

function _batched_nuts_target_binding(
    plan::BatchedNUTSKernelTargetPlan,
    buffer::BatchedNUTSKernelBuffer,
)
    for binding in plan.bindings
        binding.slot.binding.buffer == buffer || continue
        return binding
    end
    throw(KeyError(buffer))
end

function _batched_nuts_target_allocation(
    ::Val{:metal},
    slot::BatchedNUTSKernelDeviceBufferSlot,
)
    if slot.segment == NUTSKernelDeviceUniformSegment
        return NUTSKernelTargetAllocateConstant
    elseif slot.binding.alias_class in (
        NUTSKernelAliasDescriptorScratch,
        NUTSKernelAliasTreeEnergy,
        NUTSKernelAliasSubtreeSummary,
    )
        return NUTSKernelTargetAllocateShared
    end
    return NUTSKernelTargetAllocateDevice
end

function _batched_nuts_target_allocation(
    ::Val{:cuda},
    slot::BatchedNUTSKernelDeviceBufferSlot,
)
    if slot.segment == NUTSKernelDeviceUniformSegment
        return NUTSKernelTargetAllocateConstant
    elseif slot.binding.alias_class in (
        NUTSKernelAliasDescriptorScratch,
        NUTSKernelAliasTreeEnergy,
        NUTSKernelAliasSubtreeSummary,
    )
        return NUTSKernelTargetAllocateShared
    end
    return NUTSKernelTargetAllocateDevice
end

function _batched_nuts_target_allocation(
    ::Val{:gpu},
    slot::BatchedNUTSKernelDeviceBufferSlot,
)
    if slot.segment == NUTSKernelDeviceUniformSegment
        return NUTSKernelTargetAllocateConstant
    end
    return NUTSKernelTargetAllocateDevice
end

function _batched_nuts_target_allocation(
    target::Symbol,
    slot::BatchedNUTSKernelDeviceBufferSlot,
)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS device target $(target)"))
    return _batched_nuts_target_allocation(Val(target), slot)
end

function _batched_nuts_target_barrier_strategy(
    ::Val{:metal},
    hint::BatchedNUTSKernelDeviceBarrierHint,
)
    return NUTSKernelDeviceScratchSegment in hint.segments ?
           NUTSKernelTargetMetalThreadgroupBarrier :
           NUTSKernelTargetSequentialBarrier
end

function _batched_nuts_target_barrier_strategy(
    ::Val{:cuda},
    hint::BatchedNUTSKernelDeviceBarrierHint,
)
    return NUTSKernelDeviceScratchSegment in hint.segments ?
           NUTSKernelTargetCUDAThreadBlockBarrier :
           NUTSKernelTargetSequentialBarrier
end

function _batched_nuts_target_barrier_strategy(
    ::Val{:gpu},
    hint::BatchedNUTSKernelDeviceBarrierHint,
)
    return NUTSKernelTargetSequentialBarrier
end

function _batched_nuts_target_barrier_strategy(
    target::Symbol,
    hint::BatchedNUTSKernelDeviceBarrierHint,
)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS device target $(target)"))
    return _batched_nuts_target_barrier_strategy(Val(target), hint)
end

function _batched_nuts_target_stage(
    plan::BatchedNUTSKernelTargetPlan,
    device_stage::BatchedNUTSKernelDeviceStage,
)
    return BatchedNUTSKernelTargetStage(
        device_stage,
        device_stage.read_slots,
        device_stage.write_slots,
        Tuple(
            BatchedNUTSKernelTargetBarrierHint(
                hint,
                _batched_nuts_target_barrier_strategy(plan.target, hint),
            ) for hint in device_stage.barriers_after
        ),
    )
end

function _batched_nuts_target_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    device_plan = _batched_nuts_device_plan(program; target=target)
    bindings = Tuple(
        BatchedNUTSKernelTargetBufferBinding(
            slot,
            _batched_nuts_target_allocation(target, slot),
        ) for slot in _batched_nuts_device_slots(device_plan)
    )
    plan = BatchedNUTSKernelTargetPlan(
        target,
        device_plan,
        bindings,
        (),
    )
    stages = Tuple(
        _batched_nuts_target_stage(plan, device_stage) for
        device_stage in _batched_nuts_device_stages(device_plan)
    )
    return BatchedNUTSKernelTargetPlan(
        target,
        device_plan,
        bindings,
        stages,
    )
end
