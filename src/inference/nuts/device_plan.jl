@enum BatchedNUTSKernelDeviceSegment::UInt8 begin
    NUTSKernelDeviceUniformSegment = 0
    NUTSKernelDevicePersistentSegment = 1
    NUTSKernelDeviceScratchSegment = 2
end

_batched_nuts_device_segment(::Val{NUTSKernelStorageUniform}) = NUTSKernelDeviceUniformSegment
_batched_nuts_device_segment(::Val{NUTSKernelStoragePersistent}) = NUTSKernelDevicePersistentSegment
_batched_nuts_device_segment(::Val{NUTSKernelStorageScratch}) = NUTSKernelDeviceScratchSegment

function _batched_nuts_device_segment(storage_class::BatchedNUTSKernelStorageClass)
    return _batched_nuts_device_segment(Val(storage_class))
end

struct BatchedNUTSKernelDeviceBufferSlot
    binding::BatchedNUTSKernelBufferBinding
    segment::BatchedNUTSKernelDeviceSegment
    segment_slot::Int
end

struct BatchedNUTSKernelDeviceBarrierHint{S,B}
    placement::BatchedNUTSKernelBarrierPlacement
    segments::S
    slot_indices::B
end

struct BatchedNUTSKernelDeviceStage{S,R,W,B}
    step_binding::S
    read_slots::R
    write_slots::W
    barriers_after::B
end

struct BatchedNUTSKernelDevicePlan{B,S}
    target::Symbol
    backend_block::BatchedNUTSKernelBackendExecutionBlock
    slots::B
    stages::S
end

_batched_nuts_device_slots(plan::BatchedNUTSKernelDevicePlan) = plan.slots
_batched_nuts_device_stages(plan::BatchedNUTSKernelDevicePlan) = plan.stages
_batched_nuts_device_step_binding(stage::BatchedNUTSKernelDeviceStage) = stage.step_binding
_batched_nuts_device_read_slots(stage::BatchedNUTSKernelDeviceStage) = stage.read_slots
_batched_nuts_device_write_slots(stage::BatchedNUTSKernelDeviceStage) = stage.write_slots
_batched_nuts_device_barriers_after(stage::BatchedNUTSKernelDeviceStage) = stage.barriers_after

function _batched_nuts_device_slot(
    plan::BatchedNUTSKernelDevicePlan,
    buffer::BatchedNUTSKernelBuffer,
)
    for slot in plan.slots
        slot.binding.buffer == buffer || continue
        return slot
    end
    throw(KeyError(buffer))
end

function _batched_nuts_device_barrier_hint(
    plan::BatchedNUTSKernelDevicePlan,
    placement::BatchedNUTSKernelBarrierPlacement,
)
    slot_indices = Int[]
    segments = BatchedNUTSKernelDeviceSegment[]
    for buffer in placement.buffers
        slot = _batched_nuts_device_slot(plan, buffer)
        push!(slot_indices, slot.binding.index)
        push!(segments, slot.segment)
    end
    return BatchedNUTSKernelDeviceBarrierHint(
        placement,
        Tuple(unique(segments)),
        Tuple(slot_indices),
    )
end

function _batched_nuts_device_stage(
    plan::BatchedNUTSKernelDevicePlan,
    step_binding::BatchedNUTSKernelBackendStepBinding,
)
    return BatchedNUTSKernelDeviceStage(
        step_binding,
        step_binding.read_bindings,
        step_binding.write_bindings,
        Tuple(
            _batched_nuts_device_barrier_hint(plan, placement) for
            placement in step_binding.barriers_after
        ),
    )
end

function _batched_nuts_device_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    backend_block = _batched_nuts_backend_execution_block(program)
    segment_counts = Dict{BatchedNUTSKernelDeviceSegment,Int}()
    slot_vector = BatchedNUTSKernelDeviceBufferSlot[]
    for binding in _batched_nuts_backend_bindings(backend_block)
        segment = _batched_nuts_device_segment(binding.storage_class)
        segment_slot = get(segment_counts, segment, 0) + 1
        segment_counts[segment] = segment_slot
        push!(
            slot_vector,
            BatchedNUTSKernelDeviceBufferSlot(binding, segment, segment_slot),
        )
    end

    plan = BatchedNUTSKernelDevicePlan(
        target,
        backend_block,
        Tuple(slot_vector),
        (),
    )
    stages = Tuple(
        _batched_nuts_device_stage(plan, step_binding) for
        step_binding in _batched_nuts_backend_steps(backend_block)
    )
    return BatchedNUTSKernelDevicePlan(
        target,
        backend_block,
        plan.slots,
        stages,
    )
end
