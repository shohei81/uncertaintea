@enum BatchedNUTSKernelStorageClass::UInt8 begin
    NUTSKernelStorageUniform = 0
    NUTSKernelStoragePersistent = 1
    NUTSKernelStorageScratch = 2
end

_batched_nuts_kernel_storage_class(::Val{NUTSKernelAliasControlBlock}) = NUTSKernelStorageUniform
_batched_nuts_kernel_storage_class(::Val{NUTSKernelAliasSchedulerState}) = NUTSKernelStoragePersistent
_batched_nuts_kernel_storage_class(::Val{NUTSKernelAliasControlState}) = NUTSKernelStoragePersistent
_batched_nuts_kernel_storage_class(::Val{NUTSKernelAliasDescriptorScratch}) = NUTSKernelStorageScratch
_batched_nuts_kernel_storage_class(::Val{NUTSKernelAliasTreeState}) = NUTSKernelStorageScratch
_batched_nuts_kernel_storage_class(::Val{NUTSKernelAliasTreeEnergy}) = NUTSKernelStorageScratch
_batched_nuts_kernel_storage_class(::Val{NUTSKernelAliasSubtreeSummary}) = NUTSKernelStorageScratch
_batched_nuts_kernel_storage_class(::Val{NUTSKernelAliasContinuationState}) = NUTSKernelStoragePersistent
_batched_nuts_kernel_storage_class(::Val{NUTSKernelAliasContinuationSummary}) = NUTSKernelStoragePersistent

function _batched_nuts_kernel_storage_class(alias_class::BatchedNUTSKernelAliasClass)
    return _batched_nuts_kernel_storage_class(Val(alias_class))
end

struct BatchedNUTSKernelBufferBinding
    index::Int
    buffer::BatchedNUTSKernelBuffer
    alias_class::BatchedNUTSKernelAliasClass
    storage_class::BatchedNUTSKernelStorageClass
    resource_group::Int
    lifecycle::BatchedNUTSKernelBufferLifecycle
end

struct BatchedNUTSKernelBackendStepBinding{S,R,W,B}
    stage::S
    read_bindings::R
    write_bindings::W
    barriers_after::B
end

struct BatchedNUTSKernelBackendExecutionBlock{P,B,S}
    plan::P
    bindings::B
    steps::S
end

_batched_nuts_backend_plan(block::BatchedNUTSKernelBackendExecutionBlock) = block.plan
_batched_nuts_backend_bindings(block::BatchedNUTSKernelBackendExecutionBlock) = block.bindings
_batched_nuts_backend_steps(block::BatchedNUTSKernelBackendExecutionBlock) = block.steps
_batched_nuts_backend_stage(step::BatchedNUTSKernelBackendStepBinding) = step.stage
_batched_nuts_backend_read_bindings(step::BatchedNUTSKernelBackendStepBinding) = step.read_bindings
_batched_nuts_backend_write_bindings(step::BatchedNUTSKernelBackendStepBinding) = step.write_bindings
_batched_nuts_backend_barriers_after(step::BatchedNUTSKernelBackendStepBinding) = step.barriers_after

function _batched_nuts_backend_buffer_binding(
    block::BatchedNUTSKernelBackendExecutionBlock,
    buffer::BatchedNUTSKernelBuffer,
)
    for binding in block.bindings
        binding.buffer == buffer || continue
        return binding
    end
    throw(KeyError(buffer))
end

function _batched_nuts_backend_step_read_buffers(
    block::BatchedNUTSKernelBackendExecutionBlock,
    step::BatchedNUTSKernelBackendStepBinding,
)
    return Tuple(block.bindings[index].buffer for index in step.read_bindings)
end

function _batched_nuts_backend_step_write_buffers(
    block::BatchedNUTSKernelBackendExecutionBlock,
    step::BatchedNUTSKernelBackendStepBinding,
)
    return Tuple(block.bindings[index].buffer for index in step.write_bindings)
end

function _batched_nuts_backend_step_binding_indices(
    lookup::Dict{BatchedNUTSKernelBuffer,Int},
    buffers,
)
    indices = Int[]
    for buffer in buffers
        push!(indices, lookup[buffer])
    end
    return Tuple(indices)
end

function _batched_nuts_backend_execution_block(
    program::AbstractBatchedNUTSKernelProgram,
)
    plan = _batched_nuts_kernel_resource_plan(program)
    resource_lookup = Dict{BatchedNUTSKernelBuffer,Int}()
    for (resource_index, group) in enumerate(_batched_nuts_kernel_resource_groups(plan))
        for buffer in group.buffers
            resource_lookup[buffer] = resource_index
        end
    end

    bindings_vector = BatchedNUTSKernelBufferBinding[]
    binding_lookup = Dict{BatchedNUTSKernelBuffer,Int}()
    for (binding_index, lifecycle) in enumerate(_batched_nuts_kernel_schedule_lifecycles(plan.schedule))
        binding = BatchedNUTSKernelBufferBinding(
            binding_index,
            lifecycle.buffer,
            lifecycle.alias_class,
            _batched_nuts_kernel_storage_class(lifecycle.alias_class),
            resource_lookup[lifecycle.buffer],
            lifecycle,
        )
        push!(bindings_vector, binding)
        binding_lookup[lifecycle.buffer] = binding_index
    end

    step_bindings = BatchedNUTSKernelBackendStepBinding[]
    for stage in _batched_nuts_kernel_schedule_stages(plan.schedule)
        dataflow = _batched_nuts_kernel_stage_dataflow(stage)
        push!(
            step_bindings,
            BatchedNUTSKernelBackendStepBinding(
                stage,
                _batched_nuts_backend_step_binding_indices(
                    binding_lookup,
                    _batched_nuts_kernel_reads(dataflow),
                ),
                _batched_nuts_backend_step_binding_indices(
                    binding_lookup,
                    _batched_nuts_kernel_writes(dataflow),
                ),
                _batched_nuts_kernel_barriers_after(plan, stage.index),
            ),
        )
    end

    return BatchedNUTSKernelBackendExecutionBlock(
        plan,
        Tuple(bindings_vector),
        Tuple(step_bindings),
    )
end
