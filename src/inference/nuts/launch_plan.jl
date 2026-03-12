@enum BatchedNUTSKernelLaunchKind::UInt8 begin
    NUTSKernelLaunchSequential = 0
    NUTSKernelLaunchMetalCompute = 1
    NUTSKernelLaunchCUDAKernel = 2
end

struct BatchedNUTSKernelLaunchBinding
    binding::BatchedNUTSKernelTargetBufferBinding
    slot::Int
end

struct BatchedNUTSKernelLaunchStageExecutor{S,C,D,H,R,W,B}
    target_stage::S
    launch_kind::BatchedNUTSKernelLaunchKind
    constant_bindings::C
    device_bindings::D
    shared_bindings::H
    read_bindings::R
    write_bindings::W
    barriers_after::B
end

struct BatchedNUTSKernelLaunchPlan{P,C,D,E}
    target::Symbol
    target_plan::P
    constant_bindings::C
    device_bindings::D
    executors::E
end

_batched_nuts_launch_target_plan(plan::BatchedNUTSKernelLaunchPlan) = plan.target_plan
_batched_nuts_launch_constant_bindings(plan::BatchedNUTSKernelLaunchPlan) = plan.constant_bindings
_batched_nuts_launch_device_bindings(plan::BatchedNUTSKernelLaunchPlan) = plan.device_bindings
_batched_nuts_launch_executors(plan::BatchedNUTSKernelLaunchPlan) = plan.executors

_batched_nuts_launch_target_stage(executor::BatchedNUTSKernelLaunchStageExecutor) = executor.target_stage
_batched_nuts_launch_kind(executor::BatchedNUTSKernelLaunchStageExecutor) = executor.launch_kind
_batched_nuts_launch_constant_bindings(executor::BatchedNUTSKernelLaunchStageExecutor) =
    executor.constant_bindings
_batched_nuts_launch_device_bindings(executor::BatchedNUTSKernelLaunchStageExecutor) =
    executor.device_bindings
_batched_nuts_launch_shared_bindings(executor::BatchedNUTSKernelLaunchStageExecutor) =
    executor.shared_bindings
_batched_nuts_launch_read_bindings(executor::BatchedNUTSKernelLaunchStageExecutor) =
    executor.read_bindings
_batched_nuts_launch_write_bindings(executor::BatchedNUTSKernelLaunchStageExecutor) =
    executor.write_bindings
_batched_nuts_launch_barriers_after(executor::BatchedNUTSKernelLaunchStageExecutor) =
    executor.barriers_after

function _batched_nuts_launch_binding_index(binding::BatchedNUTSKernelLaunchBinding)
    return binding.binding.slot.binding.index
end

function _batched_nuts_launch_binding(
    plan::BatchedNUTSKernelLaunchPlan,
    buffer::BatchedNUTSKernelBuffer,
)
    for binding in plan.constant_bindings
        binding.binding.slot.binding.buffer == buffer && return binding
    end
    for binding in plan.device_bindings
        binding.binding.slot.binding.buffer == buffer && return binding
    end
    throw(KeyError(buffer))
end

function _batched_nuts_launch_stage_binding(
    executor::BatchedNUTSKernelLaunchStageExecutor,
    buffer::BatchedNUTSKernelBuffer,
)
    for binding in executor.constant_bindings
        binding.binding.slot.binding.buffer == buffer && return binding
    end
    for binding in executor.device_bindings
        binding.binding.slot.binding.buffer == buffer && return binding
    end
    for binding in executor.shared_bindings
        binding.binding.slot.binding.buffer == buffer && return binding
    end
    throw(KeyError(buffer))
end

_batched_nuts_launch_kind(::Val{:gpu}) = NUTSKernelLaunchSequential
_batched_nuts_launch_kind(::Val{:metal}) = NUTSKernelLaunchMetalCompute
_batched_nuts_launch_kind(::Val{:cuda}) = NUTSKernelLaunchCUDAKernel

function _batched_nuts_launch_kind(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS launch target $(target)"))
    return _batched_nuts_launch_kind(Val(target))
end

function _batched_nuts_launch_global_bindings(
    target_plan::BatchedNUTSKernelTargetPlan,
    allocation::BatchedNUTSKernelTargetAllocation,
)
    bindings = BatchedNUTSKernelLaunchBinding[]
    slot = 0
    for target_binding in _batched_nuts_target_bindings(target_plan)
        target_binding.allocation == allocation || continue
        slot += 1
        push!(bindings, BatchedNUTSKernelLaunchBinding(target_binding, slot))
    end
    return Tuple(bindings)
end

function _batched_nuts_launch_stage_binding_indices(
    stage::BatchedNUTSKernelTargetStage,
)
    indices = Int[]
    seen = Set{Int}()
    for index in _batched_nuts_target_read_bindings(stage)
        index in seen && continue
        push!(indices, index)
        push!(seen, index)
    end
    for index in _batched_nuts_target_write_bindings(stage)
        index in seen && continue
        push!(indices, index)
        push!(seen, index)
    end
    return Tuple(indices)
end

function _batched_nuts_launch_stage_global_bindings(
    global_bindings::Tuple,
    stage::BatchedNUTSKernelTargetStage,
)
    lookup = Dict{Int,BatchedNUTSKernelLaunchBinding}()
    for binding in global_bindings
        lookup[_batched_nuts_launch_binding_index(binding)] = binding
    end
    selected = BatchedNUTSKernelLaunchBinding[]
    for index in _batched_nuts_launch_stage_binding_indices(stage)
        binding = get(lookup, index, nothing)
        binding === nothing || push!(selected, binding)
    end
    return Tuple(selected)
end

function _batched_nuts_launch_stage_shared_bindings(
    target_plan::BatchedNUTSKernelTargetPlan,
    stage::BatchedNUTSKernelTargetStage,
)
    selected = BatchedNUTSKernelLaunchBinding[]
    slot = 0
    for index in _batched_nuts_launch_stage_binding_indices(stage)
        target_binding = _batched_nuts_target_bindings(target_plan)[index]
        target_binding.allocation == NUTSKernelTargetAllocateShared || continue
        slot += 1
        push!(selected, BatchedNUTSKernelLaunchBinding(target_binding, slot))
    end
    return Tuple(selected)
end

function _batched_nuts_launch_stage_lookup(
    constant_bindings::Tuple,
    device_bindings::Tuple,
    shared_bindings::Tuple,
)
    lookup = Dict{Int,BatchedNUTSKernelLaunchBinding}()
    for binding in constant_bindings
        lookup[_batched_nuts_launch_binding_index(binding)] = binding
    end
    for binding in device_bindings
        lookup[_batched_nuts_launch_binding_index(binding)] = binding
    end
    for binding in shared_bindings
        lookup[_batched_nuts_launch_binding_index(binding)] = binding
    end
    return lookup
end

function _batched_nuts_launch_stage_executor(
    plan::BatchedNUTSKernelLaunchPlan,
    target_stage::BatchedNUTSKernelTargetStage,
)
    constant_bindings = _batched_nuts_launch_stage_global_bindings(
        _batched_nuts_launch_constant_bindings(plan),
        target_stage,
    )
    device_bindings = _batched_nuts_launch_stage_global_bindings(
        _batched_nuts_launch_device_bindings(plan),
        target_stage,
    )
    shared_bindings = _batched_nuts_launch_stage_shared_bindings(
        _batched_nuts_launch_target_plan(plan),
        target_stage,
    )
    lookup = _batched_nuts_launch_stage_lookup(
        constant_bindings,
        device_bindings,
        shared_bindings,
    )
    return BatchedNUTSKernelLaunchStageExecutor(
        target_stage,
        _batched_nuts_launch_kind(plan.target),
        constant_bindings,
        device_bindings,
        shared_bindings,
        Tuple(lookup[index] for index in _batched_nuts_target_read_bindings(target_stage)),
        Tuple(lookup[index] for index in _batched_nuts_target_write_bindings(target_stage)),
        _batched_nuts_target_barriers_after(target_stage),
    )
end

function _batched_nuts_launch_stage_dataflow(
    executor::BatchedNUTSKernelLaunchStageExecutor,
)
    return _batched_nuts_kernel_stage_dataflow(
        _batched_nuts_backend_stage(
            _batched_nuts_device_step_binding(
                _batched_nuts_target_device_stage(
                    _batched_nuts_launch_target_stage(executor),
                ),
            ),
        ),
    )
end

function _batched_nuts_launch_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    target_plan = _batched_nuts_target_plan(program; target=target)
    constant_bindings = _batched_nuts_launch_global_bindings(
        target_plan,
        NUTSKernelTargetAllocateConstant,
    )
    device_bindings = _batched_nuts_launch_global_bindings(
        target_plan,
        NUTSKernelTargetAllocateDevice,
    )
    plan = BatchedNUTSKernelLaunchPlan(
        target,
        target_plan,
        constant_bindings,
        device_bindings,
        (),
    )
    executors = Tuple(
        _batched_nuts_launch_stage_executor(plan, stage) for
        stage in _batched_nuts_target_stages(target_plan)
    )
    return BatchedNUTSKernelLaunchPlan(
        target,
        target_plan,
        constant_bindings,
        device_bindings,
        executors,
    )
end
