@enum BatchedNUTSKernelExecutorKind::UInt8 begin
    NUTSKernelSequentialExecutor = 0
    NUTSKernelMetalExecutor = 1
    NUTSKernelCUDAExecutor = 2
end

@enum BatchedNUTSKernelArgumentClass::UInt8 begin
    NUTSKernelConstantArgument = 0
    NUTSKernelDeviceArgument = 1
    NUTSKernelMetalThreadgroupArgument = 2
    NUTSKernelCUDASharedArgument = 3
end

struct BatchedNUTSKernelExecutorBinding
    binding::BatchedNUTSKernelLaunchBinding
    argument_class::BatchedNUTSKernelArgumentClass
end

struct BatchedNUTSKernelExecutorStage{S,C,D,H,B}
    launch_stage::S
    executor_kind::BatchedNUTSKernelExecutorKind
    kernel_symbol::Symbol
    constant_arguments::C
    device_arguments::D
    shared_arguments::H
    barriers_after::B
end

struct BatchedNUTSKernelExecutorPlan{P,C,D,S}
    target::Symbol
    launch_plan::P
    constant_arguments::C
    device_arguments::D
    stages::S
end

_batched_nuts_executor_launch_plan(plan::BatchedNUTSKernelExecutorPlan) = plan.launch_plan
_batched_nuts_executor_constant_arguments(plan::BatchedNUTSKernelExecutorPlan) =
    plan.constant_arguments
_batched_nuts_executor_device_arguments(plan::BatchedNUTSKernelExecutorPlan) =
    plan.device_arguments
_batched_nuts_executor_stages(plan::BatchedNUTSKernelExecutorPlan) = plan.stages

_batched_nuts_executor_launch_stage(stage::BatchedNUTSKernelExecutorStage) = stage.launch_stage
_batched_nuts_executor_kind(stage::BatchedNUTSKernelExecutorStage) = stage.executor_kind
_batched_nuts_executor_kernel_symbol(stage::BatchedNUTSKernelExecutorStage) = stage.kernel_symbol
_batched_nuts_executor_constant_arguments(stage::BatchedNUTSKernelExecutorStage) =
    stage.constant_arguments
_batched_nuts_executor_device_arguments(stage::BatchedNUTSKernelExecutorStage) =
    stage.device_arguments
_batched_nuts_executor_shared_arguments(stage::BatchedNUTSKernelExecutorStage) =
    stage.shared_arguments
_batched_nuts_executor_barriers_after(stage::BatchedNUTSKernelExecutorStage) =
    stage.barriers_after

function _batched_nuts_executor_binding_index(
    binding::BatchedNUTSKernelExecutorBinding,
)
    return _batched_nuts_launch_binding_index(binding.binding)
end

function _batched_nuts_executor_binding(
    plan::BatchedNUTSKernelExecutorPlan,
    buffer::BatchedNUTSKernelBuffer,
)
    for binding in plan.constant_arguments
        binding.binding.binding.slot.binding.buffer == buffer && return binding
    end
    for binding in plan.device_arguments
        binding.binding.binding.slot.binding.buffer == buffer && return binding
    end
    throw(KeyError(buffer))
end

function _batched_nuts_executor_stage_binding(
    stage::BatchedNUTSKernelExecutorStage,
    buffer::BatchedNUTSKernelBuffer,
)
    for binding in stage.constant_arguments
        binding.binding.binding.slot.binding.buffer == buffer && return binding
    end
    for binding in stage.device_arguments
        binding.binding.binding.slot.binding.buffer == buffer && return binding
    end
    for binding in stage.shared_arguments
        binding.binding.binding.slot.binding.buffer == buffer && return binding
    end
    throw(KeyError(buffer))
end

_batched_nuts_executor_kind(::Val{:gpu}) = NUTSKernelSequentialExecutor
_batched_nuts_executor_kind(::Val{:metal}) = NUTSKernelMetalExecutor
_batched_nuts_executor_kind(::Val{:cuda}) = NUTSKernelCUDAExecutor

function _batched_nuts_executor_kind(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS executor target $(target)"))
    return _batched_nuts_executor_kind(Val(target))
end

function _batched_nuts_argument_class(
    ::Val{:metal},
    binding::BatchedNUTSKernelLaunchBinding,
)
    allocation = binding.binding.allocation
    allocation == NUTSKernelTargetAllocateConstant && return NUTSKernelConstantArgument
    allocation == NUTSKernelTargetAllocateDevice && return NUTSKernelDeviceArgument
    return NUTSKernelMetalThreadgroupArgument
end

function _batched_nuts_argument_class(
    ::Val{:cuda},
    binding::BatchedNUTSKernelLaunchBinding,
)
    allocation = binding.binding.allocation
    allocation == NUTSKernelTargetAllocateConstant && return NUTSKernelConstantArgument
    allocation == NUTSKernelTargetAllocateDevice && return NUTSKernelDeviceArgument
    return NUTSKernelCUDASharedArgument
end

function _batched_nuts_argument_class(
    ::Val{:gpu},
    binding::BatchedNUTSKernelLaunchBinding,
)
    allocation = binding.binding.allocation
    allocation == NUTSKernelTargetAllocateConstant && return NUTSKernelConstantArgument
    return NUTSKernelDeviceArgument
end

function _batched_nuts_argument_class(
    target::Symbol,
    binding::BatchedNUTSKernelLaunchBinding,
)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS executor target $(target)"))
    return _batched_nuts_argument_class(Val(target), binding)
end

function _batched_nuts_executor_bindings(
    target::Symbol,
    bindings::Tuple,
)
    return Tuple(
        BatchedNUTSKernelExecutorBinding(
            binding,
            _batched_nuts_argument_class(target, binding),
        ) for binding in bindings
    )
end

function _batched_nuts_executor_step_symbol(step)
    step isa BatchedNUTSReloadControlStep && return :nuts_reload_control
    step isa BatchedNUTSLeapfrogStep && return :nuts_leapfrog
    step isa BatchedNUTSHamiltonianStep && return :nuts_hamiltonian
    step isa BatchedNUTSAdvanceStep && return :nuts_advance
    step isa BatchedNUTSActivateMergeStep && return :nuts_activate_merge
    step isa BatchedNUTSMergeStep && return :nuts_merge
    step isa BatchedNUTSTransitionPhaseStep && return :nuts_transition_phase
    throw(ArgumentError("unsupported NUTS kernel step $(typeof(step))"))
end

function _batched_nuts_executor_stage(
    plan::BatchedNUTSKernelExecutorPlan,
    launch_stage::BatchedNUTSKernelLaunchStageExecutor,
)
    dataflow = _batched_nuts_launch_stage_dataflow(launch_stage)
    return BatchedNUTSKernelExecutorStage(
        launch_stage,
        _batched_nuts_executor_kind(plan.target),
        _batched_nuts_executor_step_symbol(_batched_nuts_kernel_step(dataflow)),
        _batched_nuts_executor_bindings(
            plan.target,
            _batched_nuts_launch_constant_bindings(launch_stage),
        ),
        _batched_nuts_executor_bindings(
            plan.target,
            _batched_nuts_launch_device_bindings(launch_stage),
        ),
        _batched_nuts_executor_bindings(
            plan.target,
            _batched_nuts_launch_shared_bindings(launch_stage),
        ),
        _batched_nuts_launch_barriers_after(launch_stage),
    )
end

function _batched_nuts_executor_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    launch_plan = _batched_nuts_launch_plan(program; target=target)
    constant_arguments = _batched_nuts_executor_bindings(
        target,
        _batched_nuts_launch_constant_bindings(launch_plan),
    )
    device_arguments = _batched_nuts_executor_bindings(
        target,
        _batched_nuts_launch_device_bindings(launch_plan),
    )
    plan = BatchedNUTSKernelExecutorPlan(
        target,
        launch_plan,
        constant_arguments,
        device_arguments,
        (),
    )
    stages = Tuple(
        _batched_nuts_executor_stage(plan, stage) for
        stage in _batched_nuts_launch_executors(launch_plan)
    )
    return BatchedNUTSKernelExecutorPlan(
        target,
        launch_plan,
        constant_arguments,
        device_arguments,
        stages,
    )
end
