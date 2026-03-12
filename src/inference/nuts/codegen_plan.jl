@enum BatchedNUTSKernelCodegenBackend::UInt8 begin
    NUTSKernelCPUCodegen = 0
    NUTSKernelMetalCodegen = 1
    NUTSKernelCUDACodegen = 2
end

struct BatchedNUTSKernelCodegenArgument
    binding::BatchedNUTSKernelExecutorBinding
    symbol::Symbol
end

struct BatchedNUTSKernelCodegenStage{S,C,D,H,B}
    executor_stage::S
    backend::BatchedNUTSKernelCodegenBackend
    module_symbol::Symbol
    entry_symbol::Symbol
    constant_arguments::C
    device_arguments::D
    shared_arguments::H
    barriers_after::B
end

struct BatchedNUTSKernelCodegenPlan{P,S}
    target::Symbol
    executor_plan::P
    module_symbol::Symbol
    stages::S
end

_batched_nuts_codegen_executor_plan(plan::BatchedNUTSKernelCodegenPlan) = plan.executor_plan
_batched_nuts_codegen_module_symbol(plan::BatchedNUTSKernelCodegenPlan) = plan.module_symbol
_batched_nuts_codegen_stages(plan::BatchedNUTSKernelCodegenPlan) = plan.stages

_batched_nuts_codegen_executor_stage(stage::BatchedNUTSKernelCodegenStage) = stage.executor_stage
_batched_nuts_codegen_backend(stage::BatchedNUTSKernelCodegenStage) = stage.backend
_batched_nuts_codegen_module_symbol(stage::BatchedNUTSKernelCodegenStage) = stage.module_symbol
_batched_nuts_codegen_entry_symbol(stage::BatchedNUTSKernelCodegenStage) = stage.entry_symbol
_batched_nuts_codegen_constant_arguments(stage::BatchedNUTSKernelCodegenStage) =
    stage.constant_arguments
_batched_nuts_codegen_device_arguments(stage::BatchedNUTSKernelCodegenStage) =
    stage.device_arguments
_batched_nuts_codegen_shared_arguments(stage::BatchedNUTSKernelCodegenStage) =
    stage.shared_arguments
_batched_nuts_codegen_barriers_after(stage::BatchedNUTSKernelCodegenStage) =
    stage.barriers_after

_batched_nuts_codegen_backend(::Val{:gpu}) = NUTSKernelCPUCodegen
_batched_nuts_codegen_backend(::Val{:metal}) = NUTSKernelMetalCodegen
_batched_nuts_codegen_backend(::Val{:cuda}) = NUTSKernelCUDACodegen

function _batched_nuts_codegen_backend(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS codegen target $(target)"))
    return _batched_nuts_codegen_backend(Val(target))
end

_batched_nuts_codegen_module_symbol(::Val{:gpu}) = :UncertainTeaCPUBackend
_batched_nuts_codegen_module_symbol(::Val{:metal}) = :UncertainTeaMetalBackend
_batched_nuts_codegen_module_symbol(::Val{:cuda}) = :UncertainTeaCUDABackend

function _batched_nuts_codegen_module_symbol(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS codegen target $(target)"))
    return _batched_nuts_codegen_module_symbol(Val(target))
end

function _batched_nuts_codegen_argument_symbol(
    prefix::Symbol,
    binding::BatchedNUTSKernelExecutorBinding,
)
    return Symbol(prefix, :_, _batched_nuts_executor_binding_index(binding))
end

function _batched_nuts_codegen_arguments(
    prefix::Symbol,
    bindings::Tuple,
)
    return Tuple(
        BatchedNUTSKernelCodegenArgument(
            binding,
            _batched_nuts_codegen_argument_symbol(prefix, binding),
        ) for binding in bindings
    )
end

function _batched_nuts_codegen_entry_symbol(
    module_symbol::Symbol,
    executor_stage::BatchedNUTSKernelExecutorStage,
    stage_index::Int,
)
    return Symbol(
        lowercase(String(module_symbol)),
        :__,
        _batched_nuts_executor_kernel_symbol(executor_stage),
        :__,
        :stage_,
        stage_index,
    )
end

function _batched_nuts_codegen_stage(
    plan::BatchedNUTSKernelCodegenPlan,
    executor_stage::BatchedNUTSKernelExecutorStage,
    stage_index::Int,
)
    return BatchedNUTSKernelCodegenStage(
        executor_stage,
        _batched_nuts_codegen_backend(plan.target),
        plan.module_symbol,
        _batched_nuts_codegen_entry_symbol(
            plan.module_symbol,
            executor_stage,
            stage_index,
        ),
        _batched_nuts_codegen_arguments(
            :const_arg,
            _batched_nuts_executor_constant_arguments(executor_stage),
        ),
        _batched_nuts_codegen_arguments(
            :device_arg,
            _batched_nuts_executor_device_arguments(executor_stage),
        ),
        _batched_nuts_codegen_arguments(
            :shared_arg,
            _batched_nuts_executor_shared_arguments(executor_stage),
        ),
        _batched_nuts_executor_barriers_after(executor_stage),
    )
end

function _batched_nuts_codegen_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    executor_plan = _batched_nuts_executor_plan(program; target=target)
    module_symbol = _batched_nuts_codegen_module_symbol(target)
    plan = BatchedNUTSKernelCodegenPlan(
        target,
        executor_plan,
        module_symbol,
        (),
    )
    stages = Tuple(
        _batched_nuts_codegen_stage(plan, stage, stage_index) for
        (stage_index, stage) in enumerate(_batched_nuts_executor_stages(executor_plan))
    )
    return BatchedNUTSKernelCodegenPlan(
        target,
        executor_plan,
        module_symbol,
        stages,
    )
end
