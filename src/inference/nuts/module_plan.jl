@enum BatchedNUTSKernelModuleBackend::UInt8 begin
    NUTSKernelCPUModule = 0
    NUTSKernelMetalModule = 1
    NUTSKernelCUDAModule = 2
end

struct BatchedNUTSKernelModuleStage{S}
    source_stage::S
    backend::BatchedNUTSKernelModuleBackend
    module_symbol::Symbol
    entry_symbol::Symbol
    filename::String
    source_blob::String
end

struct BatchedNUTSKernelModulePlan{P,S}
    target::Symbol
    source_plan::P
    backend::BatchedNUTSKernelModuleBackend
    module_symbol::Symbol
    stages::S
end

_batched_nuts_module_source_plan(plan::BatchedNUTSKernelModulePlan) = plan.source_plan
_batched_nuts_module_backend(plan::BatchedNUTSKernelModulePlan) = plan.backend
_batched_nuts_module_symbol(plan::BatchedNUTSKernelModulePlan) = plan.module_symbol
_batched_nuts_module_stages(plan::BatchedNUTSKernelModulePlan) = plan.stages

_batched_nuts_module_source_stage(stage::BatchedNUTSKernelModuleStage) = stage.source_stage
_batched_nuts_module_backend(stage::BatchedNUTSKernelModuleStage) = stage.backend
_batched_nuts_module_symbol(stage::BatchedNUTSKernelModuleStage) = stage.module_symbol
_batched_nuts_module_entry_symbol(stage::BatchedNUTSKernelModuleStage) = stage.entry_symbol
_batched_nuts_module_filename(stage::BatchedNUTSKernelModuleStage) = stage.filename
_batched_nuts_module_source_blob(stage::BatchedNUTSKernelModuleStage) = stage.source_blob

_batched_nuts_module_backend(::Val{:gpu}) = NUTSKernelCPUModule
_batched_nuts_module_backend(::Val{:metal}) = NUTSKernelMetalModule
_batched_nuts_module_backend(::Val{:cuda}) = NUTSKernelCUDAModule

function _batched_nuts_module_backend(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS module target $(target)"))
    return _batched_nuts_module_backend(Val(target))
end

_batched_nuts_module_symbol(::Val{:gpu}) = :UncertainTeaCPUModules
_batched_nuts_module_symbol(::Val{:metal}) = :UncertainTeaMetalModules
_batched_nuts_module_symbol(::Val{:cuda}) = :UncertainTeaCUDAModules

function _batched_nuts_module_symbol(target::Symbol)
    return _batched_nuts_module_symbol(Val(_gpu_backend_require_target(target)))
end

_batched_nuts_module_extension(target::Symbol) = gpu_backend_module_extension(target)

function _batched_nuts_module_filename(
    module_symbol::Symbol,
    entry_symbol::Symbol,
    target::Symbol,
)
    return gpu_backend_module_filename(module_symbol, entry_symbol, target)
end

function _batched_nuts_module_source_blob(source_stage::BatchedNUTSKernelSourceStage)
    return join(_batched_nuts_source_lines(source_stage), "\n")
end

function _batched_nuts_module_stage(
    plan::BatchedNUTSKernelModulePlan,
    source_stage::BatchedNUTSKernelSourceStage,
)
    entry_symbol = _batched_nuts_source_entry(source_stage)
    return BatchedNUTSKernelModuleStage(
        source_stage,
        plan.backend,
        plan.module_symbol,
        entry_symbol,
        _batched_nuts_module_filename(plan.module_symbol, entry_symbol, plan.target),
        _batched_nuts_module_source_blob(source_stage),
    )
end

function _batched_nuts_module_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    source_plan = _batched_nuts_source_plan(program; target=target)
    backend = _batched_nuts_module_backend(target)
    module_symbol = _batched_nuts_module_symbol(target)
    plan = BatchedNUTSKernelModulePlan(
        target,
        source_plan,
        backend,
        module_symbol,
        (),
    )
    stages = Tuple(
        _batched_nuts_module_stage(plan, stage) for
        stage in _batched_nuts_source_stages(source_plan)
    )
    return BatchedNUTSKernelModulePlan(
        target,
        source_plan,
        backend,
        module_symbol,
        stages,
    )
end
