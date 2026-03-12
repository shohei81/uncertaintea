@enum BatchedNUTSKernelBundleBackend::UInt8 begin
    NUTSKernelCPUBundle = 0
    NUTSKernelMetalBundle = 1
    NUTSKernelCUDABundle = 2
end

struct BatchedNUTSKernelBundleStage{S}
    module_stage::S
    bundle_symbol::Symbol
    relative_path::String
end

struct BatchedNUTSKernelBundlePlan{P,S}
    target::Symbol
    module_plan::P
    backend::BatchedNUTSKernelBundleBackend
    bundle_symbol::Symbol
    manifest_filename::String
    manifest_blob::String
    stages::S
end

_batched_nuts_bundle_module_plan(plan::BatchedNUTSKernelBundlePlan) = plan.module_plan
_batched_nuts_bundle_backend(plan::BatchedNUTSKernelBundlePlan) = plan.backend
_batched_nuts_bundle_symbol(plan::BatchedNUTSKernelBundlePlan) = plan.bundle_symbol
_batched_nuts_bundle_manifest_filename(plan::BatchedNUTSKernelBundlePlan) = plan.manifest_filename
_batched_nuts_bundle_manifest_blob(plan::BatchedNUTSKernelBundlePlan) = plan.manifest_blob
_batched_nuts_bundle_stages(plan::BatchedNUTSKernelBundlePlan) = plan.stages

_batched_nuts_bundle_module_stage(stage::BatchedNUTSKernelBundleStage) = stage.module_stage
_batched_nuts_bundle_symbol(stage::BatchedNUTSKernelBundleStage) = stage.bundle_symbol
_batched_nuts_bundle_relative_path(stage::BatchedNUTSKernelBundleStage) = stage.relative_path

_batched_nuts_bundle_backend(::Val{:gpu}) = NUTSKernelCPUBundle
_batched_nuts_bundle_backend(::Val{:metal}) = NUTSKernelMetalBundle
_batched_nuts_bundle_backend(::Val{:cuda}) = NUTSKernelCUDABundle

function _batched_nuts_bundle_backend(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS bundle target $(target)"))
    return _batched_nuts_bundle_backend(Val(target))
end

_batched_nuts_bundle_symbol(::Val{:gpu}) = :UncertainTeaCPUBundle
_batched_nuts_bundle_symbol(::Val{:metal}) = :UncertainTeaMetalBundle
_batched_nuts_bundle_symbol(::Val{:cuda}) = :UncertainTeaCUDABundle

function _batched_nuts_bundle_symbol(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS bundle target $(target)"))
    return _batched_nuts_bundle_symbol(Val(target))
end

function _batched_nuts_bundle_manifest_filename(bundle_symbol::Symbol)
    return string(lowercase(String(bundle_symbol)), "__manifest.toml")
end

function _batched_nuts_bundle_stage(
    plan::BatchedNUTSKernelBundlePlan,
    module_stage::BatchedNUTSKernelModuleStage,
)
    return BatchedNUTSKernelBundleStage(
        module_stage,
        plan.bundle_symbol,
        _batched_nuts_module_filename(module_stage),
    )
end

function _batched_nuts_bundle_manifest_blob(
    bundle_symbol::Symbol,
    stages::Tuple,
)
    lines = String[
        string("bundle = \"", bundle_symbol, "\""),
        string("count = ", length(stages)),
    ]
    for (index, stage) in enumerate(stages)
        push!(lines, string("[[stage]]"))
        push!(lines, string("index = ", index))
        push!(lines, string("entry = \"", _batched_nuts_module_entry_symbol(_batched_nuts_bundle_module_stage(stage)), "\""))
        push!(lines, string("file = \"", _batched_nuts_bundle_relative_path(stage), "\""))
    end
    return join(lines, "\n")
end

function _batched_nuts_bundle_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    module_plan = _batched_nuts_module_plan(program; target=target)
    backend = _batched_nuts_bundle_backend(target)
    bundle_symbol = _batched_nuts_bundle_symbol(target)
    plan = BatchedNUTSKernelBundlePlan(
        target,
        module_plan,
        backend,
        bundle_symbol,
        _batched_nuts_bundle_manifest_filename(bundle_symbol),
        "",
        (),
    )
    stages = Tuple(
        _batched_nuts_bundle_stage(plan, stage) for
        stage in _batched_nuts_module_stages(module_plan)
    )
    return BatchedNUTSKernelBundlePlan(
        target,
        module_plan,
        backend,
        bundle_symbol,
        _batched_nuts_bundle_manifest_filename(bundle_symbol),
        _batched_nuts_bundle_manifest_blob(bundle_symbol, stages),
        stages,
    )
end
