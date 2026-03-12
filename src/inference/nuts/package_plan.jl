@enum BatchedNUTSKernelPackageBackend::UInt8 begin
    NUTSKernelCPUPackage = 0
    NUTSKernelMetalPackage = 1
    NUTSKernelCUDAPackage = 2
end

struct BatchedNUTSKernelPackageManifestFile
    relative_path::String
    contents::String
end

struct BatchedNUTSKernelPackageStageFile{S}
    bundle_stage::S
    relative_path::String
    contents::String
end

struct BatchedNUTSKernelPackagePlan{P,S}
    target::Symbol
    bundle_plan::P
    backend::BatchedNUTSKernelPackageBackend
    package_symbol::Symbol
    root_dir::String
    manifest_file::BatchedNUTSKernelPackageManifestFile
    stage_files::S
end

_batched_nuts_package_bundle_plan(plan::BatchedNUTSKernelPackagePlan) = plan.bundle_plan
_batched_nuts_package_backend(plan::BatchedNUTSKernelPackagePlan) = plan.backend
_batched_nuts_package_symbol(plan::BatchedNUTSKernelPackagePlan) = plan.package_symbol
_batched_nuts_package_root_dir(plan::BatchedNUTSKernelPackagePlan) = plan.root_dir
_batched_nuts_package_manifest_file(plan::BatchedNUTSKernelPackagePlan) = plan.manifest_file
_batched_nuts_package_stage_files(plan::BatchedNUTSKernelPackagePlan) = plan.stage_files

_batched_nuts_package_bundle_stage(file::BatchedNUTSKernelPackageStageFile) = file.bundle_stage
_batched_nuts_package_relative_path(file::Union{
    BatchedNUTSKernelPackageManifestFile,
    BatchedNUTSKernelPackageStageFile,
}) = file.relative_path
_batched_nuts_package_contents(file::Union{
    BatchedNUTSKernelPackageManifestFile,
    BatchedNUTSKernelPackageStageFile,
}) = file.contents

_batched_nuts_package_backend(::Val{:gpu}) = NUTSKernelCPUPackage
_batched_nuts_package_backend(::Val{:metal}) = NUTSKernelMetalPackage
_batched_nuts_package_backend(::Val{:cuda}) = NUTSKernelCUDAPackage

function _batched_nuts_package_backend(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS package target $(target)"))
    return _batched_nuts_package_backend(Val(target))
end

_batched_nuts_package_symbol(::Val{:gpu}) = :UncertainTeaCPUPackage
_batched_nuts_package_symbol(::Val{:metal}) = :UncertainTeaMetalPackage
_batched_nuts_package_symbol(::Val{:cuda}) = :UncertainTeaCUDAPackage

function _batched_nuts_package_symbol(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS package target $(target)"))
    return _batched_nuts_package_symbol(Val(target))
end

function _batched_nuts_package_root_dir(package_symbol::Symbol)
    return lowercase(String(package_symbol))
end

function _batched_nuts_package_manifest_entry(
    plan::BatchedNUTSKernelPackagePlan,
)
    return BatchedNUTSKernelPackageManifestFile(
        joinpath(
            plan.root_dir,
            _batched_nuts_bundle_manifest_filename(_batched_nuts_package_bundle_plan(plan)),
        ),
        _batched_nuts_bundle_manifest_blob(_batched_nuts_package_bundle_plan(plan)),
    )
end

function _batched_nuts_package_stage_file(
    plan::BatchedNUTSKernelPackagePlan,
    bundle_stage::BatchedNUTSKernelBundleStage,
)
    module_stage = _batched_nuts_bundle_module_stage(bundle_stage)
    return BatchedNUTSKernelPackageStageFile(
        bundle_stage,
        joinpath(plan.root_dir, _batched_nuts_bundle_relative_path(bundle_stage)),
        _batched_nuts_module_source_blob(module_stage),
    )
end

function _batched_nuts_package_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    bundle_plan = _batched_nuts_bundle_plan(program; target=target)
    backend = _batched_nuts_package_backend(target)
    package_symbol = _batched_nuts_package_symbol(target)
    root_dir = _batched_nuts_package_root_dir(package_symbol)
    plan = BatchedNUTSKernelPackagePlan(
        target,
        bundle_plan,
        backend,
        package_symbol,
        root_dir,
        BatchedNUTSKernelPackageManifestFile("", ""),
        (),
    )
    manifest_file = _batched_nuts_package_manifest_entry(plan)
    stage_files = Tuple(
        _batched_nuts_package_stage_file(plan, stage) for
        stage in _batched_nuts_bundle_stages(bundle_plan)
    )
    return BatchedNUTSKernelPackagePlan(
        target,
        bundle_plan,
        backend,
        package_symbol,
        root_dir,
        manifest_file,
        stage_files,
    )
end
