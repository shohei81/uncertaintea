function _batched_nuts_package_file_entries(plan::BatchedNUTSKernelPackagePlan)
    manifest_file = _batched_nuts_package_manifest_file(plan)
    manifest_entry = GPUBackendFileEntry(
        _batched_nuts_package_relative_path(manifest_file),
        _batched_nuts_package_contents(manifest_file),
    )
    stage_entries = Tuple(
        GPUBackendFileEntry(
            _batched_nuts_package_relative_path(stage_file),
            _batched_nuts_package_contents(stage_file),
        ) for stage_file in _batched_nuts_package_stage_files(plan)
    )
    return (manifest_entry, stage_entries...)
end

function batched_nuts_package_layout(plan::BatchedNUTSKernelPackagePlan)
    return GPUBackendPackageLayout(
        plan.target,
        _batched_nuts_package_symbol(plan),
        _batched_nuts_package_root_dir(plan),
        _batched_nuts_package_file_entries(plan),
    )
end

function batched_nuts_package_layout(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    return batched_nuts_package_layout(
        _batched_nuts_package_plan(program; target=target),
    )
end

function emit_batched_nuts_package(
    plan::BatchedNUTSKernelPackagePlan,
    output_root::AbstractString;
    overwrite::Bool=false,
)
    return emit_gpu_backend_package(
        batched_nuts_package_layout(plan),
        output_root;
        overwrite=overwrite,
    )
end

function emit_batched_nuts_package(
    program::AbstractBatchedNUTSKernelProgram,
    output_root::AbstractString;
    target::Symbol=:gpu,
    overwrite::Bool=false,
)
    return emit_batched_nuts_package(
        _batched_nuts_package_plan(program; target=target),
        output_root;
        overwrite=overwrite,
    )
end
