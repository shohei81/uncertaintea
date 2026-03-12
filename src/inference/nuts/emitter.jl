function _batched_nuts_backend_manifest_file(plan::BatchedNUTSKernelPackagePlan)
    manifest_file = _batched_nuts_package_manifest_file(plan)
    return GPUBackendManifestFile(
        _batched_nuts_package_relative_path(manifest_file),
        _batched_nuts_package_contents(manifest_file),
    )
end

function _batched_nuts_backend_stage_files(plan::BatchedNUTSKernelPackagePlan)
    return Tuple(
        GPUBackendStageFile(
            _batched_nuts_module_entry_symbol(
                _batched_nuts_bundle_module_stage(
                    _batched_nuts_package_bundle_stage(stage_file),
                ),
            ),
            _batched_nuts_package_relative_path(stage_file),
            _batched_nuts_package_contents(stage_file),
        ) for stage_file in _batched_nuts_package_stage_files(plan)
    )
end

function _batched_nuts_backend_bundle_layout(plan::BatchedNUTSKernelPackagePlan)
    bundle_symbol = _batched_nuts_bundle_symbol(_batched_nuts_package_bundle_plan(plan))
    stages = Tuple(
        GPUBackendCodegenStage(
            stage_file.stage_name,
            stage_file.stage_name,
            basename(stage_file.relative_path),
            stage_file.contents,
        ) for stage_file in _batched_nuts_backend_stage_files(plan)
    )
    return gpu_backend_codegen_bundle(
        plan.target,
        bundle_symbol,
        stages;
        manifest_lines=(
            string("source_manifest = \"", basename(_batched_nuts_package_relative_path(_batched_nuts_package_manifest_file(plan))), "\""),
        ),
    )
end

function batched_nuts_package_layout(plan::BatchedNUTSKernelPackagePlan)
    return gpu_backend_codegen_package_layout(
        plan.target,
        _batched_nuts_package_symbol(plan),
        _batched_nuts_package_root_dir(plan),
        (_batched_nuts_backend_bundle_layout(plan),),
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
