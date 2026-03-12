function gpu_backend_codegen_bundle(
    target::Symbol,
    bundle_symbol::Symbol,
    stages;
    manifest_lines=(),
)
    return GPUBackendCodegenBundle(
        target,
        bundle_symbol,
        Tuple(string(line) for line in manifest_lines),
        Tuple(stages),
    )
end

function _gpu_backend_codegen_manifest_filename(bundle::GPUBackendCodegenBundle)
    return string(lowercase(String(bundle.bundle_symbol)), "__manifest.toml")
end

function _gpu_backend_codegen_manifest_contents(bundle::GPUBackendCodegenBundle)
    lines = String[
        string("target = \"", bundle.target, "\""),
        string("bundle = \"", bundle.bundle_symbol, "\""),
    ]
    append!(lines, gpu_backend_codegen_manifest_lines(bundle))
    push!(lines, string("count = ", length(gpu_backend_codegen_stages(bundle))))
    for (index, stage) in enumerate(gpu_backend_codegen_stages(bundle))
        push!(lines, "[[stage]]")
        push!(lines, string("index = ", index))
        push!(lines, string("name = \"", stage.stage_name, "\""))
        push!(lines, string("kind = \"", gpu_backend_stage_kind(stage), "\""))
        push!(lines, string("entry = \"", stage.entry_symbol, "\""))
        push!(lines, string("file = \"", stage.filename, "\""))
    end
    return join(lines, "\n")
end

function _gpu_backend_bundle_layout(
    root_dir::String,
    bundle::GPUBackendCodegenBundle,
)
    stages = Tuple(
        GPUBackendStageFile(
            stage.stage_name,
            joinpath(root_dir, stage.filename),
            stage.source_blob,
        ) for stage in gpu_backend_codegen_stages(bundle)
    )
    return gpu_backend_bundle_layout(
        bundle.target,
        bundle.bundle_symbol,
        GPUBackendManifestFile(
            joinpath(root_dir, _gpu_backend_codegen_manifest_filename(bundle)),
            _gpu_backend_codegen_manifest_contents(bundle),
        ),
        stages,
    )
end

function gpu_backend_codegen_package_layout(
    target::Symbol,
    backend_symbol::Symbol,
    root_dir::String,
    bundles,
)
    bundle_tuple = Tuple(
        _gpu_backend_bundle_layout(root_dir, bundle) for bundle in Tuple(bundles)
    )
    return gpu_backend_package_layout(target, backend_symbol, root_dir, bundle_tuple)
end
