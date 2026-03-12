_gpu_backend_file_entry(file::GPUBackendManifestFile) = GPUBackendFileEntry(
    file.relative_path,
    file.contents,
)

_gpu_backend_file_entry(file::GPUBackendStageFile) = GPUBackendFileEntry(
    file.relative_path,
    file.contents,
)

function gpu_backend_bundle_layout(
    target::Symbol,
    bundle_symbol::Symbol,
    manifest::GPUBackendManifestFile,
    stages,
)
    return GPUBackendBundleLayout(target, bundle_symbol, manifest, Tuple(stages))
end

function _gpu_backend_bundle_file_entries(bundle::GPUBackendBundleLayout)
    return (
        _gpu_backend_file_entry(gpu_backend_manifest_file(bundle)),
        (_gpu_backend_file_entry(stage) for stage in gpu_backend_stage_files(bundle))...,
    )
end

function _gpu_backend_validate_file_entries(files)
    seen = Set{String}()
    for file in files
        file.relative_path in seen &&
            throw(ArgumentError("duplicate backend file path $(file.relative_path)"))
        push!(seen, file.relative_path)
    end
    return files
end

function gpu_backend_package_layout(
    target::Symbol,
    backend_symbol::Symbol,
    root_dir::String,
    bundles,
)
    bundle_tuple = Tuple(bundles)
    files = Tuple(
        file for bundle in bundle_tuple for file in _gpu_backend_bundle_file_entries(bundle)
    )
    return GPUBackendPackageLayout(
        target,
        backend_symbol,
        root_dir,
        bundle_tuple,
        _gpu_backend_validate_file_entries(files),
    )
end
