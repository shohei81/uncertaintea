struct GPUBackendManifestFile
    relative_path::String
    contents::String
end

struct GPUBackendStageFile{N}
    stage_name::N
    relative_path::String
    contents::String
end

struct GPUBackendBundleLayout{M,S}
    target::Symbol
    bundle_symbol::Symbol
    manifest::M
    stages::S
end

struct GPUBackendCodegenStage{N,E}
    stage_name::N
    entry_symbol::E
    filename::String
    source_blob::String
end

struct GPUBackendCodegenBundle{S,M}
    target::Symbol
    bundle_symbol::Symbol
    manifest_lines::M
    stages::S
end

struct GPUBackendFileEntry
    relative_path::String
    contents::String
end

struct GPUBackendPackageLayout{B,F}
    target::Symbol
    backend_symbol::Symbol
    root_dir::String
    bundles::B
    files::F
end

struct GPUBackendEmission{P}
    package::P
    output_root::String
    written_files::Vector{String}
end

gpu_backend_manifest_file(bundle::GPUBackendBundleLayout) = bundle.manifest
gpu_backend_stage_files(bundle::GPUBackendBundleLayout) = bundle.stages
gpu_backend_bundles(layout::GPUBackendPackageLayout) = layout.bundles
gpu_backend_files(layout::GPUBackendPackageLayout) = layout.files
gpu_backend_codegen_manifest_lines(bundle::GPUBackendCodegenBundle) = bundle.manifest_lines
gpu_backend_codegen_stages(bundle::GPUBackendCodegenBundle) = bundle.stages

function GPUBackendPackageLayout(
    target::Symbol,
    backend_symbol::Symbol,
    root_dir::String,
    files,
)
    return GPUBackendPackageLayout(target, backend_symbol, root_dir, (), files)
end
