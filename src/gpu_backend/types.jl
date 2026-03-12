struct GPUBackendFileEntry
    relative_path::String
    contents::String
end

struct GPUBackendPackageLayout{F}
    target::Symbol
    backend_symbol::Symbol
    root_dir::String
    files::F
end

struct GPUBackendEmission{P}
    package::P
    output_root::String
    written_files::Vector{String}
end

gpu_backend_files(layout::GPUBackendPackageLayout) = layout.files
