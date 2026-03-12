function _gpu_backend_output_path(
    output_root::AbstractString,
    file::GPUBackendFileEntry,
)
    isabspath(file.relative_path) &&
        throw(ArgumentError("backend file path must be relative: $(file.relative_path)"))
    root = normpath(String(output_root))
    path = normpath(joinpath(root, file.relative_path))
    relative = relpath(path, root)
    if relative == ".." || startswith(relative, string("..", Base.Filesystem.path_separator))
        throw(ArgumentError("backend file path escapes output root: $(file.relative_path)"))
    end
    return path
end

function emit_gpu_backend_package(
    layout::GPUBackendPackageLayout,
    output_root::AbstractString;
    overwrite::Bool=false,
)
    written_files = String[]
    for file in gpu_backend_files(layout)
        path = _gpu_backend_output_path(output_root, file)
        if ispath(path) && !overwrite
            throw(ArgumentError("refusing to overwrite existing backend file $(path)"))
        end
        mkpath(dirname(path))
        write(path, file.contents)
        push!(written_files, path)
    end
    return GPUBackendEmission(layout, String(output_root), written_files)
end
