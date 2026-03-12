function _gpu_backend_require_target(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported GPU backend target $(target)"))
    return target
end

gpu_backend_target_name(::Val{:gpu}) = "GPU"
gpu_backend_target_name(::Val{:metal}) = "Metal"
gpu_backend_target_name(::Val{:cuda}) = "CUDA"

function gpu_backend_target_name(target::Symbol)
    return gpu_backend_target_name(Val(_gpu_backend_require_target(target)))
end

gpu_backend_module_extension(::Val{:gpu}) = ".jl"
gpu_backend_module_extension(::Val{:metal}) = ".metal"
gpu_backend_module_extension(::Val{:cuda}) = ".cu"

function gpu_backend_module_extension(target::Symbol)
    return gpu_backend_module_extension(Val(_gpu_backend_require_target(target)))
end

function gpu_backend_module_filename(
    module_symbol::Symbol,
    entry_symbol::Union{Nothing,Symbol},
    target::Symbol,
)
    stem = lowercase(String(module_symbol))
    suffix = isnothing(entry_symbol) ? "" : string("__", entry_symbol)
    return string(stem, suffix, gpu_backend_module_extension(target))
end

gpu_backend_buffer_argument_type(::Val{:gpu}) = :AbstractBuffer
gpu_backend_buffer_argument_type(::Val{:metal}) = :MetalBuffer
gpu_backend_buffer_argument_type(::Val{:cuda}) = :CUDABuffer

function gpu_backend_buffer_argument_type(target::Symbol)
    return gpu_backend_buffer_argument_type(Val(_gpu_backend_require_target(target)))
end

function gpu_backend_buffer_argument_declaration(
    target::Symbol,
    argument_symbol::Symbol,
)
    return string(argument_symbol, "::", gpu_backend_buffer_argument_type(target))
end
