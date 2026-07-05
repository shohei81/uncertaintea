module UncertainTeaMetalExt

using UncertainTea
using Metal

# Metal GPUs are Float32-only: name Float32 as the natural precision and disallow a
# Float64 request so the core precision guard raises a clear error rather than
# silently producing wrong results.
UncertainTea.default_device_precision(::Metal.MetalBackend) = Float32
UncertainTea._device_supports_float64(::Metal.MetalBackend) = false

end # module
