    @test gpu_backend_target_name(:gpu) == "GPU"
    @test gpu_backend_target_name(:metal) == "Metal"
    @test gpu_backend_target_name(:cuda) == "CUDA"
    @test gpu_backend_module_extension(:gpu) == ".jl"
    @test gpu_backend_module_extension(:metal) == ".metal"
    @test gpu_backend_module_extension(:cuda) == ".cu"
    @test gpu_backend_module_filename(:ManualModule, nothing, :gpu) == "manualmodule.jl"
    @test gpu_backend_module_filename(:ManualModule, :entry, :metal) ==
        "manualmodule__entry.metal"
    @test gpu_backend_module_filename(:ManualModule, :entry, :cuda) ==
        "manualmodule__entry.cu"
    @test gpu_backend_buffer_argument_type(:gpu) == :AbstractBuffer
    @test gpu_backend_buffer_argument_type(:metal) == :MetalBuffer
    @test gpu_backend_buffer_argument_type(:cuda) == :CUDABuffer
    @test gpu_backend_buffer_argument_declaration(:gpu, :buf) == "buf::AbstractBuffer"
    @test gpu_backend_buffer_argument_declaration(:metal, :buf) == "buf::MetalBuffer"
    @test gpu_backend_buffer_argument_declaration(:cuda, :buf) == "buf::CUDABuffer"
    @test_throws ArgumentError gpu_backend_target_name(:bogus)
    @test_throws ArgumentError gpu_backend_module_extension(:bogus)
    @test_throws ArgumentError gpu_backend_buffer_argument_type(:bogus)
