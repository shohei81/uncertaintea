    expand_metal_module_plan = UncertainTea._batched_nuts_module_plan(
        expand_direct_program;
        target=:metal,
    )
    @test expand_metal_module_plan.target == :metal
    @test UncertainTea._batched_nuts_module_backend(expand_metal_module_plan) ==
        UncertainTea.NUTSKernelMetalModule
    @test UncertainTea._batched_nuts_module_symbol(expand_metal_module_plan) ==
        :UncertainTeaMetalModules
    expand_metal_module_stage = UncertainTea._batched_nuts_module_stages(
        expand_metal_module_plan,
    )[4]
    @test UncertainTea._batched_nuts_module_backend(expand_metal_module_stage) ==
        UncertainTea.NUTSKernelMetalModule
    @test UncertainTea._batched_nuts_module_symbol(expand_metal_module_stage) ==
        :UncertainTeaMetalModules
    @test UncertainTea._batched_nuts_module_entry_symbol(expand_metal_module_stage) ==
        UncertainTea._batched_nuts_source_entry(
            UncertainTea._batched_nuts_module_source_stage(expand_metal_module_stage),
        )
    @test endswith(
        UncertainTea._batched_nuts_module_filename(expand_metal_module_stage),
        ".metal",
    )
    @test UncertainTea._batched_nuts_module_source_blob(expand_metal_module_stage) ==
        join(
            UncertainTea._batched_nuts_source_lines(
                UncertainTea._batched_nuts_module_source_stage(expand_metal_module_stage),
            ),
            "\n",
        )

    merge_cuda_module_plan = UncertainTea._batched_nuts_module_plan(
        merge_program;
        target=:cuda,
    )
    merge_cuda_module_stage = UncertainTea._batched_nuts_module_stages(
        merge_cuda_module_plan,
    )[3]
    @test UncertainTea._batched_nuts_module_backend(merge_cuda_module_plan) ==
        UncertainTea.NUTSKernelCUDAModule
    @test UncertainTea._batched_nuts_module_symbol(merge_cuda_module_plan) ==
        :UncertainTeaCUDAModules
    @test endswith(
        UncertainTea._batched_nuts_module_filename(merge_cuda_module_stage),
        ".cu",
    )
    @test occursin(
        "::CUDABuffer",
        UncertainTea._batched_nuts_module_source_blob(merge_cuda_module_stage),
    )

    done_module_plan = UncertainTea._batched_nuts_module_plan(done_program; target=:gpu)
    done_module_stage = UncertainTea._batched_nuts_module_stages(done_module_plan)[1]
    @test done_module_plan.target == :gpu
    @test UncertainTea._batched_nuts_module_backend(done_module_plan) ==
        UncertainTea.NUTSKernelCPUModule
    @test UncertainTea._batched_nuts_module_symbol(done_module_plan) ==
        :UncertainTeaCPUModules
    @test endswith(
        UncertainTea._batched_nuts_module_filename(done_module_stage),
        ".jl",
    )
    @test occursin(
        "::AbstractBuffer",
        UncertainTea._batched_nuts_module_source_blob(done_module_stage),
    )
    @test_throws ArgumentError UncertainTea._batched_nuts_module_plan(
        done_program;
        target=:bogus,
    )
