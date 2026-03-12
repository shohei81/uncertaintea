    expand_metal_bundle_plan = UncertainTea._batched_nuts_bundle_plan(
        expand_direct_program;
        target=:metal,
    )
    @test expand_metal_bundle_plan.target == :metal
    @test UncertainTea._batched_nuts_bundle_backend(expand_metal_bundle_plan) ==
        UncertainTea.NUTSKernelMetalBundle
    @test UncertainTea._batched_nuts_bundle_symbol(expand_metal_bundle_plan) ==
        :UncertainTeaMetalBundle
    @test UncertainTea._batched_nuts_bundle_manifest_filename(expand_metal_bundle_plan) ==
        "uncertainteametalbundle__manifest.toml"
    @test occursin(
        "bundle = \"UncertainTeaMetalBundle\"",
        UncertainTea._batched_nuts_bundle_manifest_blob(expand_metal_bundle_plan),
    )
    expand_metal_bundle_stage = UncertainTea._batched_nuts_bundle_stages(
        expand_metal_bundle_plan,
    )[4]
    @test UncertainTea._batched_nuts_bundle_symbol(expand_metal_bundle_stage) ==
        :UncertainTeaMetalBundle
    @test UncertainTea._batched_nuts_bundle_relative_path(expand_metal_bundle_stage) ==
        UncertainTea._batched_nuts_module_filename(
            UncertainTea._batched_nuts_bundle_module_stage(expand_metal_bundle_stage),
        )

    merge_cuda_bundle_plan = UncertainTea._batched_nuts_bundle_plan(
        merge_program;
        target=:cuda,
    )
    @test UncertainTea._batched_nuts_bundle_backend(merge_cuda_bundle_plan) ==
        UncertainTea.NUTSKernelCUDABundle
    @test UncertainTea._batched_nuts_bundle_symbol(merge_cuda_bundle_plan) ==
        :UncertainTeaCUDABundle
    @test occursin(
        "count = 4",
        UncertainTea._batched_nuts_bundle_manifest_blob(merge_cuda_bundle_plan),
    )
    merge_cuda_bundle_stage = UncertainTea._batched_nuts_bundle_stages(
        merge_cuda_bundle_plan,
    )[3]
    @test endswith(
        UncertainTea._batched_nuts_bundle_relative_path(merge_cuda_bundle_stage),
        ".cu",
    )

    done_bundle_plan = UncertainTea._batched_nuts_bundle_plan(done_program; target=:gpu)
    done_bundle_stage = UncertainTea._batched_nuts_bundle_stages(done_bundle_plan)[1]
    @test done_bundle_plan.target == :gpu
    @test UncertainTea._batched_nuts_bundle_backend(done_bundle_plan) ==
        UncertainTea.NUTSKernelCPUBundle
    @test UncertainTea._batched_nuts_bundle_symbol(done_bundle_plan) ==
        :UncertainTeaCPUBundle
    @test endswith(
        UncertainTea._batched_nuts_bundle_relative_path(done_bundle_stage),
        ".jl",
    )
    @test_throws ArgumentError UncertainTea._batched_nuts_bundle_plan(
        done_program;
        target=:bogus,
    )
