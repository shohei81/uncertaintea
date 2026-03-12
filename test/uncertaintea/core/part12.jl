    expand_metal_package_plan = UncertainTea._batched_nuts_package_plan(
        expand_direct_program;
        target=:metal,
    )
    @test expand_metal_package_plan.target == :metal
    @test UncertainTea._batched_nuts_package_backend(expand_metal_package_plan) ==
        UncertainTea.NUTSKernelMetalPackage
    @test UncertainTea._batched_nuts_package_symbol(expand_metal_package_plan) ==
        :UncertainTeaMetalPackage
    @test UncertainTea._batched_nuts_package_root_dir(expand_metal_package_plan) ==
        "uncertainteametalpackage"
    @test UncertainTea._batched_nuts_package_relative_path(
        UncertainTea._batched_nuts_package_manifest_file(expand_metal_package_plan),
    ) == "uncertainteametalpackage/uncertainteametalbundle__manifest.toml"
    expand_metal_stage_file = UncertainTea._batched_nuts_package_stage_files(
        expand_metal_package_plan,
    )[4]
    @test endswith(
        UncertainTea._batched_nuts_package_relative_path(expand_metal_stage_file),
        ".metal",
    )
    @test UncertainTea._batched_nuts_package_contents(expand_metal_stage_file) ==
        UncertainTea._batched_nuts_module_source_blob(
            UncertainTea._batched_nuts_bundle_module_stage(
                UncertainTea._batched_nuts_package_bundle_stage(expand_metal_stage_file),
            ),
        )

    merge_cuda_package_plan = UncertainTea._batched_nuts_package_plan(
        merge_program;
        target=:cuda,
    )
    @test UncertainTea._batched_nuts_package_backend(merge_cuda_package_plan) ==
        UncertainTea.NUTSKernelCUDAPackage
    @test occursin(
        "count = 4",
        UncertainTea._batched_nuts_package_contents(
            UncertainTea._batched_nuts_package_manifest_file(merge_cuda_package_plan),
        ),
    )
    merge_cuda_stage_file = UncertainTea._batched_nuts_package_stage_files(
        merge_cuda_package_plan,
    )[3]
    @test endswith(
        UncertainTea._batched_nuts_package_relative_path(merge_cuda_stage_file),
        ".cu",
    )

    done_package_plan = UncertainTea._batched_nuts_package_plan(done_program; target=:gpu)
    done_stage_file = UncertainTea._batched_nuts_package_stage_files(done_package_plan)[1]
    @test done_package_plan.target == :gpu
    @test UncertainTea._batched_nuts_package_backend(done_package_plan) ==
        UncertainTea.NUTSKernelCPUPackage
    @test endswith(
        UncertainTea._batched_nuts_package_relative_path(done_stage_file),
        ".jl",
    )
    @test_throws ArgumentError UncertainTea._batched_nuts_package_plan(
        done_program;
        target=:bogus,
    )
