    expand_metal_artifact_plan = UncertainTea._batched_nuts_artifact_plan(
        expand_direct_program;
        target=:metal,
    )
    @test expand_metal_artifact_plan.target == :metal
    @test UncertainTea._batched_nuts_artifact_backend(expand_metal_artifact_plan) ==
        UncertainTea.NUTSKernelMetalArtifact
    @test UncertainTea._batched_nuts_artifact_module_symbol(expand_metal_artifact_plan) ==
        :UncertainTeaMetalArtifacts
    expand_metal_artifact_stage = UncertainTea._batched_nuts_artifact_stages(
        expand_metal_artifact_plan,
    )[4]
    @test UncertainTea._batched_nuts_artifact_backend(expand_metal_artifact_stage) ==
        UncertainTea.NUTSKernelMetalArtifact
    @test UncertainTea._batched_nuts_artifact_module_symbol(expand_metal_artifact_stage) ==
        :UncertainTeaMetalArtifacts
    @test UncertainTea._batched_nuts_artifact_symbol(expand_metal_artifact_stage) ==
        :uncertainteametalartifacts__uncertainteametalbackend__nuts_advance__stage_4__artifact_4
    @test map(
        UncertainTea._batched_nuts_artifact_argument_symbol,
        UncertainTea._batched_nuts_artifact_shared_arguments(expand_metal_artifact_stage),
    ) == map(
        arg -> arg.symbol,
        UncertainTea._batched_nuts_codegen_shared_arguments(
            UncertainTea._batched_nuts_artifact_codegen_stage(expand_metal_artifact_stage),
        ),
    )

    merge_cuda_artifact_plan = UncertainTea._batched_nuts_artifact_plan(
        merge_program;
        target=:cuda,
    )
    merge_cuda_artifact_stage = UncertainTea._batched_nuts_artifact_stages(
        merge_cuda_artifact_plan,
    )[3]
    @test UncertainTea._batched_nuts_artifact_backend(merge_cuda_artifact_plan) ==
        UncertainTea.NUTSKernelCUDAArtifact
    @test UncertainTea._batched_nuts_artifact_module_symbol(merge_cuda_artifact_plan) ==
        :UncertainTeaCUDAArtifacts
    @test UncertainTea._batched_nuts_artifact_symbol(merge_cuda_artifact_stage) ==
        :uncertainteacudaartifacts__uncertainteacudabackend__nuts_merge__stage_3__artifact_3
    @test map(
        arg -> arg.slot,
        UncertainTea._batched_nuts_artifact_device_arguments(merge_cuda_artifact_stage),
    ) == Tuple(
        1:length(
            UncertainTea._batched_nuts_artifact_device_arguments(merge_cuda_artifact_stage),
        )
    )
    @test map(
        UncertainTea._batched_nuts_artifact_argument_symbol,
        UncertainTea._batched_nuts_artifact_shared_arguments(merge_cuda_artifact_stage),
    ) == map(
        arg -> arg.symbol,
        UncertainTea._batched_nuts_codegen_shared_arguments(
            UncertainTea._batched_nuts_artifact_codegen_stage(merge_cuda_artifact_stage),
        ),
    )

    done_artifact_plan = UncertainTea._batched_nuts_artifact_plan(done_program; target=:gpu)
    done_artifact_stage = UncertainTea._batched_nuts_artifact_stages(done_artifact_plan)[1]
    @test done_artifact_plan.target == :gpu
    @test UncertainTea._batched_nuts_artifact_backend(done_artifact_plan) ==
        UncertainTea.NUTSKernelCPUArtifact
    @test UncertainTea._batched_nuts_artifact_module_symbol(done_artifact_plan) ==
        :UncertainTeaCPUArtifacts
    @test UncertainTea._batched_nuts_artifact_symbol(done_artifact_stage) ==
        :uncertainteacpuartifacts__uncertainteacpubackend__nuts_reload_control__stage_1__artifact_1
    @test isempty(UncertainTea._batched_nuts_artifact_shared_arguments(done_artifact_stage))
    @test_throws ArgumentError UncertainTea._batched_nuts_artifact_plan(
        done_program;
        target=:bogus,
    )
