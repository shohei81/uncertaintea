    expand_metal_source_plan = UncertainTea._batched_nuts_source_plan(
        expand_direct_program;
        target=:metal,
    )
    @test expand_metal_source_plan.target == :metal
    @test UncertainTea._batched_nuts_source_backend(expand_metal_source_plan) ==
        UncertainTea.NUTSKernelMetalSource
    @test UncertainTea._batched_nuts_source_module(expand_metal_source_plan) ==
        :UncertainTeaMetalSources
    expand_metal_source_stage = UncertainTea._batched_nuts_source_stages(
        expand_metal_source_plan,
    )[4]
    @test UncertainTea._batched_nuts_source_backend(expand_metal_source_stage) ==
        UncertainTea.NUTSKernelMetalSource
    @test UncertainTea._batched_nuts_source_module(expand_metal_source_stage) ==
        :UncertainTeaMetalSources
    @test UncertainTea._batched_nuts_source_entry(expand_metal_source_stage) ==
        :uncertainteametalsources__uncertainteametalartifacts__uncertainteametalbackend__nuts_advance__stage_4__artifact_4__stub
    @test UncertainTea._batched_nuts_source_lines(expand_metal_source_stage)[2] ==
        "const STAGE_KIND = :nuts_advance"
    @test UncertainTea._batched_nuts_source_lines(expand_metal_source_stage)[3] ==
        "const TARGET_POLICY = :metal"
    @test first(UncertainTea._batched_nuts_source_lines(expand_metal_source_stage)) ==
        "module UncertainTeaMetalSources"
    @test last(UncertainTea._batched_nuts_source_lines(expand_metal_source_stage)) == "end"
    @test occursin(
        "function uncertainteametalsources__uncertainteametalartifacts__uncertainteametalbackend__nuts_advance__stage_4__artifact_4__stub(",
        join(UncertainTea._batched_nuts_source_lines(expand_metal_source_stage), "\n"),
    )

    merge_cuda_source_plan = UncertainTea._batched_nuts_source_plan(
        merge_program;
        target=:cuda,
    )
    merge_cuda_source_stage = UncertainTea._batched_nuts_source_stages(
        merge_cuda_source_plan,
    )[3]
    @test UncertainTea._batched_nuts_source_backend(merge_cuda_source_plan) ==
        UncertainTea.NUTSKernelCUDASource
    @test UncertainTea._batched_nuts_source_module(merge_cuda_source_plan) ==
        :UncertainTeaCUDASources
    @test UncertainTea._batched_nuts_source_entry(merge_cuda_source_stage) ==
        :uncertainteacudasources__uncertainteacudaartifacts__uncertainteacudabackend__nuts_merge__stage_3__artifact_3__stub
    @test UncertainTea._batched_nuts_source_lines(merge_cuda_source_stage)[2] ==
        "const STAGE_KIND = :nuts_merge"
    @test all(
        declaration -> occursin("::CUDABuffer", declaration.declaration),
        (
            UncertainTea._batched_nuts_source_device_arguments(merge_cuda_source_stage)...,
            UncertainTea._batched_nuts_source_shared_arguments(merge_cuda_source_stage)...,
        ),
    )

    done_source_plan = UncertainTea._batched_nuts_source_plan(done_program; target=:gpu)
    done_source_stage = UncertainTea._batched_nuts_source_stages(done_source_plan)[1]
    @test done_source_plan.target == :gpu
    @test UncertainTea._batched_nuts_source_backend(done_source_plan) ==
        UncertainTea.NUTSKernelCPUSource
    @test UncertainTea._batched_nuts_source_module(done_source_plan) ==
        :UncertainTeaCPUSources
    @test UncertainTea._batched_nuts_source_entry(done_source_stage) ==
        :uncertainteacpusources__uncertainteacpuartifacts__uncertainteacpubackend__nuts_reload_control__stage_1__artifact_1__stub
    @test UncertainTea._batched_nuts_source_lines(done_source_stage)[2] ==
        "const STAGE_KIND = :nuts_reload_control"
    @test isempty(UncertainTea._batched_nuts_source_shared_arguments(done_source_stage))
    @test_throws ArgumentError UncertainTea._batched_nuts_source_plan(
        done_program;
        target=:bogus,
    )
