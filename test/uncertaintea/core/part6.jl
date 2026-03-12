    expand_metal_executor_plan = UncertainTea._batched_nuts_executor_plan(
        expand_direct_program;
        target=:metal,
    )
    @test expand_metal_executor_plan.target == :metal
    @test UncertainTea._batched_nuts_executor_binding(
        expand_metal_executor_plan,
        UncertainTea.NUTSKernelBufferControlBlock,
    ).argument_class == UncertainTea.NUTSKernelConstantArgument
    @test UncertainTea._batched_nuts_executor_binding(
        expand_metal_executor_plan,
        UncertainTea.NUTSKernelBufferTreeNextState,
    ).argument_class == UncertainTea.NUTSKernelDeviceArgument
    expand_metal_executor = UncertainTea._batched_nuts_executor_stages(
        expand_metal_executor_plan,
    )[4]
    @test UncertainTea._batched_nuts_executor_kind(expand_metal_executor) ==
        UncertainTea.NUTSKernelMetalExecutor
    @test UncertainTea._batched_nuts_executor_kernel_symbol(expand_metal_executor) ==
        :nuts_advance
    @test map(
        binding -> binding.argument_class,
        UncertainTea._batched_nuts_executor_shared_arguments(expand_metal_executor),
    ) == (
        UncertainTea.NUTSKernelMetalThreadgroupArgument,
        UncertainTea.NUTSKernelMetalThreadgroupArgument,
        UncertainTea.NUTSKernelMetalThreadgroupArgument,
    )
    @test UncertainTea._batched_nuts_executor_barriers_after(expand_metal_executor) ==
        UncertainTea._batched_nuts_launch_barriers_after(
            UncertainTea._batched_nuts_executor_launch_stage(expand_metal_executor),
        )

    merge_cuda_executor_plan = UncertainTea._batched_nuts_executor_plan(
        merge_program;
        target=:cuda,
    )
    merge_cuda_executor = UncertainTea._batched_nuts_executor_stages(
        merge_cuda_executor_plan,
    )[3]
    @test UncertainTea._batched_nuts_executor_kind(merge_cuda_executor) ==
        UncertainTea.NUTSKernelCUDAExecutor
    @test UncertainTea._batched_nuts_executor_kernel_symbol(merge_cuda_executor) ==
        :nuts_merge
    @test UncertainTea._batched_nuts_executor_stage_binding(
        merge_cuda_executor,
        UncertainTea.NUTSKernelBufferDescriptorScratch,
    ).argument_class == UncertainTea.NUTSKernelCUDASharedArgument
    @test UncertainTea._batched_nuts_executor_stage_binding(
        merge_cuda_executor,
        UncertainTea.NUTSKernelBufferContinuationSummary,
    ).argument_class == UncertainTea.NUTSKernelDeviceArgument

    done_executor_plan = UncertainTea._batched_nuts_executor_plan(
        done_program;
        target=:gpu,
    )
    done_executor = UncertainTea._batched_nuts_executor_stages(done_executor_plan)[1]
    @test done_executor_plan.target == :gpu
    @test UncertainTea._batched_nuts_executor_kind(done_executor) ==
        UncertainTea.NUTSKernelSequentialExecutor
    @test UncertainTea._batched_nuts_executor_kernel_symbol(done_executor) ==
        :nuts_reload_control
    @test isempty(UncertainTea._batched_nuts_executor_shared_arguments(done_executor))
    @test map(
        binding -> binding.argument_class,
        UncertainTea._batched_nuts_executor_device_arguments(done_executor),
    ) == (
        UncertainTea.NUTSKernelDeviceArgument,
        UncertainTea.NUTSKernelDeviceArgument,
    )
    @test_throws ArgumentError UncertainTea._batched_nuts_executor_plan(
        done_program;
        target=:bogus,
    )
