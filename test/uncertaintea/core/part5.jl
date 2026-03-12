    expand_metal_launch_plan = UncertainTea._batched_nuts_launch_plan(
        expand_direct_program;
        target=:metal,
    )
    @test expand_metal_launch_plan.target == :metal
    @test UncertainTea._batched_nuts_launch_binding(
        expand_metal_launch_plan,
        UncertainTea.NUTSKernelBufferControlBlock,
    ).slot == 1
    @test UncertainTea._batched_nuts_launch_binding(
        expand_metal_launch_plan,
        UncertainTea.NUTSKernelBufferTreeNextState,
    ).binding.allocation == UncertainTea.NUTSKernelTargetAllocateDevice
    expand_metal_executor = UncertainTea._batched_nuts_launch_executors(
        expand_metal_launch_plan,
    )[4]
    expand_metal_shared = UncertainTea._batched_nuts_launch_shared_bindings(
        expand_metal_executor,
    )
    @test UncertainTea._batched_nuts_launch_kind(expand_metal_executor) ==
        UncertainTea.NUTSKernelLaunchMetalCompute
    @test UncertainTea._batched_nuts_launch_stage_dataflow(expand_metal_executor) ===
        expand_dataflows[4]
    @test map(binding -> binding.binding.slot.binding.buffer, expand_metal_shared) ==
        (
            UncertainTea.NUTSKernelBufferDescriptorScratch,
            UncertainTea.NUTSKernelBufferTreeEnergy,
            UncertainTea.NUTSKernelBufferSubtreeSummary,
        )
    @test map(binding -> binding.slot, expand_metal_shared) == (1, 2, 3)
    @test map(
        binding -> binding.binding.slot.binding.buffer,
        UncertainTea._batched_nuts_launch_read_bindings(expand_metal_executor),
    ) == UncertainTea._batched_nuts_kernel_reads(expand_dataflows[4])
    @test map(
        binding -> binding.binding.slot.binding.buffer,
        UncertainTea._batched_nuts_launch_write_bindings(expand_metal_executor),
    ) == UncertainTea._batched_nuts_kernel_writes(expand_dataflows[4])

    expand_cuda_launch_plan = UncertainTea._batched_nuts_launch_plan(
        expand_direct_program;
        target=:cuda,
    )
    expand_cuda_executor = UncertainTea._batched_nuts_launch_executors(
        expand_cuda_launch_plan,
    )[4]
    @test UncertainTea._batched_nuts_launch_kind(expand_cuda_executor) ==
        UncertainTea.NUTSKernelLaunchCUDAKernel
    @test UncertainTea._batched_nuts_launch_barriers_after(expand_cuda_executor) ==
        UncertainTea._batched_nuts_target_barriers_after(
            UncertainTea._batched_nuts_launch_target_stage(expand_cuda_executor),
        )
    @test UncertainTea._batched_nuts_launch_stage_binding(
        expand_cuda_executor,
        UncertainTea.NUTSKernelBufferDescriptorScratch,
    ).slot == 1
    @test UncertainTea._batched_nuts_launch_stage_binding(
        expand_cuda_executor,
        UncertainTea.NUTSKernelBufferTreeEnergy,
    ).slot == 2
    @test UncertainTea._batched_nuts_launch_stage_binding(
        expand_cuda_executor,
        UncertainTea.NUTSKernelBufferSubtreeSummary,
    ).slot == 3

    done_launch_plan = UncertainTea._batched_nuts_launch_plan(done_program; target=:gpu)
    done_executor = UncertainTea._batched_nuts_launch_executors(done_launch_plan)[1]
    @test done_launch_plan.target == :gpu
    @test UncertainTea._batched_nuts_launch_kind(done_executor) ==
        UncertainTea.NUTSKernelLaunchSequential
    @test isempty(UncertainTea._batched_nuts_launch_shared_bindings(done_executor))
    @test map(
        binding -> binding.binding.slot.binding.buffer,
        UncertainTea._batched_nuts_launch_read_bindings(done_executor),
    ) == (UncertainTea.NUTSKernelBufferControlBlock,)
    @test map(
        binding -> binding.binding.slot.binding.buffer,
        UncertainTea._batched_nuts_launch_write_bindings(done_executor),
    ) == (
        UncertainTea.NUTSKernelBufferSchedulerState,
        UncertainTea.NUTSKernelBufferControlState,
    )
    @test_throws ArgumentError UncertainTea._batched_nuts_launch_plan(
        done_program;
        target=:bogus,
    )
