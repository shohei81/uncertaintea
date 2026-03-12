    merge_target_plan = UncertainTea._batched_nuts_target_plan(
        merge_program;
        target=:metal,
    )
    @test merge_target_plan.target == :metal
    @test UncertainTea._batched_nuts_target_device_plan(merge_target_plan).target == :metal
    @test UncertainTea._batched_nuts_target_binding(
        merge_target_plan,
        UncertainTea.NUTSKernelBufferControlBlock,
    ).allocation == UncertainTea.NUTSKernelTargetAllocateConstant
    @test UncertainTea._batched_nuts_target_binding(
        merge_target_plan,
        UncertainTea.NUTSKernelBufferDescriptorScratch,
    ).allocation == UncertainTea.NUTSKernelTargetAllocateShared
    merge_target_stage = UncertainTea._batched_nuts_target_stages(merge_target_plan)[3]
    @test UncertainTea._batched_nuts_target_device_stage(merge_target_stage) ===
        UncertainTea._batched_nuts_device_stages(
            UncertainTea._batched_nuts_target_device_plan(merge_target_plan),
        )[3]
    @test UncertainTea._batched_nuts_target_read_bindings(merge_target_stage) ==
        UncertainTea._batched_nuts_device_read_slots(
            UncertainTea._batched_nuts_target_device_stage(merge_target_stage),
        )
    @test isempty(UncertainTea._batched_nuts_target_barriers_after(merge_target_stage))

    expand_metal_target_plan = UncertainTea._batched_nuts_target_plan(
        expand_direct_program;
        target=:metal,
    )
    expand_metal_stage = UncertainTea._batched_nuts_target_stages(expand_metal_target_plan)[2]
    @test UncertainTea._batched_nuts_target_binding(
        expand_metal_target_plan,
        UncertainTea.NUTSKernelBufferTreeEnergy,
    ).allocation == UncertainTea.NUTSKernelTargetAllocateShared
    @test UncertainTea._batched_nuts_target_binding(
        expand_metal_target_plan,
        UncertainTea.NUTSKernelBufferTreeNextState,
    ).allocation == UncertainTea.NUTSKernelTargetAllocateDevice
    @test UncertainTea._batched_nuts_target_barriers_after(expand_metal_stage) ==
        (
            UncertainTea.BatchedNUTSKernelTargetBarrierHint(
                UncertainTea._batched_nuts_device_barriers_after(
                    UncertainTea._batched_nuts_target_device_stage(expand_metal_stage),
                )[1],
                UncertainTea.NUTSKernelTargetMetalThreadgroupBarrier,
            ),
        )

    expand_cuda_target_plan = UncertainTea._batched_nuts_target_plan(
        expand_direct_program;
        target=:cuda,
    )
    expand_cuda_stage = UncertainTea._batched_nuts_target_stages(expand_cuda_target_plan)[2]
    @test UncertainTea._batched_nuts_target_binding(
        expand_cuda_target_plan,
        UncertainTea.NUTSKernelBufferDescriptorScratch,
    ).allocation == UncertainTea.NUTSKernelTargetAllocateShared
    @test UncertainTea._batched_nuts_target_barriers_after(expand_cuda_stage) ==
        (
            UncertainTea.BatchedNUTSKernelTargetBarrierHint(
                UncertainTea._batched_nuts_device_barriers_after(
                    UncertainTea._batched_nuts_target_device_stage(expand_cuda_stage),
                )[1],
                UncertainTea.NUTSKernelTargetCUDAThreadBlockBarrier,
            ),
        )

    done_target_plan = UncertainTea._batched_nuts_target_plan(done_program; target=:gpu)
    done_target_stage = UncertainTea._batched_nuts_target_stages(done_target_plan)[1]
    @test done_target_plan.target == :gpu
    @test UncertainTea._batched_nuts_target_binding(
        done_target_plan,
        UncertainTea.NUTSKernelBufferControlBlock,
    ).allocation == UncertainTea.NUTSKernelTargetAllocateConstant
    @test UncertainTea._batched_nuts_target_binding(
        done_target_plan,
        UncertainTea.NUTSKernelBufferControlState,
    ).allocation == UncertainTea.NUTSKernelTargetAllocateDevice
    @test isempty(UncertainTea._batched_nuts_target_barriers_after(done_target_stage))
    @test_throws ArgumentError UncertainTea._batched_nuts_target_plan(
        done_program;
        target=:bogus,
    )
