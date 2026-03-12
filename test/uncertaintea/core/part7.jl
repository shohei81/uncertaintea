    expand_metal_codegen_plan = UncertainTea._batched_nuts_codegen_plan(
        expand_direct_program;
        target=:metal,
    )
    @test expand_metal_codegen_plan.target == :metal
    @test UncertainTea._batched_nuts_codegen_module_symbol(expand_metal_codegen_plan) ==
        :UncertainTeaMetalBackend
    expand_metal_codegen_stage = UncertainTea._batched_nuts_codegen_stages(
        expand_metal_codegen_plan,
    )[4]
    @test UncertainTea._batched_nuts_codegen_backend(expand_metal_codegen_stage) ==
        UncertainTea.NUTSKernelMetalCodegen
    @test UncertainTea._batched_nuts_codegen_module_symbol(expand_metal_codegen_stage) ==
        :UncertainTeaMetalBackend
    @test UncertainTea._batched_nuts_codegen_entry_symbol(expand_metal_codegen_stage) ==
        :uncertainteametalbackend__nuts_advance__stage_4
    @test map(
        arg -> arg.symbol,
        UncertainTea._batched_nuts_codegen_shared_arguments(expand_metal_codegen_stage),
    ) == map(
        binding -> Symbol(
            :shared_arg_,
            UncertainTea._batched_nuts_executor_binding_index(binding),
        ),
        UncertainTea._batched_nuts_executor_shared_arguments(
            UncertainTea._batched_nuts_codegen_executor_stage(expand_metal_codegen_stage),
        ),
    )
    @test map(
        arg -> arg.binding.argument_class,
        UncertainTea._batched_nuts_codegen_shared_arguments(expand_metal_codegen_stage),
    ) == (
        UncertainTea.NUTSKernelMetalThreadgroupArgument,
        UncertainTea.NUTSKernelMetalThreadgroupArgument,
        UncertainTea.NUTSKernelMetalThreadgroupArgument,
    )

    merge_cuda_codegen_plan = UncertainTea._batched_nuts_codegen_plan(
        merge_program;
        target=:cuda,
    )
    merge_cuda_codegen_stage = UncertainTea._batched_nuts_codegen_stages(
        merge_cuda_codegen_plan,
    )[3]
    @test UncertainTea._batched_nuts_codegen_module_symbol(merge_cuda_codegen_plan) ==
        :UncertainTeaCUDABackend
    @test UncertainTea._batched_nuts_codegen_backend(merge_cuda_codegen_stage) ==
        UncertainTea.NUTSKernelCUDACodegen
    @test UncertainTea._batched_nuts_codegen_entry_symbol(merge_cuda_codegen_stage) ==
        :uncertainteacudabackend__nuts_merge__stage_3
    @test UncertainTea._batched_nuts_codegen_device_arguments(
        merge_cuda_codegen_stage,
    )[1].symbol == Symbol(
        :device_arg_,
        UncertainTea._batched_nuts_executor_binding_index(
            UncertainTea._batched_nuts_executor_device_arguments(
                UncertainTea._batched_nuts_codegen_executor_stage(merge_cuda_codegen_stage),
            )[1],
        ),
    )
    @test UncertainTea._batched_nuts_codegen_shared_arguments(
        merge_cuda_codegen_stage,
    )[1].symbol == Symbol(
        :shared_arg_,
        UncertainTea._batched_nuts_executor_binding_index(
            UncertainTea._batched_nuts_executor_shared_arguments(
                UncertainTea._batched_nuts_codegen_executor_stage(merge_cuda_codegen_stage),
            )[1],
        ),
    )

    done_codegen_plan = UncertainTea._batched_nuts_codegen_plan(done_program; target=:gpu)
    done_codegen_stage = UncertainTea._batched_nuts_codegen_stages(done_codegen_plan)[1]
    @test done_codegen_plan.target == :gpu
    @test UncertainTea._batched_nuts_codegen_module_symbol(done_codegen_plan) ==
        :UncertainTeaCPUBackend
    @test UncertainTea._batched_nuts_codegen_backend(done_codegen_stage) ==
        UncertainTea.NUTSKernelCPUCodegen
    @test UncertainTea._batched_nuts_codegen_entry_symbol(done_codegen_stage) ==
        :uncertainteacpubackend__nuts_reload_control__stage_1
    @test isempty(UncertainTea._batched_nuts_codegen_shared_arguments(done_codegen_stage))
    @test_throws ArgumentError UncertainTea._batched_nuts_codegen_plan(
        done_program;
        target=:bogus,
    )
