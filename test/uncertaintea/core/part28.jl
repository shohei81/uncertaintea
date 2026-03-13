@testset "Tempered SMC NUTS Dataflow" begin
    @tea static function tempered_smc_dataflow_model()
        weights ~ dirichlet([2.0f0, 3.0f0, 4.0f0])
        return weights
    end

    particles = randn(MersenneTwister(280), 2, 6)
    workspace = UncertainTea.TemperedNUTSMoveWorkspace(
        tempered_smc_dataflow_model,
        particles,
        (),
        choicemap(),
    )
    continuations = [
        UncertainTea.NUTSContinuationState(
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            1,
            false,
            false,
        ),
        UncertainTea.NUTSContinuationState(
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            2,
            false,
            false,
        ),
        UncertainTea.NUTSContinuationState(
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            2,
            false,
            false,
        ),
        UncertainTea.NUTSContinuationState(
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            0,
            true,
            false,
        ),
        UncertainTea.NUTSContinuationState(
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            0,
            false,
            true,
        ),
        UncertainTea.NUTSContinuationState(
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            UncertainTea.NUTSState(zeros(2), zeros(2), 0.0, zeros(2)),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            0,
            true,
            true,
        ),
    ]

    @test UncertainTea._begin_tempered_nuts_cohort_scheduler!(
        workspace,
        continuations,
        4,
        MersenneTwister(281),
    ) == UncertainTea.TemperedNUTSSchedulerExpand

    expand_descriptor = UncertainTea._tempered_nuts_scheduler_descriptor(
        UncertainTea._tempered_nuts_scheduler_block(workspace),
    )
    expand_dataflows = UncertainTea._tempered_smc_nuts_kernel_dataflows(expand_descriptor)
    @test length(expand_dataflows) == 3
    @test UncertainTea._tempered_smc_nuts_kernel_step(expand_dataflows[2]) ==
        UncertainTea.TemperedSMCExpandCohortStep
    @test UncertainTea._tempered_smc_nuts_kernel_reads(expand_dataflows[2]) == (
        UncertainTea.TemperedSMCBufferSchedulerState,
        UncertainTea.TemperedSMCBufferControlState,
        UncertainTea.TemperedSMCBufferCohortState,
    )
    @test UncertainTea._tempered_smc_nuts_kernel_read_aliases(expand_dataflows[2]) == (
        UncertainTea.TemperedSMCAliasSchedulerState,
        UncertainTea.TemperedSMCAliasControlState,
        UncertainTea.TemperedSMCAliasCohortState,
    )
    @test UncertainTea._tempered_smc_nuts_kernel_writes(expand_dataflows[2]) == (
        UncertainTea.TemperedSMCBufferControlState,
        UncertainTea.TemperedSMCBufferDescriptorScratch,
        UncertainTea.TemperedSMCBufferCohortState,
        UncertainTea.TemperedSMCBufferCohortSummary,
    )

    expand_schedule = UncertainTea._tempered_smc_nuts_kernel_schedule(expand_descriptor)
    @test length(UncertainTea._tempered_smc_nuts_kernel_schedule_stages(expand_schedule)) == 3
    expand_stage2 = UncertainTea._tempered_smc_nuts_kernel_stage_dataflow(
        UncertainTea._tempered_smc_nuts_kernel_schedule_stages(expand_schedule)[2],
    )
    @test expand_stage2 == expand_dataflows[2]
    expand_plan = UncertainTea._tempered_smc_nuts_kernel_resource_plan(expand_descriptor)
    @test length(UncertainTea._tempered_smc_nuts_kernel_resource_groups(expand_plan)) == 6
    @test length(UncertainTea._tempered_smc_nuts_kernel_barriers(expand_plan)) == 2
    @test UncertainTea._tempered_smc_nuts_kernel_barriers_after(expand_plan, 2) ==
        (
            UncertainTea.TemperedSMCNUTSKernelBarrierPlacement(
                2,
                UncertainTea.TemperedSMCKernelDependencyBarrier,
                (
                    UncertainTea.TemperedSMCAliasControlState,
                    UncertainTea.TemperedSMCAliasCohortSummary,
                ),
                (
                    UncertainTea.TemperedSMCBufferControlState,
                    UncertainTea.TemperedSMCBufferCohortSummary,
                ),
            ),
        )

    merge_descriptor = UncertainTea._tempered_nuts_scheduler_descriptor(
        UncertainTea._tempered_nuts_scheduler_block(
            UncertainTea.TemperedNUTSMergeIR(
                workspace.scheduler.active_depth,
                workspace.scheduler.active_depth_count,
                copy(workspace.scheduler.cohort_active),
            ),
        ),
    )
    merge_dataflows = UncertainTea._tempered_smc_nuts_kernel_dataflows(merge_descriptor)
    @test length(merge_dataflows) == 3
    @test UncertainTea._tempered_smc_nuts_kernel_step(merge_dataflows[2]) ==
        UncertainTea.TemperedSMCMergeCohortStep
    @test UncertainTea._tempered_smc_nuts_kernel_reads(merge_dataflows[2]) == (
        UncertainTea.TemperedSMCBufferSchedulerState,
        UncertainTea.TemperedSMCBufferControlState,
        UncertainTea.TemperedSMCBufferDescriptorScratch,
        UncertainTea.TemperedSMCBufferCohortState,
        UncertainTea.TemperedSMCBufferCohortSummary,
    )
    @test UncertainTea._tempered_smc_nuts_kernel_writes(merge_dataflows[2]) == (
        UncertainTea.TemperedSMCBufferSchedulerState,
        UncertainTea.TemperedSMCBufferControlState,
        UncertainTea.TemperedSMCBufferDescriptorScratch,
        UncertainTea.TemperedSMCBufferCohortSummary,
    )
    merge_plan = UncertainTea._tempered_smc_nuts_kernel_resource_plan(merge_descriptor)
    @test length(UncertainTea._tempered_smc_nuts_kernel_resource_groups(merge_plan)) == 6
    @test length(UncertainTea._tempered_smc_nuts_kernel_barriers(merge_plan)) == 2

    done_descriptor = UncertainTea._tempered_nuts_scheduler_descriptor(
        UncertainTea.TemperedNUTSDoneBlock(UncertainTea.TemperedNUTSDoneIR()),
    )
    done_dataflows = UncertainTea._tempered_smc_nuts_kernel_dataflows(done_descriptor)
    @test length(done_dataflows) == 1
    @test UncertainTea._tempered_smc_nuts_kernel_reads(done_dataflows[1]) ==
        (UncertainTea.TemperedSMCBufferControlBlock,)
    done_plan = UncertainTea._tempered_smc_nuts_kernel_resource_plan(done_descriptor)
    @test isempty(UncertainTea._tempered_smc_nuts_kernel_barriers(done_plan))

    bundle = tempered_smc_nuts_codegen_bundle(tempered_smc_dataflow_model, workspace)
    manifest = join(gpu_backend_codegen_manifest_lines(bundle), "\n")
    @test occursin("smc_tempered_nuts_expand_schedule_stages = 3", manifest)
    @test occursin("smc_tempered_nuts_expand_resource_groups = 6", manifest)
    @test occursin("smc_tempered_nuts_expand_barriers = 2", manifest)

    layout = tempered_smc_nuts_package_layout(tempered_smc_dataflow_model, workspace; target=:gpu)
    @test any(occursin("const DATAFLOW_STEP_COUNT = 3", file.contents) for file in gpu_backend_files(layout))
    @test any(occursin("const RESOURCE_GROUP_COUNT = 6", file.contents) for file in gpu_backend_files(layout))
    @test any(occursin("const BARRIER_COUNT = 2", file.contents) for file in gpu_backend_files(layout))
end
