    merge_state = UncertainTea._batched_nuts_step_state(gaussian_cohort_scheduler_workspace)
    @test merge_state isa UncertainTea.BatchedNUTSMergeStepState
    @test merge_state.descriptor.select_proposal ===
        gaussian_cohort_scheduler_workspace.continuation_select_proposal
    @test merge_state.proposal_energy ===
        gaussian_cohort_scheduler_workspace.subtree_proposal_energy
    @test merge_state.proposal_energy_error ===
        gaussian_cohort_scheduler_workspace.subtree_proposal_energy_error
    @test merge_state.candidate_log_weight ===
        gaussian_cohort_scheduler_workspace.continuation_candidate_log_weight
    @test merge_state.combined_log_weight ===
        gaussian_cohort_scheduler_workspace.continuation_combined_log_weight
    merge_frame = UncertainTea._batched_nuts_kernel_frame(gaussian_cohort_scheduler_workspace)
    @test merge_frame isa UncertainTea.BatchedNUTSMergeKernelFrame
    @test merge_frame.state.candidate_log_weight ===
        gaussian_cohort_scheduler_workspace.continuation_candidate_log_weight
    @test merge_frame.left_position === gaussian_cohort_scheduler_workspace.left_position
    @test merge_frame.right_position === gaussian_cohort_scheduler_workspace.right_position
    @test merge_frame.tree_proposal_position ===
        gaussian_cohort_scheduler_workspace.tree_proposal_position
    @test merge_frame.proposal_position === gaussian_cohort_scheduler_workspace.proposal_position
    @test merge_frame.proposed_logjoint === gaussian_cohort_scheduler_workspace.proposed_logjoint
    @test merge_frame.continuation_proposal_logjoint ===
        gaussian_cohort_scheduler_workspace.continuation_proposal_logjoint
    @test merge_frame.continuation_log_weight ===
        gaussian_cohort_scheduler_workspace.continuation_log_weight
    @test UncertainTea._batched_nuts_kernel_frame(
        gaussian_cohort_scheduler_workspace,
        merge_state,
    ).state === merge_state
    merge_access = UncertainTea._batched_nuts_kernel_access(
        gaussian_cohort_scheduler_workspace,
        merge_frame,
    )
    @test merge_access isa UncertainTea.BatchedNUTSMergeKernelAccess
    @test merge_access.block === merge_frame.state.descriptor.block
    @test merge_access.select_proposal ===
        gaussian_cohort_scheduler_workspace.continuation_select_proposal
    @test merge_access.left_position === gaussian_cohort_scheduler_workspace.left_position
    @test merge_access.right_position === gaussian_cohort_scheduler_workspace.right_position
    @test merge_access.proposal_position === gaussian_cohort_scheduler_workspace.proposal_position
    @test merge_access.continuation_log_weight ===
        gaussian_cohort_scheduler_workspace.continuation_log_weight
    merge_program = UncertainTea._batched_nuts_kernel_program(gaussian_cohort_scheduler_workspace)
    @test merge_program isa UncertainTea.BatchedNUTSMergeKernelProgram
    @test UncertainTea._batched_nuts_kernel_access(merge_program) isa
        UncertainTea.BatchedNUTSMergeKernelAccess
    @test UncertainTea._batched_nuts_kernel_access(merge_program).left_position ===
        merge_access.left_position
    @test UncertainTea._batched_nuts_kernel_ops(merge_program) ==
        (
            UncertainTea.NUTSKernelReloadControl,
            UncertainTea.NUTSKernelActivateMerge,
            UncertainTea.NUTSKernelMerge,
            UncertainTea.NUTSKernelTransitionPhase,
        )
    @test typeof.(UncertainTea._batched_nuts_kernel_steps(merge_program)) ==
        (
            UncertainTea.BatchedNUTSReloadControlStep,
            UncertainTea.BatchedNUTSActivateMergeStep,
            UncertainTea.BatchedNUTSMergeStep,
            UncertainTea.BatchedNUTSTransitionPhaseStep,
        )
    merge_dataflows = UncertainTea._batched_nuts_kernel_dataflows(merge_program)
    @test length(merge_dataflows) == 4
    @test UncertainTea._batched_nuts_kernel_step(merge_dataflows[3]) isa
        UncertainTea.BatchedNUTSMergeStep
    @test UncertainTea._batched_nuts_kernel_access(merge_dataflows[3]).proposal_position ===
        merge_access.proposal_position
    @test UncertainTea._batched_nuts_kernel_reads(merge_dataflows[3]) ==
        (
            UncertainTea.NUTSKernelBufferControlState,
            UncertainTea.NUTSKernelBufferDescriptorScratch,
            UncertainTea.NUTSKernelBufferTreeFrontierState,
            UncertainTea.NUTSKernelBufferTreeProposalState,
            UncertainTea.NUTSKernelBufferSubtreeSummary,
            UncertainTea.NUTSKernelBufferContinuationFrontierState,
            UncertainTea.NUTSKernelBufferContinuationProposalState,
            UncertainTea.NUTSKernelBufferContinuationSummary,
            UncertainTea.NUTSKernelBufferTreeEnergy,
        )
    @test UncertainTea._batched_nuts_kernel_read_aliases(merge_dataflows[3]) ==
        (
            UncertainTea.NUTSKernelAliasControlState,
            UncertainTea.NUTSKernelAliasDescriptorScratch,
            UncertainTea.NUTSKernelAliasTreeState,
            UncertainTea.NUTSKernelAliasTreeState,
            UncertainTea.NUTSKernelAliasSubtreeSummary,
            UncertainTea.NUTSKernelAliasContinuationState,
            UncertainTea.NUTSKernelAliasContinuationState,
            UncertainTea.NUTSKernelAliasContinuationSummary,
            UncertainTea.NUTSKernelAliasTreeEnergy,
        )
    @test UncertainTea._batched_nuts_kernel_writes(merge_dataflows[3]) ==
        (
            UncertainTea.NUTSKernelBufferDescriptorScratch,
            UncertainTea.NUTSKernelBufferControlState,
            UncertainTea.NUTSKernelBufferContinuationFrontierState,
            UncertainTea.NUTSKernelBufferContinuationProposalState,
            UncertainTea.NUTSKernelBufferContinuationSummary,
        )
    @test UncertainTea._batched_nuts_kernel_dependencies(merge_program) ==
        (
            UncertainTea.BatchedNUTSKernelDependency(
                2,
                3,
                UncertainTea.NUTSKernelFlowDependency,
                UncertainTea.NUTSKernelBufferControlState,
                UncertainTea.NUTSKernelAliasControlState,
            ),
            UncertainTea.BatchedNUTSKernelDependency(
                2,
                4,
                UncertainTea.NUTSKernelFlowDependency,
                UncertainTea.NUTSKernelBufferSchedulerState,
                UncertainTea.NUTSKernelAliasSchedulerState,
            ),
        )
    merge_schedule = UncertainTea._batched_nuts_kernel_schedule(merge_program)
    @test length(UncertainTea._batched_nuts_kernel_schedule_stages(merge_schedule)) == 4
    merge_stage3 = UncertainTea._batched_nuts_kernel_stage_dataflow(
        UncertainTea._batched_nuts_kernel_schedule_stages(merge_schedule)[3],
    )
    @test UncertainTea._batched_nuts_kernel_step(merge_stage3) isa
        UncertainTea.BatchedNUTSMergeStep
    @test UncertainTea._batched_nuts_kernel_access(merge_stage3).proposal_position ===
        merge_access.proposal_position
    @test UncertainTea._batched_nuts_kernel_stage_dependencies(
        UncertainTea._batched_nuts_kernel_schedule_stages(merge_schedule)[3],
    ) == (
        UncertainTea.BatchedNUTSKernelDependency(
            2,
            3,
            UncertainTea.NUTSKernelFlowDependency,
            UncertainTea.NUTSKernelBufferControlState,
            UncertainTea.NUTSKernelAliasControlState,
        ),
    )
    merge_lifecycles = UncertainTea._batched_nuts_kernel_schedule_lifecycles(merge_schedule)
    @test merge_lifecycles[1] == UncertainTea.BatchedNUTSKernelBufferLifecycle(
        UncertainTea.NUTSKernelBufferControlBlock,
        UncertainTea.NUTSKernelAliasControlBlock,
        1,
        2,
        1,
        2,
        0,
        0,
    )
    merge_plan = UncertainTea._batched_nuts_kernel_resource_plan(merge_program)
    @test length(UncertainTea._batched_nuts_kernel_schedule_stages(merge_plan.schedule)) == 4
    @test UncertainTea._batched_nuts_kernel_resource_groups(merge_plan)[1] ==
        UncertainTea.BatchedNUTSKernelResourceGroup(
            UncertainTea.NUTSKernelAliasControlBlock,
            (UncertainTea.NUTSKernelBufferControlBlock,),
            1,
            2,
        )
    @test UncertainTea._batched_nuts_kernel_barriers(merge_plan) ==
        (
            UncertainTea.BatchedNUTSKernelBarrierPlacement(
                2,
                UncertainTea.NUTSKernelDependencyBarrier,
                (
                    UncertainTea.NUTSKernelAliasControlState,
                    UncertainTea.NUTSKernelAliasSchedulerState,
                ),
                (
                    UncertainTea.NUTSKernelBufferControlState,
                    UncertainTea.NUTSKernelBufferSchedulerState,
                ),
            ),
        )
    @test UncertainTea._batched_nuts_kernel_barriers_after(merge_plan, 2) ==
        UncertainTea._batched_nuts_kernel_barriers(merge_plan)
    merge_backend_block = UncertainTea._batched_nuts_backend_execution_block(merge_program)
    merge_control_binding = UncertainTea._batched_nuts_backend_buffer_binding(
        merge_backend_block,
        UncertainTea.NUTSKernelBufferControlBlock,
    )
    @test merge_control_binding ==
        UncertainTea.BatchedNUTSKernelBufferBinding(
            1,
            UncertainTea.NUTSKernelBufferControlBlock,
            UncertainTea.NUTSKernelAliasControlBlock,
            UncertainTea.NUTSKernelStorageUniform,
            1,
            merge_lifecycles[1],
        )
    merge_step_binding = UncertainTea._batched_nuts_backend_steps(merge_backend_block)[3]
    @test UncertainTea._batched_nuts_kernel_step(
        UncertainTea._batched_nuts_kernel_stage_dataflow(
            UncertainTea._batched_nuts_backend_stage(merge_step_binding),
        ),
    ) isa UncertainTea.BatchedNUTSMergeStep
    @test UncertainTea._batched_nuts_backend_step_read_buffers(
        merge_backend_block,
        merge_step_binding,
    ) == UncertainTea._batched_nuts_kernel_reads(merge_dataflows[3])
    @test UncertainTea._batched_nuts_backend_step_write_buffers(
        merge_backend_block,
        merge_step_binding,
    ) == UncertainTea._batched_nuts_kernel_writes(merge_dataflows[3])
    @test UncertainTea._batched_nuts_backend_barriers_after(merge_step_binding) == ()
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
        UncertainTea.NUTSSchedulerMerge
    @test UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_cohort_scheduler_workspace,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        cohort_rng,
    )
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
        UncertainTea.NUTSSchedulerDone
    @test gaussian_cohort_scheduler_workspace.control.scheduler.remaining_steps == 0
    @test gaussian_cohort_scheduler_workspace.control.scheduler.subtree_started ==
        BitVector([true, true, true])
    done_ir = UncertainTea._batched_nuts_control_ir(gaussian_cohort_scheduler_workspace)
    @test done_ir isa UncertainTea.BatchedNUTSDoneIR
    done_block = UncertainTea._batched_nuts_control_block(gaussian_cohort_scheduler_workspace)
    @test done_block isa UncertainTea.BatchedNUTSDoneControlBlock
    done_descriptor = UncertainTea._batched_nuts_step_descriptor(gaussian_cohort_scheduler_workspace)
    @test done_descriptor isa UncertainTea.BatchedNUTSDoneStepDescriptor
    done_state = UncertainTea._batched_nuts_step_state(gaussian_cohort_scheduler_workspace)
    @test done_state isa UncertainTea.BatchedNUTSDoneStepState
    done_frame = UncertainTea._batched_nuts_kernel_frame(gaussian_cohort_scheduler_workspace)
    @test done_frame isa UncertainTea.BatchedNUTSDoneKernelFrame
    done_access = UncertainTea._batched_nuts_kernel_access(
        gaussian_cohort_scheduler_workspace,
        done_frame,
    )
    @test done_access isa UncertainTea.BatchedNUTSDoneKernelAccess
    @test done_access.block === done_frame.state.descriptor.block
    done_program = UncertainTea._batched_nuts_kernel_program(
        gaussian_cohort_scheduler_workspace,
        done_access,
    )
    @test done_program isa UncertainTea.BatchedNUTSDoneKernelProgram
    @test UncertainTea._batched_nuts_kernel_access(done_program).block === done_access.block
    @test UncertainTea._batched_nuts_kernel_ops(done_program) ==
        (UncertainTea.NUTSKernelReloadControl,)
    @test typeof.(UncertainTea._batched_nuts_kernel_steps(done_program)) ==
        (UncertainTea.BatchedNUTSReloadControlStep,)
    done_dataflows = UncertainTea._batched_nuts_kernel_dataflows(done_program)
    @test length(done_dataflows) == 1
    @test UncertainTea._batched_nuts_kernel_reads(done_dataflows[1]) ==
        (UncertainTea.NUTSKernelBufferControlBlock,)
    @test UncertainTea._batched_nuts_kernel_read_aliases(done_dataflows[1]) ==
        (UncertainTea.NUTSKernelAliasControlBlock,)
    @test UncertainTea._batched_nuts_kernel_writes(done_dataflows[1]) ==
        (
            UncertainTea.NUTSKernelBufferSchedulerState,
            UncertainTea.NUTSKernelBufferControlState,
        )
    @test isempty(UncertainTea._batched_nuts_kernel_dependencies(done_program))
    done_schedule = UncertainTea._batched_nuts_kernel_schedule(done_program)
    @test length(UncertainTea._batched_nuts_kernel_schedule_stages(done_schedule)) == 1
    @test isempty(UncertainTea._batched_nuts_kernel_schedule_lifecycles(done_schedule)) == false
    done_plan = UncertainTea._batched_nuts_kernel_resource_plan(done_program)
    @test isempty(UncertainTea._batched_nuts_kernel_barriers(done_plan))
    @test UncertainTea._batched_nuts_kernel_resource_groups(done_plan)[1] ==
        UncertainTea.BatchedNUTSKernelResourceGroup(
            UncertainTea.NUTSKernelAliasControlBlock,
            (UncertainTea.NUTSKernelBufferControlBlock,),
            1,
            1,
        )
    done_backend_block = UncertainTea._batched_nuts_backend_execution_block(done_program)
    @test UncertainTea._batched_nuts_backend_buffer_binding(
        done_backend_block,
        UncertainTea.NUTSKernelBufferControlState,
    ).storage_class == UncertainTea.NUTSKernelStoragePersistent
    @test isempty(
        UncertainTea._batched_nuts_backend_barriers_after(
            UncertainTea._batched_nuts_backend_steps(done_backend_block)[1],
        ),
    )
    @test !UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_cohort_scheduler_workspace,
        done_program,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        cohort_rng,
    )
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
        UncertainTea.NUTSSchedulerDone
    @test gaussian_cohort_scheduler_workspace.control.tree_depths == [2, 2, 2]
    @test gaussian_cohort_scheduler_workspace.subtree_active ==
        BitVector([true, true, true])
    @test all(isfinite, gaussian_cohort_scheduler_workspace.continuation_log_weight)
    @test all(isfinite, gaussian_cohort_scheduler_workspace.proposed_logjoint)
    gaussian_expand_ir_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_expand_ir_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(105),
    )
    expand_ir_rng = MersenneTwister(106)
    @test UncertainTea._begin_batched_nuts_subtree_scheduler!(
        gaussian_expand_ir_workspace,
        4,
        expand_ir_rng,
    )
    expand_direct_ir = UncertainTea._batched_nuts_control_ir(gaussian_expand_ir_workspace)
    @test expand_direct_ir isa UncertainTea.BatchedNUTSExpandIR
    expand_direct_block = UncertainTea._batched_nuts_control_block(expand_direct_ir)
    @test expand_direct_block isa UncertainTea.BatchedNUTSExpandControlBlock
    expand_direct_descriptor = UncertainTea._batched_nuts_step_descriptor(
        gaussian_expand_ir_workspace,
        expand_direct_block,
    )
    @test expand_direct_descriptor isa UncertainTea.BatchedNUTSExpandStepDescriptor
    expand_direct_state = UncertainTea._batched_nuts_step_state(
        gaussian_expand_ir_workspace,
        expand_direct_descriptor,
    )
    @test expand_direct_state isa UncertainTea.BatchedNUTSExpandStepState
    expand_direct_frame = UncertainTea._batched_nuts_kernel_frame(
        gaussian_expand_ir_workspace,
        expand_direct_state,
    )
    @test expand_direct_frame isa UncertainTea.BatchedNUTSExpandKernelFrame
    expand_direct_access = UncertainTea._batched_nuts_kernel_access(
        gaussian_expand_ir_workspace,
        expand_direct_frame,
    )
    @test expand_direct_access isa UncertainTea.BatchedNUTSExpandKernelAccess
    @test expand_direct_access.block === expand_direct_frame.state.descriptor.block
    @test expand_direct_access.copy_left === gaussian_expand_ir_workspace.subtree_copy_left
    @test expand_direct_access.copy_right === gaussian_expand_ir_workspace.subtree_copy_right
    @test expand_direct_access.select_proposal ===
        gaussian_expand_ir_workspace.subtree_select_proposal
    @test expand_direct_access.current_position ===
        gaussian_expand_ir_workspace.tree_current_position
    @test expand_direct_access.next_position === gaussian_expand_ir_workspace.tree_next_position
    @test expand_direct_access.proposed_energy ===
        gaussian_expand_ir_workspace.subtree_proposed_energy
    expand_direct_program = UncertainTea._batched_nuts_kernel_program(
        gaussian_expand_ir_workspace,
        expand_direct_access,
    )
    @test expand_direct_program isa UncertainTea.BatchedNUTSExpandKernelProgram
    @test UncertainTea._batched_nuts_kernel_access(expand_direct_program).next_position ===
        expand_direct_access.next_position
    expand_dataflows = UncertainTea._batched_nuts_kernel_dataflows(expand_direct_program)
    @test length(expand_dataflows) == 5
    @test UncertainTea._batched_nuts_kernel_step(expand_dataflows[2]) isa
        UncertainTea.BatchedNUTSLeapfrogStep
    @test UncertainTea._batched_nuts_kernel_access(expand_dataflows[2]).next_position ===
        expand_direct_access.next_position
    @test UncertainTea._batched_nuts_kernel_reads(expand_dataflows[2]) ==
        (
            UncertainTea.NUTSKernelBufferControlState,
            UncertainTea.NUTSKernelBufferTreeCurrentState,
        )
    @test UncertainTea._batched_nuts_kernel_read_aliases(expand_dataflows[2]) ==
        (
            UncertainTea.NUTSKernelAliasControlState,
            UncertainTea.NUTSKernelAliasTreeState,
        )
    @test UncertainTea._batched_nuts_kernel_writes(expand_dataflows[2]) ==
        (
            UncertainTea.NUTSKernelBufferControlState,
            UncertainTea.NUTSKernelBufferTreeNextState,
        )
    @test UncertainTea._batched_nuts_kernel_reads(expand_dataflows[4]) ==
        (
            UncertainTea.NUTSKernelBufferControlState,
            UncertainTea.NUTSKernelBufferDescriptorScratch,
            UncertainTea.NUTSKernelBufferTreeCurrentState,
            UncertainTea.NUTSKernelBufferTreeNextState,
            UncertainTea.NUTSKernelBufferTreeEnergy,
            UncertainTea.NUTSKernelBufferSubtreeSummary,
        )
    @test UncertainTea._batched_nuts_kernel_writes(expand_dataflows[4]) ==
        (
            UncertainTea.NUTSKernelBufferControlState,
            UncertainTea.NUTSKernelBufferDescriptorScratch,
            UncertainTea.NUTSKernelBufferTreeCurrentState,
            UncertainTea.NUTSKernelBufferTreeFrontierState,
            UncertainTea.NUTSKernelBufferTreeProposalState,
            UncertainTea.NUTSKernelBufferSubtreeSummary,
        )
    @test UncertainTea._batched_nuts_kernel_dependencies(expand_direct_program) ==
        (
            UncertainTea.BatchedNUTSKernelDependency(
                1,
                2,
                UncertainTea.NUTSKernelFlowDependency,
                UncertainTea.NUTSKernelBufferControlState,
                UncertainTea.NUTSKernelAliasControlState,
            ),
            UncertainTea.BatchedNUTSKernelDependency(
                1,
                5,
                UncertainTea.NUTSKernelFlowDependency,
                UncertainTea.NUTSKernelBufferSchedulerState,
                UncertainTea.NUTSKernelAliasSchedulerState,
            ),
            UncertainTea.BatchedNUTSKernelDependency(
                2,
                3,
                UncertainTea.NUTSKernelFlowDependency,
                UncertainTea.NUTSKernelBufferTreeNextState,
                UncertainTea.NUTSKernelAliasTreeState,
            ),
            UncertainTea.BatchedNUTSKernelDependency(
                2,
                4,
                UncertainTea.NUTSKernelFlowDependency,
                UncertainTea.NUTSKernelBufferControlState,
                UncertainTea.NUTSKernelAliasControlState,
            ),
            UncertainTea.BatchedNUTSKernelDependency(
                2,
                4,
                UncertainTea.NUTSKernelFlowDependency,
                UncertainTea.NUTSKernelBufferTreeNextState,
                UncertainTea.NUTSKernelAliasTreeState,
            ),
            UncertainTea.BatchedNUTSKernelDependency(
                3,
                4,
                UncertainTea.NUTSKernelFlowDependency,
                UncertainTea.NUTSKernelBufferTreeEnergy,
                UncertainTea.NUTSKernelAliasTreeEnergy,
            ),
        )
    expand_schedule = UncertainTea._batched_nuts_kernel_schedule(expand_direct_program)
    @test length(UncertainTea._batched_nuts_kernel_schedule_stages(expand_schedule)) == 5
    expand_stage2 = UncertainTea._batched_nuts_kernel_stage_dataflow(
        UncertainTea._batched_nuts_kernel_schedule_stages(expand_schedule)[2],
    )
    @test UncertainTea._batched_nuts_kernel_step(expand_stage2) isa
        UncertainTea.BatchedNUTSLeapfrogStep
    @test UncertainTea._batched_nuts_kernel_access(expand_stage2).next_position ===
        expand_direct_access.next_position
    @test UncertainTea._batched_nuts_kernel_stage_dependencies(
        UncertainTea._batched_nuts_kernel_schedule_stages(expand_schedule)[4],
    ) == (
        UncertainTea.BatchedNUTSKernelDependency(
            2,
            4,
            UncertainTea.NUTSKernelFlowDependency,
            UncertainTea.NUTSKernelBufferControlState,
            UncertainTea.NUTSKernelAliasControlState,
        ),
        UncertainTea.BatchedNUTSKernelDependency(
            2,
            4,
            UncertainTea.NUTSKernelFlowDependency,
            UncertainTea.NUTSKernelBufferTreeNextState,
            UncertainTea.NUTSKernelAliasTreeState,
        ),
        UncertainTea.BatchedNUTSKernelDependency(
            3,
            4,
            UncertainTea.NUTSKernelFlowDependency,
            UncertainTea.NUTSKernelBufferTreeEnergy,
            UncertainTea.NUTSKernelAliasTreeEnergy,
        ),
    )
    expand_lifecycles = UncertainTea._batched_nuts_kernel_schedule_lifecycles(expand_schedule)
    @test expand_lifecycles[1] == UncertainTea.BatchedNUTSKernelBufferLifecycle(
        UncertainTea.NUTSKernelBufferControlBlock,
        UncertainTea.NUTSKernelAliasControlBlock,
        1,
        1,
        1,
        1,
        0,
        0,
    )
    @test any(
        lifecycle ->
            lifecycle.buffer == UncertainTea.NUTSKernelBufferTreeEnergy &&
            lifecycle.first_stage == 3 &&
            lifecycle.last_stage == 4 &&
            lifecycle.first_read_stage == 4 &&
            lifecycle.last_read_stage == 4 &&
            lifecycle.first_write_stage == 3 &&
            lifecycle.last_write_stage == 3,
        expand_lifecycles,
    )
    expand_plan = UncertainTea._batched_nuts_kernel_resource_plan(expand_direct_program)
    @test UncertainTea._batched_nuts_kernel_resource_groups(expand_plan)[1] ==
        UncertainTea.BatchedNUTSKernelResourceGroup(
            UncertainTea.NUTSKernelAliasControlBlock,
            (UncertainTea.NUTSKernelBufferControlBlock,),
            1,
            1,
        )
    @test UncertainTea._batched_nuts_kernel_barriers(expand_plan) ==
        (
            UncertainTea.BatchedNUTSKernelBarrierPlacement(
                1,
                UncertainTea.NUTSKernelDependencyBarrier,
                (
                    UncertainTea.NUTSKernelAliasControlState,
                    UncertainTea.NUTSKernelAliasSchedulerState,
                ),
                (
                    UncertainTea.NUTSKernelBufferControlState,
                    UncertainTea.NUTSKernelBufferSchedulerState,
                ),
            ),
            UncertainTea.BatchedNUTSKernelBarrierPlacement(
                2,
                UncertainTea.NUTSKernelDependencyBarrier,
                (
                    UncertainTea.NUTSKernelAliasTreeState,
                    UncertainTea.NUTSKernelAliasControlState,
                ),
                (
                    UncertainTea.NUTSKernelBufferTreeNextState,
                    UncertainTea.NUTSKernelBufferControlState,
                ),
            ),
            UncertainTea.BatchedNUTSKernelBarrierPlacement(
                3,
                UncertainTea.NUTSKernelDependencyBarrier,
                (UncertainTea.NUTSKernelAliasTreeEnergy,),
                (UncertainTea.NUTSKernelBufferTreeEnergy,),
            ),
        )
    @test UncertainTea._batched_nuts_kernel_barriers_after(expand_plan, 2) ==
        (
            UncertainTea.BatchedNUTSKernelBarrierPlacement(
                2,
                UncertainTea.NUTSKernelDependencyBarrier,
                (
                    UncertainTea.NUTSKernelAliasTreeState,
                    UncertainTea.NUTSKernelAliasControlState,
                ),
                (
                    UncertainTea.NUTSKernelBufferTreeNextState,
                    UncertainTea.NUTSKernelBufferControlState,
                ),
            ),
        )
    expand_backend_block = UncertainTea._batched_nuts_backend_execution_block(expand_direct_program)
    expand_next_binding = UncertainTea._batched_nuts_backend_buffer_binding(
        expand_backend_block,
        UncertainTea.NUTSKernelBufferTreeNextState,
    )
    @test expand_next_binding.storage_class == UncertainTea.NUTSKernelStorageScratch
    @test expand_next_binding.resource_group ==
        UncertainTea._batched_nuts_backend_buffer_binding(
            expand_backend_block,
            UncertainTea.NUTSKernelBufferTreeCurrentState,
        ).resource_group
    expand_step_binding = UncertainTea._batched_nuts_backend_steps(expand_backend_block)[2]
    @test UncertainTea._batched_nuts_backend_step_read_buffers(
        expand_backend_block,
        expand_step_binding,
    ) == UncertainTea._batched_nuts_kernel_reads(expand_dataflows[2])
    @test UncertainTea._batched_nuts_backend_step_write_buffers(
        expand_backend_block,
        expand_step_binding,
    ) == UncertainTea._batched_nuts_kernel_writes(expand_dataflows[2])
    @test UncertainTea._batched_nuts_backend_barriers_after(expand_step_binding) ==
        (
            UncertainTea.BatchedNUTSKernelBarrierPlacement(
                2,
                UncertainTea.NUTSKernelDependencyBarrier,
                (
                    UncertainTea.NUTSKernelAliasTreeState,
                    UncertainTea.NUTSKernelAliasControlState,
                ),
                (
                    UncertainTea.NUTSKernelBufferTreeNextState,
                    UncertainTea.NUTSKernelBufferControlState,
                ),
            ),
        )
    fill!(gaussian_expand_ir_workspace.subtree_active, false)
    fill!(gaussian_expand_ir_workspace.control.step_direction, 0)
    fill!(gaussian_expand_ir_workspace.subtree_integration_steps, 0)
    @test UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_expand_ir_workspace,
        expand_direct_access,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        expand_ir_rng,
    )
    @test gaussian_expand_ir_workspace.control.step_direction == expand_direct_ir.step_direction
    @test gaussian_expand_ir_workspace.control.scheduler.remaining_steps ==
        expand_direct_ir.remaining_steps - 1
    @test sum(gaussian_expand_ir_workspace.subtree_integration_steps) > 0
    gaussian_merge_ir_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_merge_ir_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(107),
    )
    merge_ir_rng = MersenneTwister(108)
    @test UncertainTea._begin_batched_nuts_subtree_scheduler!(
        gaussian_merge_ir_workspace,
        4,
        merge_ir_rng,
    )
    while gaussian_merge_ir_workspace.control.scheduler.phase ==
        UncertainTea.NUTSSchedulerExpand
        @test UncertainTea._step_batched_nuts_subtree_scheduler!(
            gaussian_merge_ir_workspace,
            gaussian_mean,
            [1.0],
            (),
            choicemap((:y, 0.4)),
            0.01,
            1000.0,
            merge_ir_rng,
        )
    end
    merge_direct_ir = UncertainTea._batched_nuts_control_ir(gaussian_merge_ir_workspace)
    @test merge_direct_ir isa UncertainTea.BatchedNUTSMergeIR
    merge_direct_block = UncertainTea._batched_nuts_control_block(merge_direct_ir)
    @test merge_direct_block isa UncertainTea.BatchedNUTSMergeControlBlock
    merge_direct_descriptor = UncertainTea._batched_nuts_step_descriptor(
        gaussian_merge_ir_workspace,
        merge_direct_block,
    )
    @test merge_direct_descriptor isa UncertainTea.BatchedNUTSMergeStepDescriptor
    merge_direct_state = UncertainTea._batched_nuts_step_state(
        gaussian_merge_ir_workspace,
        merge_direct_descriptor,
    )
    @test merge_direct_state isa UncertainTea.BatchedNUTSMergeStepState
    merge_direct_frame = UncertainTea._batched_nuts_kernel_frame(
        gaussian_merge_ir_workspace,
        merge_direct_state,
    )
    @test merge_direct_frame isa UncertainTea.BatchedNUTSMergeKernelFrame
    merge_direct_access = UncertainTea._batched_nuts_kernel_access(
        gaussian_merge_ir_workspace,
        merge_direct_frame,
    )
    @test merge_direct_access isa UncertainTea.BatchedNUTSMergeKernelAccess
    @test merge_direct_access.block === merge_direct_frame.state.descriptor.block
    @test merge_direct_access.select_proposal ===
        gaussian_merge_ir_workspace.continuation_select_proposal
    @test merge_direct_access.left_position === gaussian_merge_ir_workspace.left_position
    @test merge_direct_access.tree_proposal_position ===
        gaussian_merge_ir_workspace.tree_proposal_position
    merge_direct_program = UncertainTea._batched_nuts_kernel_program(
        gaussian_merge_ir_workspace,
        merge_direct_access,
    )
    @test merge_direct_program isa UncertainTea.BatchedNUTSMergeKernelProgram
    @test UncertainTea._batched_nuts_kernel_access(merge_direct_program).proposal_position ===
        merge_direct_access.proposal_position
    merge_tree_depths = copy(gaussian_merge_ir_workspace.control.tree_depths)
    fill!(gaussian_merge_ir_workspace.control.scheduler.subtree_started, false)
    fill!(gaussian_merge_ir_workspace.subtree_active, false)
    @test UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_merge_ir_workspace,
        merge_direct_access,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        merge_ir_rng,
    )
    @test gaussian_merge_ir_workspace.control.scheduler.subtree_started ==
        merge_direct_ir.started_chains
    @test gaussian_merge_ir_workspace.subtree_active == merge_direct_ir.merge_active
    @test gaussian_merge_ir_workspace.control.tree_depths ==
        merge_tree_depths .+ Int.(merge_direct_ir.started_chains)
    @test gaussian_merge_ir_workspace.control.scheduler.phase ==
        UncertainTea.NUTSSchedulerDone
    gaussian_finalized_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._batched_nuts_proposals!(
        gaussian_finalized_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.05,
        3,
        1000.0,
        MersenneTwister(103),
    )
    @test gaussian_finalized_nuts_workspace.proposed_logjoint ≈
        gaussian_finalized_nuts_workspace.continuation_proposal_logjoint atol=1e-8
    @test gaussian_finalized_nuts_workspace.proposed_energy ≈
        gaussian_finalized_nuts_workspace.continuation_proposed_energy atol=1e-8
    @test gaussian_finalized_nuts_workspace.accept_prob ≈
        UncertainTea._mean_acceptance_stats!(
            similar(gaussian_finalized_nuts_workspace.accept_prob),
            gaussian_finalized_nuts_workspace.continuation_accept_stat_sum,
            gaussian_finalized_nuts_workspace.continuation_accept_stat_count,
        ) atol=1e-8
    @test gaussian_finalized_nuts_workspace.energy_error ≈
        gaussian_finalized_nuts_workspace.continuation_delta_energy atol=1e-8
    @test gaussian_backend_report.supported
    @test gaussian_backend_report.target == :gpu
    @test isempty(gaussian_backend_report.issues)
    @test gaussian_backend_plan.target == :gpu
    @test length(gaussian_backend_plan.steps) == 2
    @test gaussian_backend_plan.steps[1] isa UncertainTea.BackendChoicePlanStep
    @test gaussian_backend_plan.steps[1] isa UncertainTea.BackendNormalChoicePlanStep
    @test gaussian_backend_plan.numeric_slots == BitVector([true])
    @test gaussian_backend_plan.index_slots == BitVector([false])
    @test gaussian_backend_plan.generic_slots == BitVector([false])
    @test iid_backend_report.supported
    @test count(identity, iid_backend_plan.numeric_slots) == 1
    @test count(identity, iid_backend_plan.index_slots) == 2
    @test !any(iid_backend_plan.generic_slots)
    @test iid_backend_plan.index_slots[1]
    @test iid_backend_plan.numeric_slots[2]
    @test iid_backend_plan.index_slots[3]
    @test shifted_backend_report.supported
    @test shifted_backend_plan.numeric_slots == iid_backend_plan.numeric_slots
    @test shifted_backend_plan.index_slots == iid_backend_plan.index_slots
    @test offset_backend_report.supported
    @test indexed_scale_backend_report.supported
    @test indexed_scale_backend_plan.numeric_slots[indexed_scale_plan.environment_layout.slot_by_symbol[:mu]]
    @test indexed_scale_backend_plan.index_slots[indexed_scale_plan.environment_layout.slot_by_symbol[:n]]
    @test indexed_scale_backend_plan.index_slots[indexed_scale_plan.environment_layout.slot_by_symbol[:i]]
    @test !any(indexed_scale_backend_plan.generic_slots)
    @test !isnothing(UncertainTea._backend_loop_observed_choice(iid_backend_plan.steps[2]))
    @test !isnothing(UncertainTea._backend_loop_observed_choice(shifted_backend_plan.steps[2]))
    @test isnothing(UncertainTea._backend_loop_observed_choice(offset_backend_plan.steps[2]))
    @test isnothing(UncertainTea._backend_loop_observed_choice(backend_execution_plan(chain_model).steps[3]))
    @test coin_backend_report.supported
    @test coin_backend_plan.steps[1] isa UncertainTea.BackendBernoulliChoicePlanStep
    @test isempty(coin_backend_plan.numeric_slots)
    @test isempty(coin_backend_plan.index_slots)
    @test isempty(coin_backend_plan.generic_slots)
    @test coin_batch_logjoint ≈ [
        logjoint(observed_coin, Float64[], (), choicemap((:y, true))),
        logjoint(observed_coin, Float64[], (), choicemap((:y, false))),
        logjoint(observed_coin, Float64[], (), choicemap((:y, true))),
    ] atol=2e-8
    coin_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        coin_workspace,
        zeros(0, 3),
        (),
        [
            choicemap((:y, true)),
            choicemap((:y, false)),
            choicemap((:y, true)),
        ],
    )
    coin_workspace_env = coin_workspace.batched_environment[]
    coin_workspace_observed = coin_workspace_env.observed_values
    @test coin_workspace_values ≈ coin_batch_logjoint atol=2e-8
    @test UncertainTea._logjoint_with_batched_backend!(
        coin_workspace,
        zeros(0, 3),
        (),
        choicemap((:y, true)),
    ) ≈ fill(logjoint(observed_coin, Float64[], (), choicemap((:y, true))), 3) atol=2e-8
    @test coin_workspace.batched_environment[] === coin_workspace_env
    @test coin_workspace_env.observed_values === coin_workspace_observed
    @test deterministic_backend_report.supported
    @test length(deterministic_backend_plan.steps) == 4
    @test deterministic_backend_plan.steps[3] isa UncertainTea.BackendDeterministicPlanStep
    @test all(deterministic_backend_plan.numeric_slots)
    @test !any(deterministic_backend_plan.index_slots)
    @test !any(deterministic_backend_plan.generic_slots)
    @test !unsupported_backend_report.supported
    @test any(occursin("sin", issue) for issue in unsupported_backend_report.issues)
    @test unsupported_backend_logjoint ≈ [
        logjoint(
            unsupported_backend_model,
            unsupported_backend_params[:, index],
            (),
            unsupported_backend_constraints[index],
        ) for index in 1:2
    ] atol=1e-8
    @test short_warmup_schedule.initial_buffer == 5
    @test short_warmup_schedule.slow_window_ends == [12]
    @test short_warmup_schedule.terminal_buffer == 0
    @test long_warmup_schedule.initial_buffer == 15
    @test long_warmup_schedule.slow_window_ends == [40, 90, 135]
    @test long_warmup_schedule.terminal_buffer == 15
    @test UncertainTea._warmup_window_length(short_warmup_schedule, 1) == 7
    @test UncertainTea._warmup_window_length(long_warmup_schedule, 1) == 25
    @test UncertainTea._warmup_window_length(long_warmup_schedule, 2) == 50
    @test UncertainTea._warmup_window_length(long_warmup_schedule, 3) == 45
    @test UncertainTea._mean_batched_adaptation_probability([0.8, 0.6, 0.4], falses(3)) ≈ 0.6 atol=1e-8
    @test UncertainTea._mean_batched_adaptation_probability([0.8, 0.6, 0.4], BitVector([false, true, false])) ≈
        0.4 atol=1e-8
    early_window_state = UncertainTea._running_variance_state(1, 24)
    late_window_state = UncertainTea._running_variance_state(1, 24)
    late_window_state.count = 24
    @test UncertainTea._mass_adaptation_weight(early_window_state, true, 0.1, false) == 1.0
    @test UncertainTea._mass_adaptation_weight(early_window_state, false, 0.25, false) == 1.0
    @test UncertainTea._mass_adaptation_weight(late_window_state, false, 0.25, false) == 0.25
    @test UncertainTea._mass_adaptation_weight(early_window_state, false, 0.25, true) == 0.0
    mass_adaptation_weights = zeros(3)
    UncertainTea._mass_adaptation_weights!(
        late_window_state,
        mass_adaptation_weights,
        BitVector([true, false, false]),
        [0.2, 0.3, 0.4],
        BitVector([false, false, true]),
    )
    @test mass_adaptation_weights ≈ [1.0, 0.3, 0.0] atol=1e-8
    @test UncertainTea._running_variance_clip_scale(early_window_state) == 8.0
    early_window_state.count = 4
    @test UncertainTea._running_variance_clip_scale(early_window_state) == 8.0
    early_window_state.count = 14
    @test UncertainTea._running_variance_clip_scale(early_window_state) ≈ 6.5 atol=1e-8
    @test UncertainTea._running_variance_window_progress(early_window_state) ≈ 0.5 atol=1e-8
    @test UncertainTea._running_variance_clip_scale(late_window_state) == 5.0
    @test UncertainTea._running_variance_window_progress(late_window_state) == 1.0
    masked_variance_state = UncertainTea._running_variance_state(2)
    UncertainTea._update_running_variance!(
        masked_variance_state,
        [1.0 2.0 5.0; 10.0 20.0 50.0],
        BitVector([true, false, true]),
    )
    @test masked_variance_state.count == 2
    @test masked_variance_state.mean ≈ [3.0, 30.0] atol=1e-8
    @test masked_variance_state.m2 ≈ [8.0, 800.0] atol=1e-8
    @test masked_variance_state.weight_sum == 2.0
    @test masked_variance_state.weight_square_sum == 2.0
    @test UncertainTea._running_variance_effective_count(masked_variance_state) == 2.0
    weighted_variance_state = UncertainTea._running_variance_state(1)
    UncertainTea._update_running_variance!(weighted_variance_state, [0.0], 1.0)
    UncertainTea._update_running_variance!(weighted_variance_state, [10.0], 0.25)
    @test weighted_variance_state.count == 2
    @test weighted_variance_state.weight_sum ≈ 1.25 atol=1e-8
    @test weighted_variance_state.weight_square_sum ≈ 1.0625 atol=1e-8
    @test UncertainTea._running_variance_effective_count(weighted_variance_state) ≈
        (1.25^2 / 1.0625) atol=1e-8
    robust_variance_state = UncertainTea._running_variance_state(1)
    for value in (0.0, 0.05, -0.05, 0.1, 100.0)
        UncertainTea._update_running_variance!(robust_variance_state, [value])
    end
    @test robust_variance_state.count == 5
    @test robust_variance_state.mean[1] < 1.0
    @test robust_variance_state.m2[1] < 1.0
    @test UncertainTea._inverse_mass_matrix(robust_variance_state, 1e-3)[1] > 1.0
