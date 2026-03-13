@testset "Tempered SMC NUTS GPU Package" begin
    @tea static function tempered_smc_emitter_model()
        weights ~ dirichlet([2.0f0, 3.0f0, 4.0f0])
        return weights
    end

    particles = randn(MersenneTwister(270), 2, 6)
    workspace = UncertainTea.TemperedNUTSMoveWorkspace(
        tempered_smc_emitter_model,
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
            1,
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
        MersenneTwister(271),
    ) == UncertainTea.TemperedNUTSSchedulerExpand

    bundle = tempered_smc_nuts_codegen_bundle(tempered_smc_emitter_model, workspace)
    @test bundle.target == :gpu
    @test bundle.bundle_symbol == :UncertainTeaGPUTemperedSMCNUTSBundle__tempered_smc_emitter_model
    @test length(gpu_backend_codegen_stages(bundle)) == 4
    @test gpu_backend_stage_kind(gpu_backend_codegen_stages(bundle)[2]) == :smc_tempered_nuts_expand
    @test occursin("current_phase = \"TemperedNUTSSchedulerExpand\"", join(gpu_backend_codegen_manifest_lines(bundle), "\n"))

    gpu_layout = tempered_smc_nuts_package_layout(tempered_smc_emitter_model, workspace; target=:gpu)
    @test gpu_layout.target == :gpu
    @test gpu_layout.backend_symbol == :UncertainTeaGPUTemperedSMCNUTSPackage__tempered_smc_emitter_model
    @test length(gpu_backend_bundles(gpu_layout)) == 1
    @test length(gpu_backend_files(gpu_layout)) == 5
    @test occursin("kind = \"smc_tempered_nuts_expand\"", gpu_backend_manifest_file(gpu_backend_bundles(gpu_layout)[1]).contents)
    @test any(occursin("const ACTIVE_DEPTH = 2", file.contents) for file in gpu_backend_files(gpu_layout))
    @test any(occursin("const STEP_DIRECTIONS = [", file.contents) for file in gpu_backend_files(gpu_layout))

    metal_layout = tempered_smc_nuts_package_layout(tempered_smc_emitter_model, workspace; target=:metal)
    cuda_layout = tempered_smc_nuts_package_layout(tempered_smc_emitter_model, workspace; target=:cuda)
    @test endswith(gpu_backend_files(metal_layout)[2].relative_path, ".metal")
    @test endswith(gpu_backend_files(cuda_layout)[2].relative_path, ".cu")

    idle_layout = tempered_smc_nuts_package_layout(
        tempered_smc_emitter_model,
        particles,
        (),
        choicemap();
        target=:gpu,
    )
    @test occursin("current_phase = \"TemperedNUTSSchedulerIdle\"", gpu_backend_manifest_file(gpu_backend_bundles(idle_layout)[1]).contents)

    mktempdir() do temp_dir
        emission = emit_tempered_smc_nuts_package(
            tempered_smc_emitter_model,
            workspace,
            temp_dir;
            target=:gpu,
        )
        @test emission.package.target == :gpu
        @test length(emission.written_files) == 5
        @test all(isfile, emission.written_files)
        @test_throws ArgumentError emit_tempered_smc_nuts_package(
            tempered_smc_emitter_model,
            workspace,
            temp_dir;
            target=:gpu,
        )
        overwrite_emission = emit_tempered_smc_nuts_package(
            tempered_smc_emitter_model,
            workspace,
            temp_dir;
            target=:gpu,
            overwrite=true,
        )
        @test overwrite_emission.written_files == emission.written_files
    end
end
