    codegen_bundle = gpu_backend_codegen_bundle(
        :gpu,
        :ManualCodegenBundle,
        (
            GPUBackendCodegenStage(
                :manual_stage,
                :manual_entry,
                "manual_stage.jl",
                "module ManualStage\nend",
            ),
        );
        manifest_lines=("model = \"manual\"",),
    )
    @test codegen_bundle.target == :gpu
    @test codegen_bundle.bundle_symbol == :ManualCodegenBundle
    @test gpu_backend_codegen_manifest_lines(codegen_bundle) == ("model = \"manual\"",)
    @test length(gpu_backend_codegen_stages(codegen_bundle)) == 1
    @test gpu_backend_codegen_stages(codegen_bundle)[1].entry_symbol == :manual_entry
    @test gpu_backend_stage_kind(gpu_backend_codegen_stages(codegen_bundle)[1]) ==
        :manual_stage

    codegen_layout = gpu_backend_codegen_package_layout(
        :gpu,
        :ManualCodegenPackage,
        "manualcodegenpackage",
        (codegen_bundle,),
    )
    @test codegen_layout.backend_symbol == :ManualCodegenPackage
    @test length(gpu_backend_bundles(codegen_layout)) == 1
    @test gpu_backend_manifest_file(gpu_backend_bundles(codegen_layout)[1]).relative_path ==
        "manualcodegenpackage/manualcodegenbundle__manifest.toml"
    @test occursin(
        "entry = \"manual_entry\"",
        gpu_backend_manifest_file(gpu_backend_bundles(codegen_layout)[1]).contents,
    )
    @test occursin(
        "kind = \"manual_stage\"",
        gpu_backend_manifest_file(gpu_backend_bundles(codegen_layout)[1]).contents,
    )
    @test gpu_backend_stage_files(gpu_backend_bundles(codegen_layout)[1])[1].relative_path ==
        "manualcodegenpackage/manual_stage.jl"

    gaussian_backend_bundle = gpu_backend_bundles(backend_package_layout(gaussian_mean))[1]
    done_nuts_bundle = gpu_backend_bundles(batched_nuts_package_layout(done_program))[1]
    @test gaussian_backend_bundle isa GPUBackendBundleLayout
    @test done_nuts_bundle isa GPUBackendBundleLayout
    @test occursin("kind = \"backend_execute\"", gpu_backend_manifest_file(gaussian_backend_bundle).contents)
    @test occursin("kind = \"nuts_reload_control\"", gpu_backend_manifest_file(done_nuts_bundle).contents)
    @test occursin("entry = \"execute_backend__gaussian_mean\"", gpu_backend_manifest_file(gaussian_backend_bundle).contents)
    @test occursin("entry = \"", gpu_backend_manifest_file(done_nuts_bundle).contents)
