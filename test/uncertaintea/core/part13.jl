    done_layout = batched_nuts_package_layout(done_program; target=:gpu)
    done_package_plan = UncertainTea._batched_nuts_package_plan(done_program; target=:gpu)
    done_layout_files = gpu_backend_files(done_layout)
    done_bundles = gpu_backend_bundles(done_layout)
    @test done_layout.target == :gpu
    @test done_layout.backend_symbol == :UncertainTeaCPUPackage
    @test done_layout.root_dir == "uncertainteacpupackage"
    @test length(done_bundles) == 1
    @test done_bundles[1].target == :gpu
    @test done_bundles[1].bundle_symbol == :UncertainTeaCPUBundle
    @test gpu_backend_manifest_file(done_bundles[1]).relative_path ==
        "uncertainteacpupackage/uncertainteacpubundle__manifest.toml"
    @test length(gpu_backend_stage_files(done_bundles[1])) == 1
    @test gpu_backend_stage_files(done_bundles[1])[1].stage_name ==
        UncertainTea._batched_nuts_module_entry_symbol(
            UncertainTea._batched_nuts_bundle_module_stage(
                UncertainTea._batched_nuts_package_bundle_stage(
                    UncertainTea._batched_nuts_package_stage_files(done_package_plan)[1],
                ),
            ),
        )
    @test length(done_layout_files) == 2
    @test done_layout_files[1].relative_path ==
        "uncertainteacpupackage/uncertainteacpubundle__manifest.toml"
    @test occursin("count = 1", done_layout_files[1].contents)
    @test endswith(done_layout_files[2].relative_path, ".jl")
    @test done_layout_files[2].contents == UncertainTea._batched_nuts_package_contents(
        UncertainTea._batched_nuts_package_stage_files(
            done_package_plan,
        )[1],
    )

    expand_metal_layout = batched_nuts_package_layout(
        expand_direct_program;
        target=:metal,
    )
    expand_metal_plan = UncertainTea._batched_nuts_package_plan(
        expand_direct_program;
        target=:metal,
    )
    @test expand_metal_layout.target == :metal
    @test expand_metal_layout.backend_symbol == :UncertainTeaMetalPackage
    @test length(gpu_backend_files(expand_metal_layout)) ==
        1 + length(UncertainTea._batched_nuts_package_stage_files(expand_metal_plan))
    @test endswith(gpu_backend_files(expand_metal_layout)[2].relative_path, ".metal")

    merge_cuda_layout = batched_nuts_package_layout(merge_program; target=:cuda)
    @test merge_cuda_layout.target == :cuda
    @test merge_cuda_layout.backend_symbol == :UncertainTeaCUDAPackage
    @test length(gpu_backend_files(merge_cuda_layout)) == 5
    @test endswith(gpu_backend_files(merge_cuda_layout)[4].relative_path, ".cu")

    manual_bundle = gpu_backend_bundle_layout(
        :gpu,
        :ManualBundle,
        GPUBackendManifestFile("manualbackend/manifest.toml", "manifest"),
        (GPUBackendStageFile(:kernel, "manualbackend/kernel.jl", "kernel"),),
    )
    manual_layout = gpu_backend_package_layout(
        :gpu,
        :ManualBackend,
        "manualbackend",
        (manual_bundle,),
    )
    @test length(gpu_backend_bundles(manual_layout)) == 1
    @test length(gpu_backend_files(manual_layout)) == 2
    @test gpu_backend_files(manual_layout)[2].relative_path == "manualbackend/kernel.jl"

    duplicate_bundle = gpu_backend_bundle_layout(
        :gpu,
        :DuplicateBundle,
        GPUBackendManifestFile("duplicate/manifest.toml", "manifest"),
        (
            GPUBackendStageFile(:stage1, "duplicate/kernel.jl", "kernel1"),
            GPUBackendStageFile(:stage2, "duplicate/kernel.jl", "kernel2"),
        ),
    )
    @test_throws ArgumentError gpu_backend_package_layout(
        :gpu,
        :DuplicateBackend,
        "duplicate",
        (duplicate_bundle,),
    )

    mktempdir() do temp_dir
        emission = emit_batched_nuts_package(done_program, temp_dir; target=:gpu)
        @test emission.package.target == done_layout.target
        @test emission.package.backend_symbol == done_layout.backend_symbol
        @test emission.package.root_dir == done_layout.root_dir
        @test length(gpu_backend_bundles(emission.package)) == 1
        @test length(gpu_backend_files(emission.package)) == length(done_layout_files)
        @test emission.output_root == temp_dir
        @test length(emission.written_files) == 2
        @test all(isfile, emission.written_files)

        manifest_path = joinpath(temp_dir, done_layout_files[1].relative_path)
        stage_path = joinpath(temp_dir, done_layout_files[2].relative_path)
        @test read(manifest_path, String) == done_layout_files[1].contents
        @test read(stage_path, String) == done_layout_files[2].contents

        @test_throws ArgumentError emit_batched_nuts_package(
            done_program,
            temp_dir;
            target=:gpu,
        )

        overwrite_emission = emit_batched_nuts_package(
            done_program,
            temp_dir;
            target=:gpu,
            overwrite=true,
        )
        @test overwrite_emission.written_files == emission.written_files
    end

    mktempdir() do temp_dir
        layout = GPUBackendPackageLayout(
            :gpu,
            :ManualBackend,
            "manualbackend",
            (
                GPUBackendFileEntry("manualbackend/kernel.jl", "kernel"),
                GPUBackendFileEntry("manualbackend/manifest.toml", "manifest"),
            ),
        )
        emission = emit_gpu_backend_package(layout, temp_dir)
        @test length(emission.written_files) == 2
        @test read(joinpath(temp_dir, "manualbackend", "kernel.jl"), String) == "kernel"
        @test_throws ArgumentError emit_gpu_backend_package(layout, temp_dir)
        emission2 = emit_gpu_backend_package(layout, temp_dir; overwrite=true)
        @test emission2.written_files == emission.written_files
    end

    mktempdir() do temp_dir
        escaping_layout = GPUBackendPackageLayout(
            :gpu,
            :EscapingBackend,
            "escapingbackend",
            (GPUBackendFileEntry("../escape.jl", "escape"),),
        )
        @test_throws ArgumentError emit_gpu_backend_package(escaping_layout, temp_dir)
    end
