    gaussian_backend_layout = backend_package_layout(gaussian_mean)
    gaussian_backend_files = gpu_backend_files(gaussian_backend_layout)
    gaussian_backend_bundles = gpu_backend_bundles(gaussian_backend_layout)
    @test gaussian_backend_layout.target == :gpu
    @test gaussian_backend_layout.backend_symbol == :UncertainTeaGPUBackendPackage__gaussian_mean
    @test gaussian_backend_layout.root_dir == "uncertainteagpubackendpackage__gaussian_mean"
    @test length(gaussian_backend_bundles) == 1
    @test gaussian_backend_bundles[1].bundle_symbol ==
        :UncertainTeaGPUBackendBundle__gaussian_mean
    @test gpu_backend_manifest_file(gaussian_backend_bundles[1]).relative_path ==
        "uncertainteagpubackendpackage__gaussian_mean/uncertainteagpubackendbundle__gaussian_mean__manifest.toml"
    @test length(gpu_backend_stage_files(gaussian_backend_bundles[1])) == 1
    @test gpu_backend_stage_files(gaussian_backend_bundles[1])[1].stage_name == :gaussian_mean
    @test length(gaussian_backend_files) == 2
    @test endswith(gaussian_backend_files[2].relative_path, ".jl")
    @test occursin("const MODEL = :gaussian_mean", gaussian_backend_files[2].contents)
    @test occursin("const STEP_COUNT = 2", gaussian_backend_files[2].contents)
    @test occursin("# 1. choice normal", gaussian_backend_files[2].contents)
    @test occursin("# 2. choice normal", gaussian_backend_files[2].contents)

    deterministic_backend_layout = backend_package_layout(deterministic_scale)
    deterministic_backend_stage = gpu_backend_stage_files(
        gpu_backend_bundles(deterministic_backend_layout)[1],
    )[1]
    @test occursin("const STEP_COUNT = 4", deterministic_backend_stage.contents)
    @test occursin("# 3. deterministic slot=", deterministic_backend_stage.contents)

    gaussian_metal_backend_layout = backend_package_layout(gaussian_mean; target=:metal)
    gaussian_cuda_backend_layout = backend_package_layout(gaussian_mean; target=:cuda)
    @test gaussian_metal_backend_layout.backend_symbol ==
        :UncertainTeaMetalBackendPackage__gaussian_mean
    @test gaussian_cuda_backend_layout.backend_symbol ==
        :UncertainTeaCUDABackendPackage__gaussian_mean
    @test endswith(gpu_backend_files(gaussian_metal_backend_layout)[2].relative_path, ".metal")
    @test endswith(gpu_backend_files(gaussian_cuda_backend_layout)[2].relative_path, ".cu")
    @test occursin("const TARGET = :metal", gpu_backend_files(gaussian_metal_backend_layout)[2].contents)
    @test occursin("const TARGET = :cuda", gpu_backend_files(gaussian_cuda_backend_layout)[2].contents)

    mktempdir() do temp_dir
        emission = emit_backend_package(gaussian_mean, temp_dir)
        @test emission.package.backend_symbol == gaussian_backend_layout.backend_symbol
        @test emission.package.root_dir == gaussian_backend_layout.root_dir
        @test length(gpu_backend_bundles(emission.package)) == 1
        @test length(emission.written_files) == 2
        @test all(isfile, emission.written_files)

        manifest_path = joinpath(temp_dir, gaussian_backend_files[1].relative_path)
        stage_path = joinpath(temp_dir, gaussian_backend_files[2].relative_path)
        @test read(manifest_path, String) == gaussian_backend_files[1].contents
        @test read(stage_path, String) == gaussian_backend_files[2].contents

        @test_throws ArgumentError emit_backend_package(gaussian_mean, temp_dir)
        overwrite_emission = emit_backend_package(
            gaussian_mean,
            temp_dir;
            overwrite=true,
        )
        @test overwrite_emission.written_files == emission.written_files
    end

    @test_throws ArgumentError backend_package_layout(unsupported_backend_model)
    @test_throws ArgumentError emit_backend_package(
        unsupported_backend_model,
        mktempdir(),
    )
    @test_throws ArgumentError backend_package_layout(gaussian_mean; target=:bogus)
