# PR 6.0: element-type generalization of the batched CPU scoring/gradient path.
# Contract: outputs adopt float(eltype(params)); Float32 inputs yield Float32
# results that match the Float64 path within Float32 precision. Float64 stays
# bit-for-bit unchanged (default type parameter remains Float64).

@testset "batched_scoring_eltype_f32" begin
    @tea static function f32_normal_scale_model()
        logsigma ~ normal(0.0f0, 1.0f0)
        sigma = exp(logsigma)
        {:y} ~ normal(0.0f0, sigma)
        return sigma
    end

    f32_normal_params64 = reshape(Float64[-0.3, 0.1, 0.45], 1, 3)
    f32_normal_params32 = Float32.(f32_normal_params64)
    f32_normal_constraints = [
        choicemap((:y, 0.8f0)),
        choicemap((:y, -0.4f0)),
        choicemap((:y, 1.3f0)),
    ]

    f32_normal_values64 = batched_logjoint(f32_normal_scale_model, f32_normal_params64, (), f32_normal_constraints)
    f32_normal_values32 = batched_logjoint(f32_normal_scale_model, f32_normal_params32, (), f32_normal_constraints)
    f32_normal_grad64 =
        batched_logjoint_gradient_unconstrained(f32_normal_scale_model, f32_normal_params64, (), f32_normal_constraints)
    f32_normal_grad32 =
        batched_logjoint_gradient_unconstrained(f32_normal_scale_model, f32_normal_params32, (), f32_normal_constraints)

    @test eltype(f32_normal_values32) == Float32
    @test eltype(f32_normal_grad32) == Float32
    @test eltype(f32_normal_values64) == Float64
    @test eltype(f32_normal_grad64) == Float64
    @test f32_normal_values32 ≈ f32_normal_values64 rtol=1e-5
    @test f32_normal_grad32 ≈ f32_normal_grad64 rtol=1e-4
    for f32_normal_index = 1:3
        @test f32_normal_values64[f32_normal_index] ≈ logjoint(
            f32_normal_scale_model,
            f32_normal_params64[:, f32_normal_index],
            (),
            f32_normal_constraints[f32_normal_index],
        ) atol=1e-8
    end

    @tea static function f32_gamma_shape_model()
        log_shape ~ normal(0.0f0, 0.4f0)
        shape = exp(log_shape)
        {:y} ~ gamma(shape, 2.0f0)
        return shape
    end

    f32_gamma_params64 = reshape(Float64[-0.2, 0.05, 0.3], 1, 3)
    f32_gamma_params32 = Float32.(f32_gamma_params64)
    f32_gamma_constraints = [
        choicemap((:y, 0.8f0)),
        choicemap((:y, 1.2f0)),
        choicemap((:y, 1.5f0)),
    ]

    f32_gamma_values64 = batched_logjoint(f32_gamma_shape_model, f32_gamma_params64, (), f32_gamma_constraints)
    f32_gamma_values32 = batched_logjoint(f32_gamma_shape_model, f32_gamma_params32, (), f32_gamma_constraints)
    f32_gamma_grad64 = batched_logjoint_gradient_unconstrained(f32_gamma_shape_model, f32_gamma_params64, (), f32_gamma_constraints)
    f32_gamma_grad32 = batched_logjoint_gradient_unconstrained(f32_gamma_shape_model, f32_gamma_params32, (), f32_gamma_constraints)

    @test eltype(f32_gamma_values32) == Float32
    @test eltype(f32_gamma_grad32) == Float32
    @test f32_gamma_values32 ≈ f32_gamma_values64 rtol=1e-5
    @test f32_gamma_grad32 ≈ f32_gamma_grad64 rtol=1e-4
    for f32_gamma_index = 1:3
        @test f32_gamma_values64[f32_gamma_index] ≈ logjoint(
            f32_gamma_shape_model,
            f32_gamma_params64[:, f32_gamma_index],
            (),
            f32_gamma_constraints[f32_gamma_index],
        ) atol=1e-8
    end

    @tea static function f32_studentt_scale_model()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ studentt(7.0f0, 0.5f0, exp(s))
        return s
    end

    f32_studentt_params64 = reshape(Float64[-0.4, 0.1, 0.6], 1, 3)
    f32_studentt_params32 = Float32.(f32_studentt_params64)
    f32_studentt_constraints = [
        choicemap((:y, -0.9f0)),
        choicemap((:y, 0.5f0)),
        choicemap((:y, 2.3f0)),
    ]

    f32_studentt_values64 = batched_logjoint(f32_studentt_scale_model, f32_studentt_params64, (), f32_studentt_constraints)
    f32_studentt_values32 = batched_logjoint(f32_studentt_scale_model, f32_studentt_params32, (), f32_studentt_constraints)
    f32_studentt_grad64 =
        batched_logjoint_gradient_unconstrained(f32_studentt_scale_model, f32_studentt_params64, (), f32_studentt_constraints)
    f32_studentt_grad32 =
        batched_logjoint_gradient_unconstrained(f32_studentt_scale_model, f32_studentt_params32, (), f32_studentt_constraints)

    @test eltype(f32_studentt_values32) == Float32
    @test eltype(f32_studentt_grad32) == Float32
    @test f32_studentt_values32 ≈ f32_studentt_values64 rtol=1e-5
    @test f32_studentt_grad32 ≈ f32_studentt_grad64 rtol=1e-4
    for f32_studentt_index = 1:3
        @test f32_studentt_values64[f32_studentt_index] ≈ logjoint(
            f32_studentt_scale_model,
            f32_studentt_params64[:, f32_studentt_index],
            (),
            f32_studentt_constraints[f32_studentt_index],
        ) atol=1e-8
    end
end
