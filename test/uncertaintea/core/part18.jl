    @test UncertainTea.logpdf(gamma(2.0, 3.0), 1.5) ≈
        2.0 * log(3.0) - UncertainTea.loggamma(2.0) + log(1.5) - 4.5 atol=1e-8
    @test UncertainTea.logpdf(gamma(2.0, 3.0), 0.0) == -Inf
    @test UncertainTea.logpdf(studentt(5.0, 0.0, 1.0), 0.25) ≈
        UncertainTea.loggamma(3.0) - UncertainTea.loggamma(2.5) -
        (log(5.0) + log(pi)) / 2 -
        3.0 * log1p(0.25^2 / 5.0) atol=1e-8

    @tea static function gamma_latent_model()
        rate ~ gamma(2.5f0, 1.5f0)
        {:y} ~ normal(rate, 0.5f0)
        return rate
    end

    gamma_latent_constraints = choicemap((:y, 0.9f0))
    gamma_latent_trace, _ = generate(
        gamma_latent_model,
        (),
        gamma_latent_constraints;
        rng=MersenneTwister(130),
    )
    gamma_latent_spec = modelspec(gamma_latent_model)
    gamma_latent_plan = executionplan(gamma_latent_model)
    gamma_latent_backend_plan = backend_execution_plan(gamma_latent_model)
    gamma_latent_params = parameter_vector(gamma_latent_trace)
    gamma_latent_unconstrained = transform_to_unconstrained(gamma_latent_trace)

    @test gamma_latent_spec.choices[1].rhs.family == :gamma
    @test gamma_latent_spec.parameter_layout.slots[1].transform isa LogTransform
    @test gamma_latent_plan.steps[1].parameter_slot == 1
    @test gamma_latent_backend_plan.steps[1] isa UncertainTea.BackendGammaChoicePlanStep
    @test gamma_latent_params[1] > 0
    @test gamma_latent_unconstrained[1] ≈ log(gamma_latent_params[1])
    @test logjoint(gamma_latent_model, gamma_latent_params, (), gamma_latent_constraints) ≈
        assess(
            gamma_latent_model,
            (),
            choicemap((:rate, gamma_latent_trace[:rate]), (:y, 0.9f0)),
        ) atol=1e-6
    @test logjoint_unconstrained(
        gamma_latent_model,
        gamma_latent_unconstrained,
        (),
        gamma_latent_constraints,
    ) ≈ logjoint(gamma_latent_model, gamma_latent_params, (), gamma_latent_constraints) +
        gamma_latent_unconstrained[1] atol=1e-6

    @tea static function studentt_latent_model()
        state ~ studentt(6.0f0, 0.0f0, 1.0f0)
        {:y} ~ normal(state, 0.5f0)
        return state
    end

    studentt_latent_constraints = choicemap((:y, -0.2f0))
    studentt_latent_trace, _ = generate(
        studentt_latent_model,
        (),
        studentt_latent_constraints;
        rng=MersenneTwister(131),
    )
    studentt_latent_spec = modelspec(studentt_latent_model)
    studentt_latent_plan = executionplan(studentt_latent_model)
    studentt_latent_backend_plan = backend_execution_plan(studentt_latent_model)
    studentt_latent_params = parameter_vector(studentt_latent_trace)
    studentt_latent_unconstrained = transform_to_unconstrained(studentt_latent_trace)

    @test studentt_latent_spec.choices[1].rhs.family == :studentt
    @test studentt_latent_spec.parameter_layout.slots[1].transform isa IdentityTransform
    @test studentt_latent_plan.steps[1].parameter_slot == 1
    @test studentt_latent_backend_plan.steps[1] isa UncertainTea.BackendStudentTChoicePlanStep
    @test studentt_latent_unconstrained ≈ studentt_latent_params atol=1e-8
    @test logjoint(studentt_latent_model, studentt_latent_params, (), studentt_latent_constraints) ≈
        assess(
            studentt_latent_model,
            (),
            choicemap((:state, studentt_latent_trace[:state]), (:y, -0.2f0)),
        ) atol=1e-6
    @test logjoint_unconstrained(
        studentt_latent_model,
        studentt_latent_unconstrained,
        (),
        studentt_latent_constraints,
    ) ≈ logjoint(studentt_latent_model, studentt_latent_params, (), studentt_latent_constraints) atol=1e-6

    @tea static function gamma_shape_model()
        log_shape ~ normal(0.0f0, 0.4f0)
        shape = exp(log_shape)
        {:y} ~ gamma(shape, 2.0f0)
        return shape
    end

    gamma_shape_constraints = choicemap((:y, 1.2f0))
    gamma_shape_trace, _ = generate(gamma_shape_model, (), gamma_shape_constraints; rng=MersenneTwister(132))
    gamma_shape_backend_plan = backend_execution_plan(gamma_shape_model)
    gamma_shape_backend_layout = backend_package_layout(gamma_shape_model)
    gamma_shape_backend_stage = gpu_backend_files(gamma_shape_backend_layout)[2]
    gamma_shape_params = parameter_vector(gamma_shape_trace)
    gamma_shape_batch_params = reshape(gamma_shape_params .+ Float64[-0.2, 0.0, 0.15], 1, 3)
    gamma_shape_batch_constraints = [
        choicemap((:y, 0.8f0)),
        choicemap((:y, 1.2f0)),
        choicemap((:y, 1.5f0)),
    ]
    gamma_shape_batch_gradient = batched_logjoint_gradient_unconstrained(
        gamma_shape_model,
        gamma_shape_batch_params,
        (),
        gamma_shape_batch_constraints,
    )
    gamma_shape_batch_cache = BatchedLogjointGradientCache(
        gamma_shape_model,
        gamma_shape_batch_params,
        (),
        gamma_shape_batch_constraints,
    )

    @test gamma_shape_backend_plan.steps[3] isa UncertainTea.BackendGammaChoicePlanStep
    @test occursin("# 3. choice gamma", gamma_shape_backend_stage.contents)
    @test gamma_shape_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            gamma_shape_model,
            gamma_shape_batch_params[:, index],
            (),
            gamma_shape_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(gamma_shape_batch_cache.backend_cache)
    @test isnothing(gamma_shape_batch_cache.flat_cache)
    @test isempty(gamma_shape_batch_cache.column_caches)

    @tea static function studentt_location_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ studentt(7.0f0, mu, 1.0f0)
        return mu
    end

    studentt_location_constraints = choicemap((:y, -0.35f0))
    studentt_location_trace, _ = generate(
        studentt_location_model,
        (),
        studentt_location_constraints;
        rng=MersenneTwister(133),
    )
    studentt_location_backend_plan = backend_execution_plan(studentt_location_model)
    studentt_location_backend_layout = backend_package_layout(studentt_location_model)
    studentt_location_backend_stage = gpu_backend_files(studentt_location_backend_layout)[2]
    studentt_location_params = parameter_vector(studentt_location_trace)
    studentt_location_batch_params = reshape(studentt_location_params .+ Float64[-0.1, 0.0, 0.2], 1, 3)
    studentt_location_batch_constraints = [
        choicemap((:y, -0.7f0)),
        choicemap((:y, -0.35f0)),
        choicemap((:y, 0.1f0)),
    ]
    studentt_location_batch_gradient = batched_logjoint_gradient_unconstrained(
        studentt_location_model,
        studentt_location_batch_params,
        (),
        studentt_location_batch_constraints,
    )
    studentt_location_batch_cache = BatchedLogjointGradientCache(
        studentt_location_model,
        studentt_location_batch_params,
        (),
        studentt_location_batch_constraints,
    )

    @test studentt_location_backend_plan.steps[2] isa UncertainTea.BackendStudentTChoicePlanStep
    @test occursin("# 2. choice studentt", studentt_location_backend_stage.contents)
    @test studentt_location_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            studentt_location_model,
            studentt_location_batch_params[:, index],
            (),
            studentt_location_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(studentt_location_batch_cache.backend_cache)
    @test isnothing(studentt_location_batch_cache.flat_cache)
    @test isempty(studentt_location_batch_cache.column_caches)
