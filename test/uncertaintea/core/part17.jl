    @test UncertainTea.logpdf(exponential(2.0), 0.5) ≈ log(2.0) - 1.0 atol=1e-8
    @test UncertainTea.logpdf(exponential(2.0), -0.5) == -Inf
    @test UncertainTea.logpdf(poisson(3.0), 4) ≈ 4 * log(3.0) - 3.0 - log(24.0) atol=1e-8
    @test UncertainTea.logpdf(poisson(3.0), -1) == -Inf

    @tea static function exponential_wait_model()
        wait ~ exponential(1.5f0)
        {:y} ~ normal(wait, 0.5f0)
        return wait
    end

    exponential_constraints = choicemap((:y, 0.8f0))
    exponential_trace, _ = generate(
        exponential_wait_model,
        (),
        exponential_constraints;
        rng=MersenneTwister(120),
    )
    exponential_spec = modelspec(exponential_wait_model)
    exponential_plan = executionplan(exponential_wait_model)
    exponential_backend_plan = backend_execution_plan(exponential_wait_model)
    exponential_params = parameter_vector(exponential_trace)
    exponential_unconstrained = transform_to_unconstrained(exponential_trace)

    @test exponential_spec.choices[1].rhs.family == :exponential
    @test parametercount(exponential_spec.parameter_layout) == 1
    @test exponential_spec.parameter_layout.slots[1].transform isa LogTransform
    @test exponential_plan.steps[1].parameter_slot == 1
    @test exponential_backend_plan.steps[1] isa UncertainTea.BackendExponentialChoicePlanStep
    @test exponential_params[1] > 0
    @test exponential_unconstrained[1] ≈ log(exponential_params[1])
    @test transform_to_constrained(exponential_wait_model, exponential_unconstrained) ≈ exponential_params atol=1e-8
    @test logjoint(exponential_wait_model, exponential_params, (), exponential_constraints) ≈
        assess(
            exponential_wait_model,
            (),
            choicemap((:wait, exponential_trace[:wait]), (:y, 0.8f0)),
        ) atol=1e-6
    @test logjoint_unconstrained(
        exponential_wait_model,
        exponential_unconstrained,
        (),
        exponential_constraints,
    ) ≈ logjoint(exponential_wait_model, exponential_params, (), exponential_constraints) +
        exponential_unconstrained[1] atol=1e-6

    @tea static function poisson_rate_model()
        rate ~ exponential(1.0f0)
        {:y} ~ poisson(rate)
        return rate
    end

    poisson_constraints = choicemap((:y, 3))
    poisson_trace, _ = generate(poisson_rate_model, (), poisson_constraints; rng=MersenneTwister(121))
    poisson_spec = modelspec(poisson_rate_model)
    poisson_plan = executionplan(poisson_rate_model)
    poisson_backend_report = backend_report(poisson_rate_model)
    poisson_backend_plan = backend_execution_plan(poisson_rate_model)
    poisson_backend_layout = backend_package_layout(poisson_rate_model)
    poisson_backend_stage = gpu_backend_files(poisson_backend_layout)[2]
    poisson_params = parameter_vector(poisson_trace)
    poisson_unconstrained = transform_to_unconstrained(poisson_trace)
    poisson_batch_params = reshape(
        poisson_unconstrained .+ Float64[0.0, 0.2, -0.15],
        1,
        3,
    )
    poisson_batch_constraints = [
        choicemap((:y, 2)),
        choicemap((:y, 3)),
        choicemap((:y, 5)),
    ]
    poisson_batch_gradient = batched_logjoint_gradient_unconstrained(
        poisson_rate_model,
        poisson_batch_params,
        (),
        poisson_batch_constraints,
    )
    poisson_batch_cache = BatchedLogjointGradientCache(
        poisson_rate_model,
        poisson_batch_params,
        (),
        poisson_batch_constraints,
    )
    poisson_combined_values = fill(-1.0, 3)
    poisson_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        poisson_combined_values,
        poisson_batch_cache,
        poisson_batch_params,
    )[2]

    @test poisson_spec.choices[1].rhs.family == :exponential
    @test poisson_spec.choices[2].rhs.family == :poisson
    @test poisson_spec.parameter_layout.slots[1].transform isa LogTransform
    @test poisson_plan.steps[1].parameter_slot == 1
    @test isnothing(poisson_plan.steps[2].parameter_slot)
    @test poisson_backend_report.supported
    @test poisson_backend_plan.steps[1] isa UncertainTea.BackendExponentialChoicePlanStep
    @test poisson_backend_plan.steps[2] isa UncertainTea.BackendPoissonChoicePlanStep
    @test occursin("# 1. choice exponential", poisson_backend_stage.contents)
    @test occursin("# 2. choice poisson", poisson_backend_stage.contents)
    @test logjoint(poisson_rate_model, poisson_params, (), poisson_constraints) ≈
        assess(
            poisson_rate_model,
            (),
            choicemap((:rate, poisson_trace[:rate]), (:y, 3)),
        ) atol=1e-6
    @test logjoint_unconstrained(poisson_rate_model, poisson_unconstrained, (), poisson_constraints) ≈
        logjoint(poisson_rate_model, poisson_params, (), poisson_constraints) +
        poisson_unconstrained[1] atol=1e-6
    @test poisson_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            poisson_rate_model,
            poisson_batch_params[:, index],
            (),
            poisson_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(poisson_batch_cache.backend_cache)
    @test isnothing(poisson_batch_cache.flat_cache)
    @test isempty(poisson_batch_cache.column_caches)
    @test poisson_combined_values ≈ [
        logjoint_unconstrained(
            poisson_rate_model,
            poisson_batch_params[:, index],
            (),
            poisson_batch_constraints[index],
        ) for index in 1:3
    ] atol=1e-8
    @test poisson_combined_gradient === poisson_batch_cache.gradient_buffer
    @test poisson_combined_gradient ≈ poisson_batch_gradient atol=1e-8
