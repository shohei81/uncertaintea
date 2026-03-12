    @test UncertainTea.logpdf(laplace(0.5, 2.0), 1.5) ≈ -log(4.0) - 0.5 atol=1e-8
    @test UncertainTea.logpdf(geometric(0.25), 3) ≈ log(0.25) + 3 * log(0.75) atol=1e-8
    @test UncertainTea.logpdf(negativebinomial(2.5, 0.4), 3) ≈
        UncertainTea.loggamma(5.5) - UncertainTea.loggamma(2.5) -
        log(6.0) + 2.5 * log(0.4) + 3 * log(0.6) atol=1e-8

    @tea static function laplace_latent_model()
        state ~ laplace(0.0f0, 1.5f0)
        {:y} ~ normal(state, 0.5f0)
        return state
    end

    laplace_constraints = choicemap((:y, 0.35f0))
    laplace_trace, _ = generate(laplace_latent_model, (), laplace_constraints; rng=MersenneTwister(160))
    laplace_spec = modelspec(laplace_latent_model)
    laplace_plan = executionplan(laplace_latent_model)
    laplace_backend_plan = backend_execution_plan(laplace_latent_model)
    laplace_params = parameter_vector(laplace_trace)
    laplace_unconstrained = transform_to_unconstrained(laplace_trace)

    @test laplace_spec.choices[1].rhs.family == :laplace
    @test laplace_spec.parameter_layout.slots[1].transform isa IdentityTransform
    @test laplace_plan.steps[1].parameter_slot == 1
    @test laplace_backend_plan.steps[1] isa UncertainTea.BackendLaplaceChoicePlanStep
    @test laplace_unconstrained ≈ laplace_params atol=1e-8
    @test logjoint(laplace_latent_model, laplace_params, (), laplace_constraints) ≈
        assess(
            laplace_latent_model,
            (),
            choicemap((:state, laplace_trace[:state]), (:y, 0.35f0)),
        ) atol=1e-6

    @tea static function geometric_probability_model()
        logit ~ normal(0.0f0, 0.6f0)
        probability = 1.0f0 / (1.0f0 + exp(-logit))
        {:y} ~ geometric(probability)
        return probability
    end

    geometric_constraints = choicemap((:y, 2))
    geometric_trace, _ = generate(
        geometric_probability_model,
        (),
        geometric_constraints;
        rng=MersenneTwister(161),
    )
    geometric_backend_report = backend_report(geometric_probability_model)
    geometric_backend_plan = backend_execution_plan(geometric_probability_model)
    geometric_backend_layout = backend_package_layout(geometric_probability_model)
    geometric_backend_stage = gpu_backend_files(geometric_backend_layout)[2]
    geometric_params = parameter_vector(geometric_trace)
    geometric_batch_params = reshape(geometric_params .+ Float64[-0.1, 0.0, 0.2], 1, 3)
    geometric_batch_constraints = [
        choicemap((:y, 0)),
        choicemap((:y, 2)),
        choicemap((:y, 5)),
    ]
    geometric_batch_gradient = batched_logjoint_gradient_unconstrained(
        geometric_probability_model,
        geometric_batch_params,
        (),
        geometric_batch_constraints,
    )
    geometric_batch_cache = BatchedLogjointGradientCache(
        geometric_probability_model,
        geometric_batch_params,
        (),
        geometric_batch_constraints,
    )

    @test geometric_backend_report.supported
    @test geometric_backend_plan.steps[3] isa UncertainTea.BackendGeometricChoicePlanStep
    @test occursin("# 3. choice geometric", geometric_backend_stage.contents)
    @test geometric_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            geometric_probability_model,
            geometric_batch_params[:, index],
            (),
            geometric_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(geometric_batch_cache.backend_cache)
    @test isnothing(geometric_batch_cache.flat_cache)
    @test isempty(geometric_batch_cache.column_caches)

    @tea static function negativebinomial_probability_model()
        log_successes ~ normal(0.0f0, 0.5f0)
        logit ~ normal(0.0f0, 0.5f0)
        successes = exp(log_successes)
        probability = 1.0f0 / (1.0f0 + exp(-logit))
        {:y} ~ negativebinomial(successes, probability)
        return successes + probability
    end

    negativebinomial_constraints = choicemap((:y, 4))
    negativebinomial_trace, _ = generate(
        negativebinomial_probability_model,
        (),
        negativebinomial_constraints;
        rng=MersenneTwister(162),
    )
    negativebinomial_backend_report = backend_report(negativebinomial_probability_model)
    negativebinomial_backend_plan = backend_execution_plan(negativebinomial_probability_model)
    negativebinomial_backend_layout = backend_package_layout(negativebinomial_probability_model)
    negativebinomial_backend_stage = gpu_backend_files(negativebinomial_backend_layout)[2]
    negativebinomial_params = parameter_vector(negativebinomial_trace)
    negativebinomial_batch_params = hcat(
        negativebinomial_params .+ Float64[-0.1, 0.05],
        negativebinomial_params,
        negativebinomial_params .+ Float64[0.2, -0.1],
    )
    negativebinomial_batch_constraints = [
        choicemap((:y, 1)),
        choicemap((:y, 4)),
        choicemap((:y, 7)),
    ]
    negativebinomial_batch_gradient = batched_logjoint_gradient_unconstrained(
        negativebinomial_probability_model,
        negativebinomial_batch_params,
        (),
        negativebinomial_batch_constraints,
    )
    negativebinomial_batch_cache = BatchedLogjointGradientCache(
        negativebinomial_probability_model,
        negativebinomial_batch_params,
        (),
        negativebinomial_batch_constraints,
    )

    @test negativebinomial_backend_report.supported
    @test negativebinomial_backend_plan.steps[5] isa UncertainTea.BackendNegativeBinomialChoicePlanStep
    @test occursin("# 5. choice negativebinomial", negativebinomial_backend_stage.contents)
    @test negativebinomial_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            negativebinomial_probability_model,
            negativebinomial_batch_params[:, index],
            (),
            negativebinomial_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(negativebinomial_batch_cache.backend_cache)
    @test isnothing(negativebinomial_batch_cache.flat_cache)
    @test isempty(negativebinomial_batch_cache.column_caches)
