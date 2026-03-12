    @test UncertainTea.logpdf(inversegamma(3.0, 2.0), 1.5) ≈
        3.0 * log(2.0) - UncertainTea.loggamma(3.0) - 4.0 * log(1.5) - 2.0 / 1.5 atol=1e-8
    @test UncertainTea.logpdf(inversegamma(3.0, 2.0), 0.0) == -Inf
    @test UncertainTea.logpdf(weibull(2.0, 3.0), 1.5) ≈
        log(2.0) + log(1.5) - 2.0 * log(3.0) - (1.5 / 3.0)^2 atol=1e-8
    @test UncertainTea.logpdf(weibull(2.0, 3.0), -0.5) == -Inf
    @test UncertainTea.logpdf(UncertainTea.binomial(5, 0.4), 2) ≈
        log(10.0) + 2 * log(0.4) + 3 * log(0.6) atol=1e-8
    @test UncertainTea.logpdf(UncertainTea.binomial(5, 0.4), 6) == -Inf

    @tea static function weibull_latent_model()
        wait ~ weibull(1.8f0, 2.5f0)
        {:y} ~ normal(wait, 0.5f0)
        return wait
    end

    weibull_constraints = choicemap((:y, 1.1f0))
    weibull_trace, _ = generate(weibull_latent_model, (), weibull_constraints; rng=MersenneTwister(150))
    weibull_spec = modelspec(weibull_latent_model)
    weibull_plan = executionplan(weibull_latent_model)
    weibull_backend_plan = backend_execution_plan(weibull_latent_model)
    weibull_params = parameter_vector(weibull_trace)
    weibull_unconstrained = transform_to_unconstrained(weibull_trace)

    @test weibull_spec.choices[1].rhs.family == :weibull
    @test weibull_spec.parameter_layout.slots[1].transform isa LogTransform
    @test weibull_plan.steps[1].parameter_slot == 1
    @test weibull_backend_plan.steps[1] isa UncertainTea.BackendWeibullChoicePlanStep
    @test weibull_params[1] > 0
    @test weibull_unconstrained[1] ≈ log(weibull_params[1]) atol=1e-6
    @test logjoint(weibull_latent_model, weibull_params, (), weibull_constraints) ≈
        assess(
            weibull_latent_model,
            (),
            choicemap((:wait, weibull_trace[:wait]), (:y, 1.1f0)),
        ) atol=1e-6
    @test logjoint_unconstrained(weibull_latent_model, weibull_unconstrained, (), weibull_constraints) ≈
        logjoint(weibull_latent_model, weibull_params, (), weibull_constraints) +
        weibull_unconstrained[1] atol=1e-6

    @tea static function inversegamma_shape_model()
        log_shape ~ normal(0.0f0, 0.4f0)
        shape = exp(log_shape)
        {:y} ~ inversegamma(shape, 2.0f0)
        return shape
    end

    inversegamma_constraints = choicemap((:y, 1.4f0))
    inversegamma_trace, _ = generate(
        inversegamma_shape_model,
        (),
        inversegamma_constraints;
        rng=MersenneTwister(151),
    )
    inversegamma_backend_plan = backend_execution_plan(inversegamma_shape_model)
    inversegamma_backend_layout = backend_package_layout(inversegamma_shape_model)
    inversegamma_backend_stage = gpu_backend_files(inversegamma_backend_layout)[2]
    inversegamma_params = parameter_vector(inversegamma_trace)
    inversegamma_batch_params = reshape(inversegamma_params .+ Float64[-0.2, 0.0, 0.15], 1, 3)
    inversegamma_batch_constraints = [
        choicemap((:y, 0.9f0)),
        choicemap((:y, 1.4f0)),
        choicemap((:y, 2.2f0)),
    ]
    inversegamma_batch_gradient = batched_logjoint_gradient_unconstrained(
        inversegamma_shape_model,
        inversegamma_batch_params,
        (),
        inversegamma_batch_constraints,
    )
    inversegamma_batch_cache = BatchedLogjointGradientCache(
        inversegamma_shape_model,
        inversegamma_batch_params,
        (),
        inversegamma_batch_constraints,
    )

    @test inversegamma_backend_plan.steps[3] isa UncertainTea.BackendInverseGammaChoicePlanStep
    @test occursin("# 3. choice inversegamma", inversegamma_backend_stage.contents)
    @test inversegamma_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            inversegamma_shape_model,
            inversegamma_batch_params[:, index],
            (),
            inversegamma_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(inversegamma_batch_cache.backend_cache)
    @test isnothing(inversegamma_batch_cache.flat_cache)
    @test isempty(inversegamma_batch_cache.column_caches)

    @tea static function binomial_probability_model()
        logit ~ normal(0.0f0, 0.5f0)
        probability = 1.0f0 / (1.0f0 + exp(-logit))
        {:y} ~ binomial(8, probability)
        return probability
    end

    binomial_constraints = choicemap((:y, 4))
    binomial_trace, _ = generate(
        binomial_probability_model,
        (),
        binomial_constraints;
        rng=MersenneTwister(152),
    )
    binomial_backend_report = backend_report(binomial_probability_model)
    binomial_backend_plan = backend_execution_plan(binomial_probability_model)
    binomial_backend_layout = backend_package_layout(binomial_probability_model)
    binomial_backend_stage = gpu_backend_files(binomial_backend_layout)[2]
    binomial_params = parameter_vector(binomial_trace)
    binomial_batch_params = reshape(binomial_params .+ Float64[-0.1, 0.0, 0.15], 1, 3)
    binomial_batch_constraints = [
        choicemap((:y, 2)),
        choicemap((:y, 4)),
        choicemap((:y, 6)),
    ]
    binomial_batch_gradient = batched_logjoint_gradient_unconstrained(
        binomial_probability_model,
        binomial_batch_params,
        (),
        binomial_batch_constraints,
    )
    binomial_batch_cache = BatchedLogjointGradientCache(
        binomial_probability_model,
        binomial_batch_params,
        (),
        binomial_batch_constraints,
    )

    @test binomial_backend_report.supported
    @test binomial_backend_plan.steps[3] isa UncertainTea.BackendBinomialChoicePlanStep
    @test occursin("# 3. choice binomial", binomial_backend_stage.contents)
    @test binomial_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            binomial_probability_model,
            binomial_batch_params[:, index],
            (),
            binomial_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(binomial_batch_cache.backend_cache)
    @test isnothing(binomial_batch_cache.flat_cache)
    @test isempty(binomial_batch_cache.column_caches)
