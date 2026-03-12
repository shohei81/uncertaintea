    @test UncertainTea.logpdf(beta(2.0, 3.0), 0.4) ≈
        UncertainTea.loggamma(5.0) - UncertainTea.loggamma(2.0) - UncertainTea.loggamma(3.0) +
        log(0.4) + 2 * log(0.6) atol=1e-8
    @test UncertainTea.logpdf(beta(2.0, 3.0), 0.0) == -Inf
    @test UncertainTea.logpdf(categorical(0.2, 0.3, 0.5), 2) ≈ log(0.3) atol=1e-8
    @test UncertainTea.logpdf(categorical(0.2, 0.3, 0.5), 4) == -Inf

    @tea static function beta_latent_model()
        p ~ beta(2.0f0, 3.0f0)
        {:y} ~ bernoulli(p)
        return p
    end

    beta_constraints = choicemap((:y, 1))
    beta_trace, _ = generate(beta_latent_model, (), beta_constraints; rng=MersenneTwister(140))
    beta_spec = modelspec(beta_latent_model)
    beta_plan = executionplan(beta_latent_model)
    beta_backend_plan = backend_execution_plan(beta_latent_model)
    beta_params = parameter_vector(beta_trace)
    beta_unconstrained = transform_to_unconstrained(beta_trace)

    @test beta_spec.choices[1].rhs.family == :beta
    @test beta_spec.parameter_layout.slots[1].transform isa LogitTransform
    @test beta_plan.steps[1].parameter_slot == 1
    @test beta_backend_plan.steps[1] isa UncertainTea.BackendBetaChoicePlanStep
    @test 0.0 < beta_params[1] < 1.0
    @test beta_unconstrained[1] ≈ log(beta_params[1]) - log1p(-beta_params[1]) atol=1e-6
    @test transform_to_constrained(beta_latent_model, beta_unconstrained) ≈ beta_params atol=1e-8
    @test logjoint(beta_latent_model, beta_params, (), beta_constraints) ≈
        assess(
            beta_latent_model,
            (),
            choicemap((:p, beta_trace[:p]), (:y, 1)),
        ) atol=1e-6
    @test logjoint_unconstrained(beta_latent_model, beta_unconstrained, (), beta_constraints) ≈
        logjoint(beta_latent_model, beta_params, (), beta_constraints) +
        log(beta_params[1]) + log1p(-beta_params[1]) atol=1e-6

    @tea static function beta_shape_model()
        log_alpha ~ normal(0.0f0, 0.3f0)
        alpha = exp(log_alpha)
        {:y} ~ beta(alpha, 2.0f0)
        return alpha
    end

    beta_shape_constraints = choicemap((:y, 0.7f0))
    beta_shape_trace, _ = generate(beta_shape_model, (), beta_shape_constraints; rng=MersenneTwister(141))
    beta_shape_backend_plan = backend_execution_plan(beta_shape_model)
    beta_shape_backend_layout = backend_package_layout(beta_shape_model)
    beta_shape_backend_stage = gpu_backend_files(beta_shape_backend_layout)[2]
    beta_shape_params = parameter_vector(beta_shape_trace)
    beta_shape_batch_params = reshape(beta_shape_params .+ Float64[-0.2, 0.0, 0.15], 1, 3)
    beta_shape_batch_constraints = [
        choicemap((:y, 0.25f0)),
        choicemap((:y, 0.7f0)),
        choicemap((:y, 0.85f0)),
    ]
    beta_shape_batch_gradient = batched_logjoint_gradient_unconstrained(
        beta_shape_model,
        beta_shape_batch_params,
        (),
        beta_shape_batch_constraints,
    )
    beta_shape_batch_cache = BatchedLogjointGradientCache(
        beta_shape_model,
        beta_shape_batch_params,
        (),
        beta_shape_batch_constraints,
    )

    @test beta_shape_backend_plan.steps[3] isa UncertainTea.BackendBetaChoicePlanStep
    @test occursin("# 3. choice beta", beta_shape_backend_stage.contents)
    @test beta_shape_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            beta_shape_model,
            beta_shape_batch_params[:, index],
            (),
            beta_shape_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(beta_shape_batch_cache.backend_cache)
    @test isnothing(beta_shape_batch_cache.flat_cache)
    @test isempty(beta_shape_batch_cache.column_caches)

    @tea static function categorical_weight_model()
        logit1 ~ normal(0.0f0, 0.5f0)
        logit2 ~ normal(0.0f0, 0.5f0)
        w1 = exp(logit1)
        w2 = exp(logit2)
        denom = w1 + w2 + 1.0f0
        {:y} ~ categorical(w1 / denom, w2 / denom, 1.0f0 / denom)
        return logit1 + logit2
    end

    categorical_constraints = choicemap((:y, 2))
    categorical_trace, _ = generate(
        categorical_weight_model,
        (),
        categorical_constraints;
        rng=MersenneTwister(142),
    )
    categorical_backend_report = backend_report(categorical_weight_model)
    categorical_backend_plan = backend_execution_plan(categorical_weight_model)
    categorical_backend_layout = backend_package_layout(categorical_weight_model)
    categorical_backend_stage = gpu_backend_files(categorical_backend_layout)[2]
    categorical_params = parameter_vector(categorical_trace)
    categorical_batch_params = hcat(
        categorical_params .+ Float64[-0.1, 0.05],
        categorical_params,
        categorical_params .+ Float64[0.15, -0.05],
    )
    categorical_batch_constraints = [
        choicemap((:y, 1)),
        choicemap((:y, 2)),
        choicemap((:y, 3)),
    ]
    categorical_batch_gradient = batched_logjoint_gradient_unconstrained(
        categorical_weight_model,
        categorical_batch_params,
        (),
        categorical_batch_constraints,
    )
    categorical_batch_cache = BatchedLogjointGradientCache(
        categorical_weight_model,
        categorical_batch_params,
        (),
        categorical_batch_constraints,
    )

    @test categorical_backend_report.supported
    @test categorical_backend_plan.steps[6] isa UncertainTea.BackendCategoricalChoicePlanStep
    @test occursin("# 6. choice categorical", categorical_backend_stage.contents)
    @test categorical_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            categorical_weight_model,
            categorical_batch_params[:, index],
            (),
            categorical_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(categorical_batch_cache.backend_cache)
    @test isnothing(categorical_batch_cache.flat_cache)
    @test isempty(categorical_batch_cache.column_caches)
