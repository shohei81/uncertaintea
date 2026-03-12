    @tea static function gaussian_mean()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 1.0f0)
        return mu
    end

    constraints = choicemap((:y, 0.3f0))
    trace, logw = generate(gaussian_mean, (), constraints; rng=MersenneTwister(1))
    spec = modelspec(gaussian_mean)
    plan = executionplan(gaussian_mean)

    @test trace.model === gaussian_mean
    @test haskey(trace.choices, :mu)
    @test trace[:y] == 0.3f0
    @test logw == trace.log_weight
    @test isfinite(logw)
    @test trace.retval == trace[:mu]
    @test spec.name == :gaussian_mean
    @test spec.mode == :static
    @test spec.arguments == Symbol[]
    @test length(spec.choices) == 2
    @test spec.choices[1].binding == :mu
    @test isstaticaddress(spec.choices[1].address)
    @test isstaticaddress(spec.choices[2].address)
    @test spec.choices[1].rhs isa DistributionSpec
    @test spec.choices[1].rhs.family == :normal
    @test spec.choices[2].rhs isa DistributionSpec
    @test !isrepeatedchoice(spec.choices[1])
    @test !hasrepeatedchoices(spec)
    @test parametercount(spec.parameter_layout) == 1
    @test parameterlayout(gaussian_mean).slots[1].binding == :mu
    @test parameterlayout(gaussian_mean).slots[1].choice_index == 1
    @test parameterlayout(gaussian_mean).slots[1].transform isa IdentityTransform
    @test length(plan.steps) == 2
    @test plan.steps[1].parameter_slot == 1
    @test isnothing(plan.steps[2].parameter_slot)

    params = parameter_vector(trace)
    initial = initialparameters(gaussian_mean; rng=MersenneTwister(11))
    overrides = parameterchoicemap(gaussian_mean, params)
    expected_joint = UncertainTea.logpdf(normal(0.0f0, 1.0f0), trace[:mu]) +
        UncertainTea.logpdf(normal(trace[:mu], 1.0f0), 0.3f0)

    @test params == [Float64(trace[:mu])]
    @test length(initial) == 1
    @test overrides[:mu] == trace[:mu]
    @test logjoint(gaussian_mean, params, (), constraints; rng=MersenneTwister(7)) ≈ expected_joint atol=1e-6
    @test logjoint_unconstrained(gaussian_mean, params, (), constraints; rng=MersenneTwister(15)) ≈ expected_joint atol=1e-6
    @test logjoint_gradient_unconstrained(gaussian_mean, params, (), constraints)[1] ≈
        (0.3f0 - 2 * trace[:mu]) atol=1e-5
    @test logjoint(gaussian_mean, params, (), constraints) ≈
        assess(gaussian_mean, (), choicemap((:mu, trace[:mu]), (:y, 0.3f0))) atol=1e-6

    nested_choices = choicemap((:state => 1 => :z, 2.5f0))
    @test nested_choices[:state => 1 => :z] == 2.5f0
    @test nested_choices[(:state, 1, :z)] == 2.5f0
    @test haskey(nested_choices, (:state, 1, :z))
    duplicate_choices = choicemap((:y, 0.1f0), (:y, 0.2f0))
    @test length(duplicate_choices) == 1
    @test duplicate_choices[:y] == 0.2f0

    @tea static function iid_model(n)
        mu ~ normal(0.0f0, 1.0f0)
        for i in 1:n
            {:y => i} ~ normal(mu, 1.0f0)
        end
        return mu
    end

    @tea static function shifted_iid_model(n)
        mu ~ normal(0.0f0, 1.0f0)
        for i in 1:n
            {:y => i + 1} ~ normal(mu, 1.0f0)
        end
        return mu
    end

    @tea static function offset_iid_model(n, offset)
        mu ~ normal(0.0f0, 1.0f0)
        for i in 1:n
            {:y => i + offset} ~ normal(mu, 1.0f0)
        end
        return mu
    end

    @tea static function indexed_scale_model(n)
        mu ~ normal(0.0f0, 1.0f0)
        for i in 1:n
            {:y => i} ~ normal(mu, exp(mu + i / 10))
        end
        return mu
    end

    ys = Float32[0.1f0, -0.2f0, 0.4f0]
    repeated = choicemap((:y => i, ys[i]) for i in eachindex(ys))
    trace2, logw2 = generate(iid_model, (length(ys),), repeated; rng=MersenneTwister(2))
    spec2 = modelspec(iid_model)
    plan2 = executionplan(iid_model)

    @test trace2[:y => 1] == ys[1]
    @test trace2[:y => 3] == ys[3]
    @test isfinite(logw2)
    @test spec2.arguments == [:n]
    @test length(spec2.choices) == 2
    @test spec2.shape_specialized
    @test isaddresstemplate(spec2.choices[2].address)
    @test spec2.choices[2].rhs isa DistributionSpec
    @test hasrepeatedchoices(spec2)
    @test !isrepeatedchoice(spec2.choices[1])
    @test isrepeatedchoice(spec2.choices[2])
    @test length(spec2.choices[2].scopes) == 1
    @test spec2.choices[2].scopes[1].iterator == :i
    @test spec2.choices[2].scopes[1].iterable == :(1:n)
    @test spec2.choices[2].scopes[1].shape_specialized
    @test parametercount(spec2.parameter_layout) == 1
    @test spec2.parameter_layout.slots[1].binding == :mu
    @test spec2.parameter_layout.slots[1].transform isa IdentityTransform
    @test plan2.steps[1].parameter_slot == 1
    @test plan2.steps[2] isa LoopPlanStep
    @test plan2.steps[2].iterator == :i
    @test plan2.steps[2].iterable == :(1:n)
    @test length(plan2.steps[2].body) == 1
    @test plan2.steps[2].body[1] isa ChoicePlanStep
    @test isempty(plan2.steps[2].body[1].scopes)

    params2 = [Float64(trace2[:mu])]
    expected_joint2 = UncertainTea.logpdf(normal(0.0f0, 1.0f0), trace2[:mu]) +
        sum(UncertainTea.logpdf(normal(trace2[:mu], 1.0f0), y) for y in ys)
    full_repeated = choicemap([(:mu, trace2[:mu]); [(:y => i, ys[i]) for i in eachindex(ys)]])

    @test logjoint(iid_model, params2, (length(ys),), repeated; rng=MersenneTwister(8)) ≈ expected_joint2 atol=1e-6
    @test logjoint(iid_model, params2, (length(ys),), repeated) ≈
        assess(iid_model, (length(ys),), full_repeated) atol=1e-6

    @tea static function step(prev)
        z ~ normal(prev, 1.0f0)
        return z
    end

    step_spec = modelspec(step)
    step_plan = executionplan(step)
    step_trace, _ = generate(step, (2.0f0,), choicemap(); rng=MersenneTwister(4))
    @test parametercount(step_spec.parameter_layout) == 1
    @test step_spec.parameter_layout.slots[1].binding == :z
    @test step_spec.parameter_layout.slots[1].transform isa IdentityTransform
    @test step_plan.steps[1].parameter_slot == 1
    @test logjoint(step, parameter_vector(step_trace), (2.0f0,), choicemap(); rng=MersenneTwister(9)) ≈
        UncertainTea.logpdf(normal(2.0f0, 1.0f0), step_trace[:z]) atol=1e-6

    @tea static function chain_model(T)
        z = ({:z => 1} ~ step(0.0f0))
        for t in 2:T
            z = ({:z => t} ~ step(z))
        end
        return z
    end

    trace3, _ = generate(chain_model, (3,), choicemap(); rng=MersenneTwister(3))
    spec3 = modelspec(chain_model)
    plan3 = executionplan(chain_model)

    @test haskey(trace3.choices, :z => 1 => :z)
    @test haskey(trace3.choices, :z => 3 => :z)
    @test length(spec3.choices) == 2
    @test isstaticaddress(spec3.choices[1].address)
    @test isaddresstemplate(spec3.choices[2].address)
    @test spec3.choices[1].rhs isa GenerativeCallSpec
    @test spec3.choices[1].rhs.callee === step
    @test spec3.choices[2].rhs isa GenerativeCallSpec
    @test !isrepeatedchoice(spec3.choices[1])
    @test isrepeatedchoice(spec3.choices[2])
    @test spec3.choices[2].scopes[1].iterator == :t
    @test spec3.choices[2].scopes[1].iterable == :(2:T)
    @test parametercount(spec3.parameter_layout) == 1
    @test length(plan3.steps) == 3
    @test plan3.steps[1] isa ChoicePlanStep
    @test plan3.steps[1].parameter_slot == 1
    @test plan3.steps[2] isa DeterministicPlanStep
    @test plan3.steps[2].binding == :z
    @test plan3.steps[3] isa LoopPlanStep
    @test plan3.steps[3].iterator == :t
    @test length(plan3.steps[3].body) == 2
    @test plan3.steps[3].body[1] isa ChoicePlanStep
    @test plan3.steps[3].body[2] isa DeterministicPlanStep
    params3 = parameter_vector(trace3)
    chain_overrides = parameterchoicemap(chain_model, params3)
    @test params3 == [Float64(trace3[:z => 1 => :z])]
    @test chain_overrides[:z => 1 => :z] == trace3[:z => 1 => :z]
    chain_constraints = choicemap(
        (:z => 2 => :z, trace3[:z => 2 => :z]),
        (:z => 3 => :z, trace3[:z => 3 => :z]),
    )
    @test logjoint(chain_model, params3, (3,), chain_constraints; rng=MersenneTwister(10)) ≈
        assess(
            chain_model,
            (3,),
            choicemap(
                (:z => 1 => :z, trace3[:z => 1 => :z]),
                (:z => 2 => :z, trace3[:z => 2 => :z]),
                (:z => 3 => :z, trace3[:z => 3 => :z]),
            ),
        ) atol=1e-6

    @tea static function observed_step()
        z = ({:state} ~ step(1.5f0))
        {:y} ~ normal(z, 1.0f0)
        return z
    end

    observed_trace, _ = generate(observed_step, (), choicemap((:y, 0.25f0)); rng=MersenneTwister(13))
    observed_spec = modelspec(observed_step)
    observed_plan = executionplan(observed_step)
    observed_params = parameter_vector(observed_trace)
    observed_overrides = parameterchoicemap(observed_step, observed_params)

    @test parametercount(observed_spec.parameter_layout) == 1
    @test observed_plan.steps[1] isa ChoicePlanStep
    @test observed_plan.steps[1].parameter_slot == 1
    @test observed_plan.steps[2] isa DeterministicPlanStep
    @test observed_plan.steps[3] isa ChoicePlanStep
    @test observed_params == [Float64(observed_trace[:state => :z])]
    @test observed_overrides[:state => :z] == observed_trace[:state => :z]
    @test logjoint(observed_step, observed_params, (), choicemap((:y, 0.25f0))) ≈
        assess(observed_step, (), choicemap((:state => :z, observed_trace[:state => :z]), (:y, 0.25f0))) atol=1e-6

    @tea static function nested_loop_model(n, m)
        z ~ normal(0.0f0, 1.0f0)
        for i in 1:n
            for j in 1:m
                {:grid => i => j} ~ bernoulli(0.5f0)
            end
        end
        return z
    end

    spec4 = modelspec(nested_loop_model)
    plan4 = executionplan(nested_loop_model)

    @test length(spec4.choices) == 2
    @test isrepeatedchoice(spec4.choices[2])
    @test length(spec4.choices[2].scopes) == 2
    @test spec4.choices[2].scopes[1].iterator == :i
    @test spec4.choices[2].scopes[2].iterator == :j
    @test parametercount(spec4.parameter_layout) == 1
    @test spec4.parameter_layout.slots[1].binding == :z
    @test length(plan4.steps) == 2
    @test plan4.steps[1].parameter_slot == 1
    @test plan4.steps[2] isa LoopPlanStep
    @test plan4.steps[2].iterator == :i
    @test length(plan4.steps[2].body) == 1
    @test plan4.steps[2].body[1] isa LoopPlanStep
    @test plan4.steps[2].body[1].iterator == :j
    @test length(plan4.steps[2].body[1].body) == 1
    @test plan4.steps[2].body[1].body[1] isa ChoicePlanStep

    @tea static function deterministic_scale()
        mu ~ normal(0.0f0, 1.0f0)
        log_sigma ~ normal(0.0f0, 1.0f0)
        sigma = exp(log_sigma)
        {:y} ~ normal(mu, sigma)
        return (; mu, sigma)
    end

    deterministic_trace, _ = generate(deterministic_scale, (), choicemap((:y, 0.4f0)); rng=MersenneTwister(12))
    deterministic_spec = modelspec(deterministic_scale)
    deterministic_plan = executionplan(deterministic_scale)
    deterministic_params = parameter_vector(deterministic_trace)

    @test parametercount(deterministic_spec.parameter_layout) == 2
    @test deterministic_plan.steps[3] isa DeterministicPlanStep
    @test deterministic_plan.steps[3].binding == :sigma
    @test logjoint(deterministic_scale, deterministic_params, (), choicemap((:y, 0.4f0))) ≈
        assess(
            deterministic_scale,
            (),
            choicemap((:mu, deterministic_trace[:mu]), (:log_sigma, deterministic_trace[:log_sigma]), (:y, 0.4f0)),
        ) atol=1e-6

    @tea static function inline_scale()
        log_sigma ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(0.0f0, exp(log_sigma))
        return log_sigma
    end

    inline_trace, _ = generate(inline_scale, (), choicemap((:y, 0.2f0)); rng=MersenneTwister(6))
    inline_params = parameter_vector(inline_trace)

    @test logjoint(inline_scale, inline_params, (), choicemap((:y, 0.2f0))) ≈
        assess(inline_scale, (), choicemap((:log_sigma, inline_trace[:log_sigma]), (:y, 0.2f0))) atol=1e-6

    @tea static function abs_scale_model()
        log_sigma ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(0.0f0, exp(abs(log_sigma)))
        return log_sigma
    end

    abs_scale_trace, _ = generate(abs_scale_model, (), choicemap((:y, 0.6f0)); rng=MersenneTwister(16))
    abs_scale_params = parameter_vector(abs_scale_trace)

    @tea static function power_scale_model()
        log_sigma ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(0.0f0, exp(log_sigma) ^ 2)
        return log_sigma
    end

    power_scale_trace, _ = generate(power_scale_model, (), choicemap((:y, 0.4f0)); rng=MersenneTwister(17))
    power_scale_params = parameter_vector(power_scale_trace)

    @tea static function min_scale_model()
        log_sigma ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(0.0f0, exp(min(log_sigma, 0.15f0)))
        return log_sigma
    end

    min_scale_trace, _ = generate(min_scale_model, (), choicemap((:y, 0.5f0)); rng=MersenneTwister(18))
    min_scale_params = parameter_vector(min_scale_trace)

    @tea static function max_scale_model()
        log_sigma ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(0.0f0, exp(max(log_sigma, -0.25f0)))
        return log_sigma
    end

    max_scale_trace, _ = generate(max_scale_model, (), choicemap((:y, 0.5f0)); rng=MersenneTwister(19))
    max_scale_params = parameter_vector(max_scale_trace)

    @tea static function mod_scale_model()
        log_sigma ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(0.0f0, exp(log_sigma % 0.5f0))
        return log_sigma
    end

    mod_scale_trace, _ = generate(mod_scale_model, (), choicemap((:y, 0.5f0)); rng=MersenneTwister(20))
    mod_scale_params = parameter_vector(mod_scale_trace)

    @tea static function clamp_scale_model()
        log_sigma ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(0.0f0, exp(clamp(log_sigma, -0.2f0, 0.3f0)))
        return log_sigma
    end

    clamp_scale_trace, _ = generate(clamp_scale_model, (), choicemap((:y, 0.5f0)); rng=MersenneTwister(21))
    clamp_scale_params = parameter_vector(clamp_scale_trace)

    @tea static function unsupported_backend_model()
        mu ~ normal(0.0f0, 1.0f0)
        sigma = sin(mu)
        {:y} ~ normal(mu, 1.0f0)
        return sigma
    end

    @tea static function observed_coin()
        {:y} ~ bernoulli(0.25f0)
        return nothing
    end

    @tea static function positive_latent()
        sigma ~ lognormal(0.0f0, 0.5f0)
        {:y} ~ normal(sigma, 1.0f0)
        return sigma
    end

    positive_trace, _ = generate(positive_latent, (), choicemap((:y, 1.5f0)); rng=MersenneTwister(5))
    positive_spec = modelspec(positive_latent)
    positive_params = parameter_vector(positive_trace)
    unconstrained = transform_to_unconstrained(positive_trace)
    reconstrained = transform_to_constrained(positive_latent, unconstrained)

    @test parametercount(positive_spec.parameter_layout) == 1
    @test positive_spec.parameter_layout.slots[1].transform isa LogTransform
    @test positive_params[1] > 0
    @test unconstrained[1] ≈ log(positive_params[1])
    @test reconstrained ≈ positive_params
    @test logjoint_unconstrained(positive_latent, unconstrained, (), choicemap((:y, 1.5f0))) ≈
        logjoint(positive_latent, positive_params, (), choicemap((:y, 1.5f0))) + unconstrained[1] atol=1e-6
    @test logjoint(positive_latent, positive_params, (), choicemap((:y, 1.5f0))) ≈
        assess(positive_latent, (), choicemap((:sigma, positive_trace[:sigma]), (:y, 1.5f0))) atol=1e-6

    @tea static function positive_step()
        sigma ~ lognormal(0.0f0, 0.5f0)
        return sigma
    end

    @tea static function observed_positive_step()
        sigma = ({:state} ~ positive_step())
        {:y} ~ normal(sigma, 1.0f0)
        return sigma
    end

    positive_step_trace, _ = generate(observed_positive_step, (), choicemap((:y, 1.2f0)); rng=MersenneTwister(14))
    positive_step_spec = modelspec(observed_positive_step)
    positive_step_params = parameter_vector(positive_step_trace)
    positive_step_unconstrained = transform_to_unconstrained(positive_step_trace)
    positive_step_reconstrained = transform_to_constrained(observed_positive_step, positive_step_unconstrained)
    gaussian_batch_params = reshape(Float64[-0.4, 0.0, 0.6], 1, 3)
    gaussian_batch_constraints = [
        choicemap((:y, -0.1f0)),
        choicemap((:y, 0.3f0)),
        choicemap((:y, 0.7f0)),
    ]
    gaussian_batch_logjoint = batched_logjoint(gaussian_mean, gaussian_batch_params, (), gaussian_batch_constraints)
    gaussian_batch_gradient = batched_logjoint_gradient_unconstrained(gaussian_mean, gaussian_batch_params, (), gaussian_batch_constraints)
    gaussian_batch_gradient_cache = BatchedLogjointGradientCache(gaussian_mean, gaussian_batch_params, (), gaussian_batch_constraints)
    gaussian_shared_batch_logjoint = batched_logjoint_unconstrained(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    gaussian_shared_batch_gradient = batched_logjoint_gradient_unconstrained(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    gaussian_batch_params_shifted = gaussian_batch_params .+ 0.15
    gaussian_batch_gradient_shifted = batched_logjoint_gradient_unconstrained(
        gaussian_batch_gradient_cache,
        gaussian_batch_params_shifted,
    )
    gaussian_workspace = UncertainTea.BatchedLogjointWorkspace(gaussian_mean)
    iid_shared_workspace = UncertainTea.BatchedLogjointWorkspace(iid_model)
    iid_batch_params = reshape(Float64[-0.2, 0.4], 1, 2)
    iid_batch_args = [(2,), (3,)]
    iid_batch_constraints = [
        choicemap((:y => 1, 0.1f0), (:y => 2, -0.2f0)),
        choicemap((:y => 1, 0.5f0), (:y => 2, 0.2f0), (:y => 3, -0.1f0)),
    ]
    heterogeneous_iid_args = Any[(Int32(2),), (3,)]
    heterogeneous_iid_gradient_cache = BatchedLogjointGradientCache(
        iid_model,
        iid_batch_params,
        heterogeneous_iid_args,
        iid_batch_constraints,
    )
    iid_batch_logjoint = batched_logjoint(iid_model, iid_batch_params, iid_batch_args, iid_batch_constraints)
    iid_shared_batch_params = reshape(Float64[-0.15, 0.25], 1, 2)
    iid_shared_batch_constraints = [
        choicemap((:y => 1, 0.0f0), (:y => 2, -0.1f0), (:y => 3, 0.3f0)),
        choicemap((:y => 1, 0.4f0), (:y => 2, 0.2f0), (:y => 3, -0.2f0)),
    ]
    iid_shared_batch_logjoint = batched_logjoint(
        iid_model,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints,
    )
    iid_shared_single_constraint_logjoint = batched_logjoint(
        iid_model,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints[1],
    )
    iid_shared_gradient_cache = BatchedLogjointGradientCache(
        iid_model,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints,
    )
    shifted_batch_params = reshape(Float64[-0.1, 0.3], 1, 2)
    shifted_batch_constraints = [
        choicemap((:y => 2, 0.0f0), (:y => 3, -0.1f0), (:y => 4, 0.2f0)),
        choicemap((:y => 2, 0.5f0), (:y => 3, 0.1f0), (:y => 4, -0.3f0)),
    ]
    shifted_batch_logjoint = batched_logjoint(
        shifted_iid_model,
        shifted_batch_params,
        (3,),
        shifted_batch_constraints,
    )
    offset_batch_params = reshape(Float64[-0.25, 0.15], 1, 2)
    offset_batch_args = [(3, 1), (3, 2)]
    offset_batch_constraints = [
        choicemap((:y => 2, 0.0f0), (:y => 3, -0.1f0), (:y => 4, 0.2f0)),
        choicemap((:y => 3, 0.5f0), (:y => 4, 0.1f0), (:y => 5, -0.3f0)),
    ]
    offset_batch_logjoint = batched_logjoint(
        offset_iid_model,
        offset_batch_params,
        offset_batch_args,
        offset_batch_constraints,
    )
    indexed_scale_batch_params = reshape(Float64[-0.2, 0.35], 1, 2)
    indexed_scale_batch_constraints = [
        choicemap((:y => 1, 0.2f0), (:y => 2, -0.1f0), (:y => 3, 0.5f0)),
        choicemap((:y => 1, -0.4f0), (:y => 2, 0.1f0), (:y => 3, 0.3f0)),
    ]
    indexed_scale_batch_logjoint = batched_logjoint(
        indexed_scale_model,
        indexed_scale_batch_params,
        (3,),
        indexed_scale_batch_constraints,
    )
    deterministic_batch_params = [
        deterministic_params[1] deterministic_params[1] + 0.15 deterministic_params[1] - 0.2
        deterministic_params[2] deterministic_params[2] - 0.05 deterministic_params[2] + 0.1
    ]
    deterministic_batch_constraints = [
        choicemap((:y, 0.4f0)),
        choicemap((:y, -0.1f0)),
        choicemap((:y, 0.8f0)),
    ]
    deterministic_batch_logjoint = batched_logjoint(
        deterministic_scale,
        deterministic_batch_params,
        (),
        deterministic_batch_constraints,
    )
    deterministic_workspace = UncertainTea.BatchedLogjointWorkspace(deterministic_scale)
    offset_workspace = UncertainTea.BatchedLogjointWorkspace(offset_iid_model)
    positive_batch_unconstrained = reshape(
        [
            positive_step_unconstrained[1] - 0.2,
            positive_step_unconstrained[1],
            positive_step_unconstrained[1] + 0.2,
        ],
        1,
        3,
    )
    positive_batch_constraints = [
        choicemap((:y, 0.8f0)),
        choicemap((:y, 1.2f0)),
        choicemap((:y, 1.5f0)),
    ]
    positive_batch_logjoint = batched_logjoint_unconstrained(
        observed_positive_step,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    )
    positive_batch_gradient = batched_logjoint_gradient_unconstrained(
        observed_positive_step,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    )
    positive_batch_gradient_cache = BatchedLogjointGradientCache(
        observed_positive_step,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    )
    abs_scale_batch_params = reshape(
        [abs_scale_params[1] - 0.25, abs_scale_params[1], abs_scale_params[1] + 0.35],
        1,
        3,
    )
    abs_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.6f0)),
        choicemap((:y, 1.1f0)),
    ]
    abs_scale_gradient = batched_logjoint_gradient_unconstrained(
        abs_scale_model,
        abs_scale_batch_params,
        (),
        abs_scale_batch_constraints,
    )
    abs_scale_gradient_cache = BatchedLogjointGradientCache(
        abs_scale_model,
        abs_scale_batch_params,
        (),
        abs_scale_batch_constraints,
    )
    power_scale_batch_params = reshape(
        [power_scale_params[1] - 0.2, power_scale_params[1], power_scale_params[1] + 0.3],
        1,
        3,
    )
    power_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.4f0)),
        choicemap((:y, 0.9f0)),
    ]
    power_scale_gradient = batched_logjoint_gradient_unconstrained(
        power_scale_model,
        power_scale_batch_params,
        (),
        power_scale_batch_constraints,
    )
    power_scale_gradient_cache = BatchedLogjointGradientCache(
        power_scale_model,
        power_scale_batch_params,
        (),
        power_scale_batch_constraints,
    )
    min_scale_batch_params = reshape(
        [min_scale_params[1] - 0.35, 0.0, min_scale_params[1] + 0.45],
        1,
        3,
    )
    min_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.5f0)),
        choicemap((:y, 0.9f0)),
    ]
    min_scale_gradient = batched_logjoint_gradient_unconstrained(
        min_scale_model,
        min_scale_batch_params,
        (),
        min_scale_batch_constraints,
    )
    min_scale_gradient_cache = BatchedLogjointGradientCache(
        min_scale_model,
        min_scale_batch_params,
        (),
        min_scale_batch_constraints,
    )
    max_scale_batch_params = reshape(
        [max_scale_params[1] - 0.45, -0.1, max_scale_params[1] + 0.35],
        1,
        3,
    )
    max_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.4f0)),
        choicemap((:y, 0.8f0)),
    ]
    max_scale_gradient = batched_logjoint_gradient_unconstrained(
        max_scale_model,
        max_scale_batch_params,
        (),
        max_scale_batch_constraints,
    )
    max_scale_gradient_cache = BatchedLogjointGradientCache(
        max_scale_model,
        max_scale_batch_params,
        (),
        max_scale_batch_constraints,
    )
    mod_scale_batch_params = reshape(
        [0.1, 0.6, 0.85],
        1,
        3,
    )
    mod_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.4f0)),
        choicemap((:y, 0.8f0)),
    ]
    mod_scale_gradient = batched_logjoint_gradient_unconstrained(
        mod_scale_model,
        mod_scale_batch_params,
        (),
        mod_scale_batch_constraints,
    )
    mod_scale_gradient_cache = BatchedLogjointGradientCache(
        mod_scale_model,
        mod_scale_batch_params,
        (),
        mod_scale_batch_constraints,
    )
    clamp_scale_batch_params = reshape(
        [clamp_scale_params[1] - 0.55, 0.05, clamp_scale_params[1] + 0.45],
        1,
        3,
    )
    clamp_scale_batch_constraints = [
        choicemap((:y, 0.25f0)),
        choicemap((:y, 0.5f0)),
        choicemap((:y, 0.85f0)),
    ]
    clamp_scale_gradient = batched_logjoint_gradient_unconstrained(
        clamp_scale_model,
        clamp_scale_batch_params,
        (),
        clamp_scale_batch_constraints,
    )
    clamp_scale_gradient_cache = BatchedLogjointGradientCache(
        clamp_scale_model,
        clamp_scale_batch_params,
        (),
        clamp_scale_batch_constraints,
    )
    unsupported_backend_params = reshape(Float64[-0.3, 0.2], 1, 2)
    unsupported_backend_constraints = [
        choicemap((:y, -0.1f0)),
        choicemap((:y, 0.4f0)),
    ]
    unsupported_backend_logjoint = batched_logjoint(
        unsupported_backend_model,
        unsupported_backend_params,
        (),
        unsupported_backend_constraints,
    )
    gaussian_backend_report = backend_report(gaussian_mean)
    gaussian_backend_plan = backend_execution_plan(gaussian_mean)
    iid_backend_report = backend_report(iid_model)
    iid_backend_plan = backend_execution_plan(iid_model)
    shifted_backend_report = backend_report(shifted_iid_model)
    shifted_backend_plan = backend_execution_plan(shifted_iid_model)
    offset_backend_report = backend_report(offset_iid_model)
    offset_backend_plan = backend_execution_plan(offset_iid_model)
    indexed_scale_plan = executionplan(indexed_scale_model)
    indexed_scale_backend_report = backend_report(indexed_scale_model)
    indexed_scale_backend_plan = backend_execution_plan(indexed_scale_model)
    coin_backend_report = backend_report(observed_coin)
    coin_backend_plan = backend_execution_plan(observed_coin)
    coin_workspace = UncertainTea.BatchedLogjointWorkspace(observed_coin)
    coin_batch_logjoint = batched_logjoint(
        observed_coin,
        zeros(0, 3),
        (),
        [
            choicemap((:y, true)),
            choicemap((:y, false)),
            choicemap((:y, true)),
        ],
    )
    deterministic_backend_report = backend_report(deterministic_scale)
    deterministic_backend_plan = backend_execution_plan(deterministic_scale)
    unsupported_backend_report = backend_report(unsupported_backend_model)
    short_warmup_schedule = UncertainTea._warmup_schedule(12)
    long_warmup_schedule = UncertainTea._warmup_schedule(150)

    @test parametercount(positive_step_spec.parameter_layout) == 1
    @test positive_step_spec.parameter_layout.slots[1].transform isa LogTransform
    @test positive_step_params[1] == Float64(positive_step_trace[:state => :sigma])
    @test positive_step_unconstrained[1] ≈ log(positive_step_params[1])
    @test positive_step_reconstrained ≈ positive_step_params
    @test logjoint_unconstrained(observed_positive_step, positive_step_unconstrained, (), choicemap((:y, 1.2f0))) ≈
        logjoint(observed_positive_step, positive_step_params, (), choicemap((:y, 1.2f0))) +
        positive_step_unconstrained[1] atol=1e-6
    @test logjoint(observed_positive_step, positive_step_params, (), choicemap((:y, 1.2f0))) ≈
        assess(
            observed_positive_step,
            (),
            choicemap((:state => :sigma, positive_step_trace[:state => :sigma]), (:y, 1.2f0)),
        ) atol=1e-6
    @test gaussian_batch_logjoint ≈ [
        logjoint(gaussian_mean, gaussian_batch_params[:, index], (), gaussian_batch_constraints[index]) for index in 1:3
    ] atol=1e-8
    @test gaussian_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(gaussian_mean, gaussian_batch_params[:, index], (), gaussian_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test batched_logjoint_gradient_unconstrained!(gaussian_batch_gradient_cache, gaussian_batch_params) ≈
        gaussian_batch_gradient atol=1e-8
    @test !isnothing(gaussian_batch_gradient_cache.backend_cache)
    @test isnothing(gaussian_batch_gradient_cache.flat_cache)
    @test isempty(gaussian_batch_gradient_cache.column_caches)
    @test gaussian_batch_gradient_shifted ≈ hcat([
        logjoint_gradient_unconstrained(gaussian_mean, gaussian_batch_params_shifted[:, index], (), gaussian_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    gaussian_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        gaussian_workspace,
        gaussian_batch_params,
        (),
        gaussian_batch_constraints,
    )
    gaussian_workspace_env = gaussian_workspace.batched_environment[]
    gaussian_workspace_totals = gaussian_workspace.batched_totals_buffer[]
    gaussian_workspace_observed = gaussian_workspace_env.observed_values
    @test gaussian_workspace_values ≈ gaussian_batch_logjoint atol=1e-8
    @test UncertainTea._logjoint_with_batched_backend!(
        gaussian_workspace,
        gaussian_batch_params_shifted,
        (),
        gaussian_batch_constraints,
    ) ≈ [
        logjoint(gaussian_mean, gaussian_batch_params_shifted[:, index], (), gaussian_batch_constraints[index]) for index in 1:3
    ] atol=1e-8
    @test gaussian_workspace.batched_environment[] === gaussian_workspace_env
    @test gaussian_workspace.batched_totals_buffer[] === gaussian_workspace_totals
    @test gaussian_workspace_env.observed_values === gaussian_workspace_observed
    @test iid_batch_logjoint ≈ [
        logjoint(iid_model, iid_batch_params[:, index], iid_batch_args[index], iid_batch_constraints[index]) for index in 1:2
    ] atol=1e-8
    @test batched_logjoint_gradient_unconstrained!(
        heterogeneous_iid_gradient_cache,
        iid_batch_params,
    ) ≈ hcat([
        logjoint_gradient_unconstrained(
            iid_model,
            iid_batch_params[:, index],
            heterogeneous_iid_args[index],
            iid_batch_constraints[index],
        ) for index in 1:2
    ]...) atol=1e-8
    @test isnothing(heterogeneous_iid_gradient_cache.backend_cache)
    @test !isnothing(heterogeneous_iid_gradient_cache.flat_cache)
    @test isempty(heterogeneous_iid_gradient_cache.column_caches)
    @test iid_shared_batch_logjoint ≈ [
        logjoint(iid_model, iid_shared_batch_params[:, index], (3,), iid_shared_batch_constraints[index]) for index in 1:2
    ] atol=1e-8
    @test !isnothing(iid_shared_gradient_cache.backend_cache)
    @test isnothing(iid_shared_gradient_cache.flat_cache)
    @test isempty(iid_shared_gradient_cache.column_caches)
    @test iid_shared_single_constraint_logjoint ≈ [
        logjoint(iid_model, iid_shared_batch_params[:, index], (3,), iid_shared_batch_constraints[1]) for index in 1:2
    ] atol=1e-8
    iid_shared_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        iid_shared_workspace,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints,
    )
    iid_shared_workspace_env = iid_shared_workspace.batched_environment[]
    iid_shared_iterable_scratch = iid_shared_workspace_env.index_scratch[1]
    iid_shared_observed_values = iid_shared_workspace_env.observed_values
    @test iid_shared_workspace_values ≈ iid_shared_batch_logjoint atol=1e-8
    @test UncertainTea._logjoint_with_batched_backend!(
        iid_shared_workspace,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints[1],
    ) ≈ iid_shared_single_constraint_logjoint atol=1e-8
    @test !isempty(iid_shared_workspace_env.index_scratch)
    @test UncertainTea._logjoint_with_batched_backend!(
        iid_shared_workspace,
        iid_shared_batch_params .+ 0.1,
        (3,),
        iid_shared_batch_constraints,
    ) ≈ [
        logjoint(iid_model, (iid_shared_batch_params .+ 0.1)[:, index], (3,), iid_shared_batch_constraints[index]) for
        index in 1:2
    ] atol=1e-8
    @test iid_shared_workspace.batched_environment[] === iid_shared_workspace_env
    @test iid_shared_workspace_env.index_scratch[1] === iid_shared_iterable_scratch
    @test iid_shared_workspace_env.observed_values === iid_shared_observed_values
    @test shifted_batch_logjoint ≈ [
        logjoint(shifted_iid_model, shifted_batch_params[:, index], (3,), shifted_batch_constraints[index]) for index in 1:2
    ] atol=1e-8
    @test offset_batch_logjoint ≈ [
        logjoint(offset_iid_model, offset_batch_params[:, index], offset_batch_args[index], offset_batch_constraints[index]) for
        index in 1:2
    ] atol=1e-8
    offset_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        offset_workspace,
        offset_batch_params,
        offset_batch_args,
        offset_batch_constraints,
    )
    offset_workspace_env = offset_workspace.batched_environment[]
    offset_workspace_scratch = offset_workspace_env.index_scratch[1]
    @test offset_workspace_values ≈ offset_batch_logjoint atol=1e-8
    @test !isempty(offset_workspace_env.index_scratch)
    @test UncertainTea._logjoint_with_batched_backend!(
        offset_workspace,
        offset_batch_params .+ 0.05,
        offset_batch_args,
        offset_batch_constraints,
    ) ≈ [
        logjoint(
            offset_iid_model,
            (offset_batch_params .+ 0.05)[:, index],
            offset_batch_args[index],
            offset_batch_constraints[index],
        ) for index in 1:2
    ] atol=1e-8
    @test offset_workspace.batched_environment[] === offset_workspace_env
    @test offset_workspace_env.index_scratch[1] === offset_workspace_scratch
    @test indexed_scale_batch_logjoint ≈ [
        logjoint(
            indexed_scale_model,
            indexed_scale_batch_params[:, index],
            (3,),
            indexed_scale_batch_constraints[index],
        ) for index in 1:2
    ] atol=1e-8
    deterministic_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        deterministic_workspace,
        deterministic_batch_params,
        (),
        deterministic_batch_constraints,
    )
    deterministic_workspace_env = deterministic_workspace.batched_environment[]
    deterministic_workspace_scratch = deterministic_workspace_env.numeric_scratch[1]
    @test deterministic_batch_logjoint ≈ [
        logjoint(
            deterministic_scale,
            deterministic_batch_params[:, index],
            (),
            deterministic_batch_constraints[index],
        ) for index in 1:3
    ] atol=1e-8
    @test deterministic_workspace_values ≈ deterministic_batch_logjoint atol=1e-8
    @test !isempty(deterministic_workspace_env.numeric_scratch)
    @test UncertainTea._logjoint_with_batched_backend!(
        deterministic_workspace,
        deterministic_batch_params .+ [0.05; -0.02],
        (),
        deterministic_batch_constraints,
    ) ≈ [
        logjoint(
            deterministic_scale,
            (deterministic_batch_params .+ [0.05; -0.02])[:, index],
            (),
            deterministic_batch_constraints[index],
        ) for index in 1:3
    ] atol=1e-8
    @test deterministic_workspace.batched_environment[] === deterministic_workspace_env
    @test deterministic_workspace_env.numeric_scratch[1] === deterministic_workspace_scratch
    @test positive_batch_logjoint ≈ [
        logjoint_unconstrained(observed_positive_step, positive_batch_unconstrained[:, index], (), positive_batch_constraints[index]) for
        index in 1:3
    ] atol=1e-8
    @test positive_batch_gradient ≈ hcat([
        logjoint_gradient_unconstrained(
            observed_positive_step,
            positive_batch_unconstrained[:, index],
            (),
            positive_batch_constraints[index],
        ) for index in 1:3
    ]...) atol=1e-8
    @test !isnothing(positive_batch_gradient_cache.backend_cache)
    @test isnothing(positive_batch_gradient_cache.flat_cache)
    @test isempty(positive_batch_gradient_cache.column_caches)
    gaussian_combined_values = fill(-1.0, 3)
    gaussian_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        gaussian_combined_values,
        gaussian_batch_gradient_cache,
        gaussian_batch_params,
    )[2]
    @test gaussian_combined_values ≈ gaussian_batch_logjoint atol=1e-8
    @test gaussian_combined_gradient === gaussian_batch_gradient_cache.gradient_buffer
    @test gaussian_combined_gradient ≈ gaussian_batch_gradient atol=1e-8
    @test abs_scale_gradient ≈ hcat([
        logjoint_gradient_unconstrained(abs_scale_model, abs_scale_batch_params[:, index], (), abs_scale_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test !isnothing(abs_scale_gradient_cache.backend_cache)
    @test isnothing(abs_scale_gradient_cache.flat_cache)
    @test isempty(abs_scale_gradient_cache.column_caches)
    abs_scale_combined_values = fill(-1.0, 3)
    abs_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        abs_scale_combined_values,
        abs_scale_gradient_cache,
        abs_scale_batch_params,
    )[2]
    @test abs_scale_combined_values ≈ [
        logjoint_unconstrained(abs_scale_model, abs_scale_batch_params[:, index], (), abs_scale_batch_constraints[index]) for
        index in 1:3
    ] atol=1e-8
    @test abs_scale_combined_gradient === abs_scale_gradient_cache.gradient_buffer
    @test abs_scale_combined_gradient ≈ abs_scale_gradient atol=1e-8
    @test power_scale_gradient ≈ hcat([
        logjoint_gradient_unconstrained(power_scale_model, power_scale_batch_params[:, index], (), power_scale_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test !isnothing(power_scale_gradient_cache.backend_cache)
    @test isnothing(power_scale_gradient_cache.flat_cache)
    @test isempty(power_scale_gradient_cache.column_caches)
    @test min_scale_gradient ≈ hcat([
        logjoint_gradient_unconstrained(min_scale_model, min_scale_batch_params[:, index], (), min_scale_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test !isnothing(min_scale_gradient_cache.backend_cache)
    @test isnothing(min_scale_gradient_cache.flat_cache)
    @test isempty(min_scale_gradient_cache.column_caches)
    min_scale_combined_values = fill(-1.0, 3)
    min_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        min_scale_combined_values,
        min_scale_gradient_cache,
        min_scale_batch_params,
    )[2]
    @test min_scale_combined_values ≈ [
        logjoint_unconstrained(min_scale_model, min_scale_batch_params[:, index], (), min_scale_batch_constraints[index]) for
        index in 1:3
    ] atol=1e-8
    @test min_scale_combined_gradient === min_scale_gradient_cache.gradient_buffer
    @test min_scale_combined_gradient ≈ min_scale_gradient atol=1e-8
    @test max_scale_gradient ≈ hcat([
        logjoint_gradient_unconstrained(max_scale_model, max_scale_batch_params[:, index], (), max_scale_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test !isnothing(max_scale_gradient_cache.backend_cache)
    @test isnothing(max_scale_gradient_cache.flat_cache)
    @test isempty(max_scale_gradient_cache.column_caches)
    max_scale_combined_values = fill(-1.0, 3)
    max_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        max_scale_combined_values,
        max_scale_gradient_cache,
        max_scale_batch_params,
    )[2]
    @test max_scale_combined_values ≈ [
        logjoint_unconstrained(max_scale_model, max_scale_batch_params[:, index], (), max_scale_batch_constraints[index]) for
        index in 1:3
    ] atol=1e-8
    @test max_scale_combined_gradient === max_scale_gradient_cache.gradient_buffer
    @test max_scale_combined_gradient ≈ max_scale_gradient atol=1e-8
