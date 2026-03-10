using Test
using Random
using UncertainTea

@testset "UncertainTea" begin
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
    gaussian_batch_params_shifted = gaussian_batch_params .+ 0.15
    gaussian_batch_gradient_shifted = batched_logjoint_gradient_unconstrained(
        gaussian_batch_gradient_cache,
        gaussian_batch_params_shifted,
    )
    iid_batch_params = reshape(Float64[-0.2, 0.4], 1, 2)
    iid_batch_args = [(2,), (3,)]
    iid_batch_constraints = [
        choicemap((:y => 1, 0.1f0), (:y => 2, -0.2f0)),
        choicemap((:y => 1, 0.5f0), (:y => 2, 0.2f0), (:y => 3, -0.1f0)),
    ]
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
    @test gaussian_batch_gradient_shifted ≈ hcat([
        logjoint_gradient_unconstrained(gaussian_mean, gaussian_batch_params_shifted[:, index], (), gaussian_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test iid_batch_logjoint ≈ [
        logjoint(iid_model, iid_batch_params[:, index], iid_batch_args[index], iid_batch_constraints[index]) for index in 1:2
    ] atol=1e-8
    @test iid_shared_batch_logjoint ≈ [
        logjoint(iid_model, iid_shared_batch_params[:, index], (3,), iid_shared_batch_constraints[index]) for index in 1:2
    ] atol=1e-8
    @test shifted_batch_logjoint ≈ [
        logjoint(shifted_iid_model, shifted_batch_params[:, index], (3,), shifted_batch_constraints[index]) for index in 1:2
    ] atol=1e-8
    @test offset_batch_logjoint ≈ [
        logjoint(offset_iid_model, offset_batch_params[:, index], offset_batch_args[index], offset_batch_constraints[index]) for
        index in 1:2
    ] atol=1e-8
    @test indexed_scale_batch_logjoint ≈ [
        logjoint(
            indexed_scale_model,
            indexed_scale_batch_params[:, index],
            (3,),
            indexed_scale_batch_constraints[index],
        ) for index in 1:2
    ] atol=1e-8
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
    @test gaussian_backend_report.supported
    @test gaussian_backend_report.target == :gpu
    @test isempty(gaussian_backend_report.issues)
    @test gaussian_backend_plan.target == :gpu
    @test length(gaussian_backend_plan.steps) == 2
    @test gaussian_backend_plan.steps[1] isa UncertainTea.BackendChoicePlanStep
    @test gaussian_backend_plan.steps[1] isa UncertainTea.BackendNormalChoicePlanStep
    @test gaussian_backend_plan.numeric_slots == BitVector([true])
    @test gaussian_backend_plan.index_slots == BitVector([false])
    @test gaussian_backend_plan.generic_slots == BitVector([false])
    @test iid_backend_report.supported
    @test count(identity, iid_backend_plan.numeric_slots) == 1
    @test count(identity, iid_backend_plan.index_slots) == 2
    @test !any(iid_backend_plan.generic_slots)
    @test iid_backend_plan.index_slots[1]
    @test iid_backend_plan.numeric_slots[2]
    @test iid_backend_plan.index_slots[3]
    @test shifted_backend_report.supported
    @test shifted_backend_plan.numeric_slots == iid_backend_plan.numeric_slots
    @test shifted_backend_plan.index_slots == iid_backend_plan.index_slots
    @test offset_backend_report.supported
    @test indexed_scale_backend_report.supported
    @test indexed_scale_backend_plan.numeric_slots[indexed_scale_plan.environment_layout.slot_by_symbol[:mu]]
    @test indexed_scale_backend_plan.index_slots[indexed_scale_plan.environment_layout.slot_by_symbol[:n]]
    @test indexed_scale_backend_plan.index_slots[indexed_scale_plan.environment_layout.slot_by_symbol[:i]]
    @test !any(indexed_scale_backend_plan.generic_slots)
    @test !isnothing(UncertainTea._backend_loop_observed_choice(iid_backend_plan.steps[2]))
    @test !isnothing(UncertainTea._backend_loop_observed_choice(shifted_backend_plan.steps[2]))
    @test isnothing(UncertainTea._backend_loop_observed_choice(offset_backend_plan.steps[2]))
    @test isnothing(UncertainTea._backend_loop_observed_choice(backend_execution_plan(chain_model).steps[3]))
    @test coin_backend_report.supported
    @test coin_backend_plan.steps[1] isa UncertainTea.BackendBernoulliChoicePlanStep
    @test isempty(coin_backend_plan.numeric_slots)
    @test isempty(coin_backend_plan.index_slots)
    @test isempty(coin_backend_plan.generic_slots)
    @test coin_batch_logjoint ≈ [
        logjoint(observed_coin, Float64[], (), choicemap((:y, true))),
        logjoint(observed_coin, Float64[], (), choicemap((:y, false))),
        logjoint(observed_coin, Float64[], (), choicemap((:y, true))),
    ] atol=1e-8
    @test deterministic_backend_report.supported
    @test length(deterministic_backend_plan.steps) == 4
    @test deterministic_backend_plan.steps[3] isa UncertainTea.BackendDeterministicPlanStep
    @test all(deterministic_backend_plan.numeric_slots)
    @test !any(deterministic_backend_plan.index_slots)
    @test !any(deterministic_backend_plan.generic_slots)
    @test !unsupported_backend_report.supported
    @test any(occursin("sin", issue) for issue in unsupported_backend_report.issues)
    @test unsupported_backend_logjoint ≈ [
        logjoint(
            unsupported_backend_model,
            unsupported_backend_params[:, index],
            (),
            unsupported_backend_constraints[index],
        ) for index in 1:2
    ] atol=1e-8
    @test short_warmup_schedule.initial_buffer == 5
    @test short_warmup_schedule.slow_window_ends == [12]
    @test short_warmup_schedule.terminal_buffer == 0
    @test long_warmup_schedule.initial_buffer == 15
    @test long_warmup_schedule.slow_window_ends == [40, 90, 135]
    @test long_warmup_schedule.terminal_buffer == 15

    gaussian_chain = hmc(
        gaussian_mean,
        (),
        constraints;
        num_samples=250,
        num_warmup=150,
        step_size=0.25,
        num_leapfrog_steps=8,
        rng=MersenneTwister(29),
    )
    gaussian_baseline_chain = hmc(
        gaussian_mean,
        (),
        constraints;
        num_samples=25,
        num_warmup=0,
        step_size=0.25,
        num_leapfrog_steps=8,
        rng=MersenneTwister(22),
    )
    gaussian_large_step_chain = hmc(
        gaussian_mean,
        (),
        constraints;
        num_samples=20,
        num_warmup=0,
        step_size=16.0,
        num_leapfrog_steps=4,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        find_reasonable_step_size=true,
        rng=MersenneTwister(23),
    )
    gaussian_small_step_chain = hmc(
        gaussian_mean,
        (),
        constraints;
        num_samples=20,
        num_warmup=0,
        step_size=1e-6,
        num_leapfrog_steps=4,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        find_reasonable_step_size=true,
        rng=MersenneTwister(24),
    )
    gaussian_divergent_chain = hmc(
        gaussian_mean,
        (),
        constraints;
        num_samples=20,
        num_warmup=0,
        step_size=2.0,
        num_leapfrog_steps=8,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        find_reasonable_step_size=false,
        divergence_threshold=1.0,
        rng=MersenneTwister(31),
    )
    gaussian_windowed_mass_chain = hmc(
        gaussian_mean,
        (),
        constraints;
        num_samples=20,
        num_warmup=20,
        step_size=0.2,
        num_leapfrog_steps=6,
        adapt_step_size=false,
        adapt_mass_matrix=true,
        mass_matrix_min_samples=3,
        rng=MersenneTwister(36),
    )
    gaussian_multichain = hmc_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=60,
        num_warmup=30,
        step_size=0.2,
        num_leapfrog_steps=6,
        rng=MersenneTwister(45),
    )
    gaussian_multichain_replay = hmc_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=60,
        num_warmup=30,
        step_size=0.2,
        num_leapfrog_steps=6,
        rng=MersenneTwister(45),
    )
    positive_multichain = hmc_chains(
        observed_positive_step,
        (),
        choicemap((:y, 1.2f0));
        num_chains=3,
        num_samples=20,
        num_warmup=10,
        step_size=0.1,
        num_leapfrog_steps=6,
        initial_params=reshape(
            [
                positive_step_unconstrained[1] - 0.2,
                positive_step_unconstrained[1],
                positive_step_unconstrained[1] + 0.2,
            ],
            1,
            3,
        ),
        rng=MersenneTwister(41),
    )
    gaussian_batched_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=40,
        num_warmup=20,
        step_size=0.18,
        num_leapfrog_steps=6,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(52),
    )
    gaussian_batched_chain_replay = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=40,
        num_warmup=20,
        step_size=0.18,
        num_leapfrog_steps=6,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(52),
    )
    iid_batched_chain = batched_hmc(
        iid_model,
        iid_batch_args,
        iid_batch_constraints;
        num_chains=2,
        num_samples=24,
        num_warmup=12,
        step_size=0.12,
        num_leapfrog_steps=5,
        initial_params=iid_batch_params,
        rng=MersenneTwister(53),
    )
    positive_batched_chain = batched_hmc(
        observed_positive_step,
        (),
        positive_batch_constraints;
        num_chains=3,
        num_samples=30,
        num_warmup=15,
        step_size=0.1,
        num_leapfrog_steps=6,
        initial_params=positive_batch_unconstrained,
        rng=MersenneTwister(54),
    )
    gaussian_rhat = rhat(gaussian_multichain)
    gaussian_rhat_unconstrained = rhat(gaussian_multichain; space=:unconstrained)
    gaussian_ess = ess(gaussian_multichain)
    gaussian_ess_unconstrained = ess(gaussian_multichain; space=:unconstrained)
    positive_rhat = rhat(positive_multichain)
    positive_ess = ess(positive_multichain)
    gaussian_batched_rhat = rhat(gaussian_batched_chain)
    gaussian_summary = summarize(gaussian_multichain)
    gaussian_summary_unconstrained = summarize(gaussian_multichain; space=:unconstrained)
    positive_summary = summarize(positive_multichain; quantiles=(0.25, 0.5, 0.75))
    gaussian_pooled_draws = vcat([vec(chain.constrained_samples[1, :]) for chain in gaussian_multichain]...)
    gaussian_pooled_mean = sum(gaussian_pooled_draws) / length(gaussian_pooled_draws)
    gaussian_chain_mean = sum(gaussian_chain.constrained_samples[1, :]) / size(gaussian_chain.constrained_samples, 2)

    @test length(gaussian_chain) == 250
    @test size(gaussian_chain.unconstrained_samples) == (1, 250)
    @test size(gaussian_chain.constrained_samples) == (1, 250)
    @test all(isfinite, gaussian_chain.unconstrained_samples)
    @test all(isfinite, gaussian_chain.constrained_samples)
    @test all(isfinite, gaussian_chain.logjoint_values)
    @test size(gaussian_chain.energies) == (250,)
    @test size(gaussian_chain.energy_errors) == (250,)
    @test size(gaussian_chain.divergent) == (250,)
    @test all(isfinite, gaussian_chain.energies)
    @test all(isfinite, gaussian_chain.energy_errors[.!gaussian_chain.divergent])
    @test 0.0 <= acceptancerate(gaussian_chain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_chain) <= 1.0
    @test any(gaussian_chain.accepted)
    @test gaussian_chain.step_size > 0
    @test gaussian_chain.mass_matrix[1] > 0
    @test gaussian_chain.target_accept == 0.8
    @test gaussian_baseline_chain.step_size == 0.25
    @test gaussian_baseline_chain.mass_matrix == [1.0]
    @test 0 < gaussian_large_step_chain.step_size < 16.0
    @test gaussian_small_step_chain.step_size > 1e-6
    @test all(isfinite, gaussian_large_step_chain.logjoint_values)
    @test all(isfinite, gaussian_small_step_chain.logjoint_values)
    @test gaussian_windowed_mass_chain.step_size == 0.2
    @test gaussian_windowed_mass_chain.mass_matrix[1] != 1.0
    @test nchains(gaussian_multichain) == 3
    @test numsamples(gaussian_multichain) == 60
    @test gaussian_multichain[1] isa HMCChain
    @test 0.0 <= acceptancerate(gaussian_multichain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_multichain) <= 1.0
    @test length(gaussian_rhat) == 1
    @test length(gaussian_ess) == 1
    @test isfinite(gaussian_rhat[1])
    @test 1.0 <= gaussian_rhat[1] < 1.1
    @test gaussian_rhat ≈ gaussian_rhat_unconstrained atol=1e-8
    @test 1.0 <= gaussian_ess[1] <= nchains(gaussian_multichain) * numsamples(gaussian_multichain)
    @test gaussian_ess ≈ gaussian_ess_unconstrained atol=1e-8
    @test length(gaussian_summary) == 1
    @test gaussian_summary.space == :constrained
    @test gaussian_summary.quantile_probs == [0.05, 0.5, 0.95]
    @test gaussian_summary[1].binding == :mu
    @test gaussian_summary[1].address == :mu
    @test gaussian_summary[1].mean ≈ gaussian_pooled_mean atol=1e-8
    @test gaussian_summary[1].sd > 0
    @test gaussian_summary[1].quantiles[1] <= gaussian_summary[1].quantiles[2] <= gaussian_summary[1].quantiles[3]
    @test gaussian_summary[1].rhat == gaussian_rhat[1]
    @test gaussian_summary[1].ess == gaussian_ess[1]
    @test gaussian_summary[1].mean ≈ gaussian_summary_unconstrained[1].mean atol=1e-8
    @test gaussian_multichain[1].unconstrained_samples[:, 1] ==
        gaussian_multichain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_multichain[1].accepted == gaussian_multichain_replay[1].accepted
    @test gaussian_multichain[1].unconstrained_samples[:, 1] != gaussian_multichain[2].unconstrained_samples[:, 1]
    @test nchains(gaussian_batched_chain) == 3
    @test numsamples(gaussian_batched_chain) == 40
    @test gaussian_batched_chain.args == ()
    @test length(gaussian_batched_chain.constraints) == 3
    @test gaussian_batched_chain[1].constraints[:y] == gaussian_batch_constraints[1][:y]
    @test gaussian_batched_chain[2].constraints[:y] == gaussian_batch_constraints[2][:y]
    @test all(chain.step_size == 0.18 for chain in gaussian_batched_chain)
    @test all(chain.mass_matrix == [1.0] for chain in gaussian_batched_chain)
    @test 0.0 <= acceptancerate(gaussian_batched_chain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_batched_chain) <= 1.0
    @test length(gaussian_batched_rhat) == 1
    @test isfinite(gaussian_batched_rhat[1])
    @test gaussian_batched_chain[1].unconstrained_samples[:, 1] ==
        gaussian_batched_chain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_batched_chain[1].accepted == gaussian_batched_chain_replay[1].accepted
    @test nchains(iid_batched_chain) == 2
    @test numsamples(iid_batched_chain) == 24
    @test iid_batched_chain.args == iid_batch_args
    @test length(iid_batched_chain.constraints) == 2
    @test iid_batched_chain[1].args == iid_batch_args[1]
    @test iid_batched_chain[2].constraints[:y => 3] == iid_batch_constraints[2][:y => 3]
    @test all(all(isfinite, chain.logjoint_values) for chain in iid_batched_chain)
    @test all(gaussian_divergent_chain.divergent)
    @test divergencerate(gaussian_divergent_chain) == 1.0
    @test all(isfinite, gaussian_divergent_chain.energies)
    @test maximum(abs, gaussian_divergent_chain.energy_errors) > 1.0
    @test !(isapprox(gaussian_chain.step_size, gaussian_baseline_chain.step_size; atol=1e-8) &&
        isapprox(gaussian_chain.mass_matrix[1], gaussian_baseline_chain.mass_matrix[1]; atol=1e-8))
    @test abs(gaussian_chain_mean - 0.15) < 0.2
    @test gaussian_chain.logjoint_values[1] ≈
        logjoint_unconstrained(gaussian_mean, gaussian_chain.unconstrained_samples[:, 1], (), constraints) atol=1e-6

    positive_chain = hmc(
        observed_positive_step,
        (),
        choicemap((:y, 1.2f0));
        num_samples=120,
        num_warmup=80,
        step_size=0.12,
        num_leapfrog_steps=10,
        initial_params=positive_step_unconstrained,
        rng=MersenneTwister(21),
    )

    @test length(positive_chain) == 120
    @test size(positive_chain.unconstrained_samples) == (1, 120)
    @test size(positive_chain.constrained_samples) == (1, 120)
    @test all(x -> x > 0, positive_chain.constrained_samples)
    @test all(isfinite, positive_chain.energies)
    @test all(isfinite, positive_chain.energy_errors[.!positive_chain.divergent])
    @test any(positive_chain.accepted)
    @test positive_chain.step_size > 0
    @test positive_chain.mass_matrix[1] > 0
    @test nchains(positive_multichain) == 3
    @test length(positive_rhat) == 1
    @test length(positive_ess) == 1
    @test isfinite(positive_rhat[1])
    @test positive_rhat[1] >= 1.0
    @test 1.0 <= positive_ess[1] <= nchains(positive_multichain) * numsamples(positive_multichain)
    @test length(positive_summary) == 1
    @test positive_summary.quantile_probs == [0.25, 0.5, 0.75]
    @test positive_summary[1].binding isa Symbol
    @test positive_summary[1].address == (:state, :sigma)
    @test positive_summary[1].mean > 0
    @test positive_summary[1].quantiles[1] <= positive_summary[1].quantiles[2] <= positive_summary[1].quantiles[3]
    @test positive_summary[1].rhat == positive_rhat[1]
    @test positive_summary[1].ess == positive_ess[1]
    @test all(all(x -> x > 0, chain.constrained_samples) for chain in positive_multichain)
    @test nchains(positive_batched_chain) == 3
    @test numsamples(positive_batched_chain) == 30
    @test all(all(x -> x > 0, chain.constrained_samples) for chain in positive_batched_chain)
    @test all(chain.mass_matrix == [1.0] for chain in positive_batched_chain)
    @test parameterchoicemap(observed_positive_step, positive_chain.constrained_samples[:, 1])[:state => :sigma] ==
        positive_chain.constrained_samples[1, 1]

    @tea static function observed_only()
        {:y} ~ bernoulli(0.5f0)
        return nothing
    end

    @test_throws DimensionMismatch parameterchoicemap(gaussian_mean, Float64[])
    @test_throws DimensionMismatch logjoint(iid_model, params2, (), repeated)
    @test_throws ArgumentError hmc(observed_only, (), choicemap((:y, true)); num_samples=10)
    @test_throws ArgumentError hmc(gaussian_mean, (), constraints; num_samples=10, divergence_threshold=0.0)
    @test_throws ArgumentError hmc_chains(gaussian_mean, (), constraints; num_chains=0, num_samples=10)
    @test_throws ArgumentError batched_hmc(gaussian_mean, (), gaussian_batch_constraints; num_chains=0, num_samples=10)
    @test_throws DimensionMismatch batched_logjoint(gaussian_mean, zeros(2, 3), (), gaussian_batch_constraints)
    @test_throws DimensionMismatch batched_logjoint(gaussian_mean, gaussian_batch_params, (), gaussian_batch_constraints[1:2])
    @test_throws ArgumentError batched_logjoint(gaussian_mean, gaussian_batch_params, [1, 2, 3], gaussian_batch_constraints)
    @test_throws DimensionMismatch batched_logjoint_gradient_unconstrained(gaussian_batch_gradient_cache, zeros(1, 2))
    @test_throws ArgumentError backend_execution_plan(unsupported_backend_model)
    @test_throws DimensionMismatch batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints[1:2];
        num_chains=3,
        num_samples=10,
    )
    @test_throws DimensionMismatch batched_hmc(
        iid_model,
        iid_batch_args[1:1],
        iid_batch_constraints;
        num_chains=2,
        num_samples=10,
    )
    @test_throws ArgumentError rhat(HMCChains(gaussian_mean, (), constraints, [gaussian_baseline_chain]))
    @test_throws ArgumentError ess(gaussian_multichain; space=:energy)
    @test_throws ArgumentError summarize(gaussian_multichain; quantiles=())
    @test_throws ArgumentError summarize(gaussian_multichain; quantiles=(-0.1, 0.5, 0.9))
    @test_throws DimensionMismatch hmc_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=10,
        initial_params=zeros(1, 2),
    )
    @test_throws DimensionMismatch batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=10,
        initial_params=zeros(1, 2),
    )
end
