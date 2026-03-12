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
    @test mod_scale_gradient ≈ hcat([
        logjoint_gradient_unconstrained(mod_scale_model, mod_scale_batch_params[:, index], (), mod_scale_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test !isnothing(mod_scale_gradient_cache.backend_cache)
    @test isnothing(mod_scale_gradient_cache.flat_cache)
    @test isempty(mod_scale_gradient_cache.column_caches)
    mod_scale_combined_values = fill(-1.0, 3)
    mod_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        mod_scale_combined_values,
        mod_scale_gradient_cache,
        mod_scale_batch_params,
    )[2]
    @test mod_scale_combined_values ≈ [
        logjoint_unconstrained(mod_scale_model, mod_scale_batch_params[:, index], (), mod_scale_batch_constraints[index]) for
        index in 1:3
    ] atol=1e-8
    @test mod_scale_combined_gradient === mod_scale_gradient_cache.gradient_buffer
    @test mod_scale_combined_gradient ≈ mod_scale_gradient atol=1e-8
    @test clamp_scale_gradient ≈ hcat([
        logjoint_gradient_unconstrained(clamp_scale_model, clamp_scale_batch_params[:, index], (), clamp_scale_batch_constraints[index]) for
        index in 1:3
    ]...) atol=1e-8
    @test !isnothing(clamp_scale_gradient_cache.backend_cache)
    @test isnothing(clamp_scale_gradient_cache.flat_cache)
    @test isempty(clamp_scale_gradient_cache.column_caches)
    clamp_scale_combined_values = fill(-1.0, 3)
    clamp_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        clamp_scale_combined_values,
        clamp_scale_gradient_cache,
        clamp_scale_batch_params,
    )[2]
    @test clamp_scale_combined_values ≈ [
        logjoint_unconstrained(clamp_scale_model, clamp_scale_batch_params[:, index], (), clamp_scale_batch_constraints[index]) for
        index in 1:3
    ] atol=1e-8
    @test clamp_scale_combined_gradient === clamp_scale_gradient_cache.gradient_buffer
    @test clamp_scale_combined_gradient ≈ clamp_scale_gradient atol=1e-8
    positive_workspace = UncertainTea.BatchedLogjointWorkspace(observed_positive_step)
    positive_workspace_values = UncertainTea._logjoint_unconstrained_batched_backend!(
        observed_positive_step,
        positive_workspace,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    )
    positive_workspace_constrained = positive_workspace.batched_constrained_buffer[]
    positive_workspace_logabsdet = positive_workspace.batched_logabsdet_buffer[]
    positive_workspace_observed = positive_workspace.batched_environment[].observed_values
    @test positive_workspace_values ≈ positive_batch_logjoint atol=1e-8
    @test UncertainTea._logjoint_unconstrained_batched_backend!(
        observed_positive_step,
        positive_workspace,
        positive_batch_unconstrained .+ 0.05,
        (),
        positive_batch_constraints,
    ) ≈ [
        logjoint_unconstrained(
            observed_positive_step,
            (positive_batch_unconstrained .+ 0.05)[:, index],
            (),
            positive_batch_constraints[index],
        ) for index in 1:3
    ] atol=1e-8
    @test positive_workspace.batched_constrained_buffer[] === positive_workspace_constrained
    @test positive_workspace.batched_logabsdet_buffer[] === positive_workspace_logabsdet
    @test positive_workspace.batched_environment[].observed_values === positive_workspace_observed
    positive_destination = fill(-1.0, 3)
    @test UncertainTea._batched_logjoint_unconstrained_with_workspace!(
        positive_destination,
        observed_positive_step,
        positive_workspace,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    ) === positive_destination
    @test positive_destination ≈ positive_batch_logjoint atol=1e-8
    positive_reconstrained = similar(positive_step_unconstrained)
    @test UncertainTea._transform_to_constrained!(
        positive_reconstrained,
        observed_positive_step,
        positive_step_unconstrained,
    ) === positive_reconstrained
    @test positive_reconstrained ≈ transform_to_constrained(
        observed_positive_step,
        positive_step_unconstrained,
    ) atol=1e-8
    gaussian_hmc_workspace = UncertainTea.BatchedHMCWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        gaussian_batch_constraints,
        [1.0],
    )
    UncertainTea._sample_batched_momentum!(
        gaussian_hmc_workspace.momentum,
        MersenneTwister(91),
        gaussian_hmc_workspace.sqrt_inverse_mass_matrix,
    )
    gaussian_hmc_current_logjoint = Vector{Float64}(undef, 3)
    UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        gaussian_hmc_current_logjoint,
        gaussian_hmc_workspace.gradient_cache,
        gaussian_batch_params,
    )
    gaussian_hmc_proposal = UncertainTea._batched_leapfrog!(
        gaussian_hmc_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_hmc_workspace.current_gradient,
        [1.0],
        (),
        gaussian_batch_constraints,
        0.1,
        2,
    )
    @test gaussian_hmc_proposal[1] === gaussian_hmc_workspace.proposal_position
    @test gaussian_hmc_proposal[2] === gaussian_hmc_workspace.proposal_momentum
    @test gaussian_hmc_proposal[3] === gaussian_hmc_workspace.proposed_logjoint
    @test gaussian_hmc_proposal[4] === gaussian_hmc_workspace.proposal_gradient
    @test gaussian_hmc_proposal[5] === gaussian_hmc_workspace.valid
    UncertainTea._sample_batched_momentum!(
        gaussian_hmc_workspace.momentum,
        MersenneTwister(92),
        gaussian_hmc_workspace.sqrt_inverse_mass_matrix,
    )
    gaussian_hmc_proposal_replay = UncertainTea._batched_leapfrog!(
        gaussian_hmc_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_hmc_workspace.current_gradient,
        [1.0],
        (),
        gaussian_batch_constraints,
        0.1,
        2,
    )
    @test gaussian_hmc_proposal_replay[1] === gaussian_hmc_proposal[1]
    @test gaussian_hmc_proposal_replay[2] === gaussian_hmc_proposal[2]
    @test gaussian_hmc_proposal_replay[3] === gaussian_hmc_proposal[3]
    @test gaussian_hmc_proposal_replay[4] === gaussian_hmc_proposal[4]
    @test gaussian_hmc_proposal_replay[5] === gaussian_hmc_proposal[5]
    gaussian_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        gaussian_batch_constraints,
    )
    gaussian_shared_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    gaussian_nuts_state = UncertainTea._batched_nuts_state(
        gaussian_nuts_workspace.proposal_position,
        gaussian_nuts_workspace.proposal_momentum,
        gaussian_nuts_workspace.proposed_logjoint,
        gaussian_nuts_workspace.proposal_gradient,
        1,
    )
    @test parent(gaussian_nuts_state.position) === gaussian_nuts_workspace.proposal_position
    @test parent(gaussian_nuts_state.momentum) === gaussian_nuts_workspace.proposal_momentum
    @test parent(gaussian_nuts_state.gradient) === gaussian_nuts_workspace.proposal_gradient
    @test length(gaussian_nuts_workspace.column_tree_workspaces) == 3
    @test gaussian_nuts_workspace.column_tree_workspaces[1] isa UncertainTea.NUTSSubtreeWorkspace
    @test length(gaussian_nuts_workspace.column_tree_workspaces[1].current.position) == 1
    @test length(gaussian_nuts_workspace.column_tree_workspaces[1].left.position) == 1
    @test length(gaussian_nuts_workspace.column_tree_workspaces[1].right.position) == 1
    @test length(gaussian_nuts_workspace.column_tree_workspaces[1].proposal.position) == 1
    @test gaussian_nuts_workspace.column_tree_workspaces[1].summary isa UncertainTea.NUTSSubtreeMetadataState
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].current.position) === gaussian_nuts_workspace.tree_current_position
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].current.momentum) === gaussian_nuts_workspace.tree_current_momentum
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].current.gradient) === gaussian_nuts_workspace.tree_current_gradient
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].next.position) === gaussian_nuts_workspace.tree_next_position
    @test parent(gaussian_nuts_workspace.column_gradient_caches[1].buffer) === gaussian_nuts_workspace.tree_next_gradient
    @test gaussian_shared_nuts_workspace.column_gradient_caches[1].objective ===
        gaussian_shared_nuts_workspace.column_gradient_caches[2].objective
    @test gaussian_shared_nuts_workspace.column_gradient_caches[1].config ===
        gaussian_shared_nuts_workspace.column_gradient_caches[2].config
    @test parent(gaussian_shared_nuts_workspace.column_gradient_caches[1].buffer) === gaussian_shared_nuts_workspace.tree_next_gradient
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].left.position) === gaussian_nuts_workspace.tree_left_position
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].right.position) === gaussian_nuts_workspace.tree_right_position
    @test parent(gaussian_nuts_workspace.column_tree_workspaces[1].proposal.position) === gaussian_nuts_workspace.tree_proposal_position
    @test length(gaussian_nuts_workspace.column_continuation_states) == 3
    @test gaussian_nuts_workspace.column_continuation_states[1] isa UncertainTea.NUTSContinuationState
    @test length(gaussian_nuts_workspace.column_continuation_states[1].proposal.position) == 1
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].left.position) === gaussian_nuts_workspace.left_position
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].left.momentum) === gaussian_nuts_workspace.left_momentum
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].left.gradient) === gaussian_nuts_workspace.left_gradient
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].right.position) === gaussian_nuts_workspace.right_position
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].right.momentum) === gaussian_nuts_workspace.right_momentum
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].right.gradient) === gaussian_nuts_workspace.right_gradient
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].proposal.position) === gaussian_nuts_workspace.proposal_position
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].proposal.momentum) === gaussian_nuts_workspace.proposal_momentum
    @test parent(gaussian_nuts_workspace.column_continuation_states[1].proposal.gradient) === gaussian_nuts_workspace.proposal_gradient
    @test length(gaussian_nuts_workspace.tree_current_logjoint) == 3
    @test length(gaussian_nuts_workspace.tree_left_logjoint) == 3
    @test length(gaussian_nuts_workspace.tree_right_logjoint) == 3
    @test length(gaussian_nuts_workspace.tree_proposal_logjoint) == 3
    @test length(gaussian_nuts_workspace.left_logjoint) == 3
    @test length(gaussian_nuts_workspace.right_logjoint) == 3
    @test length(gaussian_nuts_workspace.continuation_proposal_logjoint) == 3
    @test length(gaussian_nuts_workspace.continuation_proposed_energy) == 3
    @test length(gaussian_nuts_workspace.continuation_delta_energy) == 3
    @test length(gaussian_nuts_workspace.continuation_accept_prob) == 3
    @test length(gaussian_nuts_workspace.continuation_candidate_log_weight) == 3
    @test length(gaussian_nuts_workspace.continuation_combined_log_weight) == 3
    @test length(gaussian_nuts_workspace.continuation_select_proposal) == 3
    @test length(gaussian_nuts_workspace.subtree_proposed_energy) == 3
    @test length(gaussian_nuts_workspace.subtree_delta_energy) == 3
    @test length(gaussian_nuts_workspace.subtree_proposal_energy) == 3
    @test length(gaussian_nuts_workspace.subtree_proposal_energy_error) == 3
    @test length(gaussian_nuts_workspace.subtree_accept_prob) == 3
    @test length(gaussian_nuts_workspace.subtree_candidate_log_weight) == 3
    @test length(gaussian_nuts_workspace.subtree_combined_log_weight) == 3
    @test length(gaussian_nuts_workspace.subtree_merged_turning) == 3
    @test length(gaussian_nuts_workspace.subtree_copy_left) == 3
    @test length(gaussian_nuts_workspace.subtree_copy_right) == 3
    @test length(gaussian_nuts_workspace.subtree_select_proposal) == 3
    masked_destination = reshape(collect(1.0:6.0), 2, 3)
    masked_source = reshape(collect(7.0:12.0), 2, 3)
    UncertainTea._copy_masked_columns!(masked_destination, masked_source, BitVector([true, false, true]))
    @test masked_destination[:, 1] == masked_source[:, 1]
    @test masked_destination[:, 2] == [3.0, 4.0]
    @test masked_destination[:, 3] == masked_source[:, 3]
    single_chain_mask = UncertainTea._single_chain_mask!(falses(3), 2)
    @test single_chain_mask == BitVector([false, true, false])
    sampled_directions = zeros(Int, 3)
    UncertainTea._sample_batched_nuts_directions!(
        sampled_directions,
        MersenneTwister(105),
        BitVector([true, false, true]),
    )
    @test sampled_directions[1] in (-1, 1)
    @test sampled_directions[2] == 0
    @test sampled_directions[3] in (-1, 1)
    @test UncertainTea._nuts_continuation_active(1, 3, false, false)
    @test !UncertainTea._nuts_continuation_active(3, 3, false, false)
    @test !UncertainTea._nuts_continuation_active(1, 3, true, false)
    @test !UncertainTea._nuts_continuation_active(1, 3, false, true)
    @test UncertainTea._mean_acceptance_stat(3.0, 2) == 1.5
    @test UncertainTea._mean_acceptance_stat(0.0, 0) == 0.0
    moved_destination = falses(3)
    UncertainTea._batched_positions_moved!(
        moved_destination,
        reshape([1.0, 2.0, 3.0], 1, 3),
        reshape([1.0, 4.0, 3.0], 1, 3),
    )
    @test moved_destination == BitVector([false, true, false])
    @test UncertainTea._position_moved([1.0, 2.0], [1.0, 3.0])
    @test !UncertainTea._position_moved([1.0, 2.0], [1.0, 2.0])
    gaussian_copy_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        gaussian_batch_constraints,
    )
    gaussian_copy_nuts_workspace.tree_left_position[:, 1] .= 11.0
    gaussian_copy_nuts_workspace.tree_left_momentum[:, 1] .= 12.0
    gaussian_copy_nuts_workspace.tree_left_gradient[:, 1] .= 13.0
    gaussian_copy_nuts_workspace.tree_left_logjoint[1] = 1.5
    UncertainTea._copy_single_batched_continuation_frontier_from_tree!(
        gaussian_copy_nuts_workspace,
        1,
        -1,
    )
    @test view(gaussian_copy_nuts_workspace.left_position, :, 1) == [11.0]
    @test view(gaussian_copy_nuts_workspace.left_momentum, :, 1) == [12.0]
    @test view(gaussian_copy_nuts_workspace.left_gradient, :, 1) == [13.0]
    @test gaussian_copy_nuts_workspace.left_logjoint[1] == 1.5
    @test gaussian_copy_nuts_workspace.column_continuation_states[1].left.logjoint == 1.5
    gaussian_copy_nuts_workspace.tree_proposal_position[:, 1] .= 21.0
    gaussian_copy_nuts_workspace.tree_proposal_momentum[:, 1] .= 22.0
    gaussian_copy_nuts_workspace.tree_proposal_gradient[:, 1] .= 23.0
    gaussian_copy_nuts_workspace.tree_proposal_logjoint[1] = 2.5
    UncertainTea._copy_single_batched_continuation_proposal_from_tree!(
        gaussian_copy_nuts_workspace,
        1,
    )
    @test view(gaussian_copy_nuts_workspace.proposal_position, :, 1) == [21.0]
    @test view(gaussian_copy_nuts_workspace.proposal_momentum, :, 1) == [22.0]
    @test view(gaussian_copy_nuts_workspace.proposal_gradient, :, 1) == [23.0]
    @test gaussian_copy_nuts_workspace.continuation_proposal_logjoint[1] == 2.5
    @test gaussian_copy_nuts_workspace.column_continuation_states[1].proposal.logjoint == 2.5
    gaussian_copy_nuts_workspace.left_position[:, 1] .= 0.0
    gaussian_copy_nuts_workspace.right_position[:, 1] .= 1.0
    gaussian_copy_nuts_workspace.left_momentum[:, 1] .= 1.0
    gaussian_copy_nuts_workspace.right_momentum[:, 1] .= 1.0
    UncertainTea._update_single_batched_continuation_turning!(
        gaussian_copy_nuts_workspace,
        1,
    )
    @test !gaussian_copy_nuts_workspace.subtree_merged_turning[1]
    @test !any(gaussian_copy_nuts_workspace.subtree_active)
    scalar_tree_workspace = UncertainTea.NUTSSubtreeWorkspace(1)
    scalar_tree_workspace.summary.log_weight = 1.0
    scalar_tree_workspace.summary.accept_stat_sum = 2.0
    scalar_tree_workspace.summary.accept_stat_count = 3
    scalar_tree_workspace.summary.integration_steps = 4
    scalar_tree_workspace.summary.proposal_energy = 5.0
    scalar_tree_workspace.summary.proposal_energy_error = 6.0
    scalar_tree_workspace.summary.turning = true
    scalar_tree_workspace.summary.divergent = true
    UncertainTea._reset_nuts_subtree_summary!(scalar_tree_workspace.summary)
    scalar_tree_summary = UncertainTea._nuts_subtree_summary(scalar_tree_workspace.summary)
    @test scalar_tree_summary.log_weight == -Inf
    @test scalar_tree_summary.accept_stat_sum == 0.0
    @test scalar_tree_summary.accept_stat_count == 0
    @test scalar_tree_summary.integration_steps == 0
    @test !scalar_tree_summary.turning
    @test !scalar_tree_summary.divergent
    @test !isfinite(scalar_tree_workspace.summary.proposal_energy)
    @test !isfinite(scalar_tree_workspace.summary.proposal_energy_error)
    scalar_continuation = UncertainTea.NUTSContinuationState(1)
    UncertainTea._initialize_nuts_continuation!(
        scalar_continuation,
        scalar_tree_workspace.current,
        scalar_tree_workspace.current,
        scalar_tree_workspace.current,
        1.0,
        0.0,
        -Inf,
        0.0,
        0,
        0,
        1,
        false,
        false,
    )
    scalar_tree_workspace.proposal.position .= 7.0
    scalar_tree_workspace.proposal.momentum .= 8.0
    scalar_tree_workspace.proposal.gradient .= 9.0
    scalar_tree_workspace.proposal.logjoint = 0.25
    scalar_tree_workspace.left.position .= 3.0
    scalar_tree_workspace.left.momentum .= 4.0
    scalar_tree_workspace.left.gradient .= 5.0
    scalar_tree_workspace.left.logjoint = -0.5
    scalar_tree_workspace.summary.log_weight = -0.5
    scalar_tree_workspace.summary.accept_stat_sum = 0.75
    scalar_tree_workspace.summary.accept_stat_count = 2
    scalar_tree_workspace.summary.integration_steps = 3
    scalar_tree_workspace.summary.proposal_energy = 1.25
    scalar_tree_workspace.summary.proposal_energy_error = 0.25
    scalar_tree_workspace.summary.turning = false
    scalar_tree_workspace.summary.divergent = false
    UncertainTea._copy_nuts_continuation_frontier_from_tree!(
        scalar_continuation,
        scalar_tree_workspace,
        -1,
    )
    @test UncertainTea._nuts_subtree_start_state(scalar_continuation, -1) === scalar_continuation.left
    @test UncertainTea._nuts_subtree_start_state(scalar_continuation, 1) === scalar_continuation.right
    @test scalar_continuation.left.logjoint == -0.5
    @test scalar_continuation.left.position == [3.0]
    UncertainTea._copy_nuts_continuation_proposal_from_tree!(
        scalar_continuation,
        scalar_tree_workspace,
    )
    @test scalar_continuation.proposal.logjoint == 0.25
    @test scalar_continuation.proposal_energy == 1.25
    @test scalar_continuation.proposal_energy_error == 0.25
    UncertainTea._merge_nuts_subtree_summary!(
        scalar_continuation,
        scalar_tree_workspace,
        -0.5,
    )
    UncertainTea._merge_nuts_continuation_turning!(scalar_continuation, true)
    @test scalar_continuation.integration_steps == 3
    @test scalar_continuation.accept_stat_sum == 0.75
    @test scalar_continuation.accept_stat_count == 2
    @test scalar_continuation.proposal_energy == 1.25
    @test scalar_continuation.proposal_energy_error == 0.25
    @test scalar_continuation.proposal.logjoint == 0.25
    @test scalar_continuation.turning
    turning_destination = falses(3)
    @test UncertainTea._batched_is_turning!(
        turning_destination,
        reshape([0.0, 0.0, 0.0], 1, 3),
        reshape([1.0, -1.0, 2.0], 1, 3),
        reshape([1.0, 1.0, 1.0], 1, 3),
        reshape([1.0, -1.0, -1.0], 1, 3),
        BitVector([true, true, false]),
    ) == BitVector([false, true, false])
    gaussian_nuts_tree_current = gaussian_nuts_workspace.column_tree_workspaces[1].current
    gaussian_nuts_tree_next = gaussian_nuts_workspace.column_tree_workspaces[1].next
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_batch_logjoint,
        gaussian_batch_gradient,
        [1.0],
        (),
        gaussian_batch_constraints,
        0.15,
        1000.0,
        MersenneTwister(93),
    )
    @test gaussian_nuts_workspace.column_tree_workspaces[1].current === gaussian_nuts_tree_current
    @test gaussian_nuts_workspace.column_tree_workspaces[1].next === gaussian_nuts_tree_next
    @test gaussian_nuts_tree_current.position ≈ gaussian_batch_params[:, 1] atol=1e-8
    @test gaussian_nuts_tree_current.logjoint ≈ gaussian_batch_logjoint[1] atol=1e-8
    @test gaussian_nuts_tree_current.momentum ≈ view(gaussian_nuts_workspace.current_momentum, :, 1) atol=1e-8
    @test gaussian_nuts_tree_next.logjoint ≈
        logjoint_unconstrained(gaussian_mean, gaussian_nuts_tree_next.position, (), gaussian_batch_constraints[1]) atol=1e-8
    @test gaussian_nuts_workspace.column_continuation_states[1].left.position ≈ view(gaussian_nuts_workspace.left_position, :, 1) atol=1e-8
    @test gaussian_nuts_workspace.column_continuation_states[1].right.position ≈ view(gaussian_nuts_workspace.right_position, :, 1) atol=1e-8
    @test gaussian_nuts_workspace.column_continuation_states[1].proposal.position ≈ view(gaussian_nuts_workspace.proposal_position, :, 1) atol=1e-8
    @test gaussian_nuts_workspace.left_logjoint[1] ≈ gaussian_nuts_workspace.column_continuation_states[1].left.logjoint atol=1e-8
    @test gaussian_nuts_workspace.right_logjoint[1] ≈ gaussian_nuts_workspace.column_continuation_states[1].right.logjoint atol=1e-8
    @test gaussian_nuts_workspace.continuation_proposal_logjoint[1] ≈ gaussian_nuts_workspace.column_continuation_states[1].proposal.logjoint atol=1e-8
    @test gaussian_nuts_workspace.continuation_proposed_energy[1] ≈
        gaussian_nuts_workspace.column_continuation_states[1].proposal_energy atol=1e-8
    @test gaussian_nuts_workspace.continuation_delta_energy[1] ≈
        gaussian_nuts_workspace.column_continuation_states[1].proposal_energy_error atol=1e-8
    gaussian_nuts_summary = UncertainTea._nuts_proposal_summary(
        gaussian_nuts_workspace.column_continuation_states[1],
        gaussian_batch_params[:, 1],
    )
    @test gaussian_nuts_summary[2] ≈
        gaussian_nuts_workspace.column_continuation_states[1].proposal_energy atol=1e-8
    @test gaussian_nuts_summary[3] ≈
        gaussian_nuts_workspace.column_continuation_states[1].proposal_energy_error atol=1e-8
    gaussian_nuts_workspace.step_direction .= [1, -1, 1]
    subtree_active = BitVector([true, true, false])
    UncertainTea._initialize_batched_nuts_subtree_states!(gaussian_nuts_workspace, subtree_active)
    @test gaussian_nuts_workspace.subtree_copy_left == BitVector([false, true, false])
    @test gaussian_nuts_workspace.subtree_copy_right == BitVector([true, false, false])
    @test view(gaussian_nuts_workspace.tree_current_position, :, 1) ≈ view(gaussian_nuts_workspace.right_position, :, 1)
    @test view(gaussian_nuts_workspace.tree_current_position, :, 2) ≈ view(gaussian_nuts_workspace.left_position, :, 2)
    @test view(gaussian_nuts_workspace.tree_proposal_position, :, 1) ≈ view(gaussian_nuts_workspace.right_position, :, 1)
    @test view(gaussian_nuts_workspace.tree_proposal_position, :, 2) ≈ view(gaussian_nuts_workspace.left_position, :, 2)
    @test gaussian_nuts_workspace.tree_current_logjoint[1] ≈ gaussian_nuts_workspace.right_logjoint[1] atol=1e-8
    @test gaussian_nuts_workspace.tree_current_logjoint[2] ≈ gaussian_nuts_workspace.left_logjoint[2] atol=1e-8
    @test gaussian_nuts_workspace.tree_proposal_logjoint[1] ≈ gaussian_nuts_workspace.right_logjoint[1] atol=1e-8
    @test gaussian_nuts_workspace.tree_proposal_logjoint[2] ≈ gaussian_nuts_workspace.left_logjoint[2] atol=1e-8
    @test gaussian_nuts_workspace.tree_depths[1] == 1
    @test gaussian_nuts_workspace.integration_steps[1] in 0:1
    @test isfinite(gaussian_nuts_workspace.continuation_log_weight[1])
    @test isfinite(gaussian_nuts_workspace.continuation_proposed_energy[1])
    @test isfinite(gaussian_nuts_workspace.continuation_delta_energy[1])
    @test 0.0 <= gaussian_nuts_workspace.continuation_accept_prob[1] <= 1.0
    @test isfinite(gaussian_nuts_workspace.continuation_candidate_log_weight[1])
    @test isfinite(gaussian_nuts_workspace.continuation_combined_log_weight[1])
    @test gaussian_nuts_workspace.continuation_accept_stat_count[1] in 0:1
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_shared_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.05,
        1000.0,
        MersenneTwister(94),
    )
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_shared_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.05,
        2,
        1000.0,
        MersenneTwister(95),
    )
    @test all(depth == 2 for depth in gaussian_shared_nuts_workspace.tree_depths)
    @test all(steps >= 2 for steps in gaussian_shared_nuts_workspace.integration_steps)
    @test all(isfinite, gaussian_shared_nuts_workspace.continuation_log_weight)
    @test all(isfinite, gaussian_shared_nuts_workspace.continuation_candidate_log_weight)
    @test all(isfinite, gaussian_shared_nuts_workspace.continuation_combined_log_weight)
    @test all(0.0 .<= gaussian_shared_nuts_workspace.continuation_accept_prob .<= 1.0)
    @test all(count >= 1 for count in gaussian_shared_nuts_workspace.continuation_accept_stat_count)
    @test all(isfinite, gaussian_shared_nuts_workspace.subtree_proposed_energy)
    @test all(isfinite, gaussian_shared_nuts_workspace.subtree_delta_energy)
    @test all(0.0 .<= gaussian_shared_nuts_workspace.subtree_accept_prob .<= 1.0)
    @test all(isfinite, gaussian_shared_nuts_workspace.subtree_candidate_log_weight)
    @test all(isfinite, gaussian_shared_nuts_workspace.subtree_combined_log_weight)
    @test gaussian_shared_nuts_workspace.continuation_proposed_energy ≈
        [state.proposal_energy for state in gaussian_shared_nuts_workspace.column_continuation_states] atol=1e-8
    @test gaussian_shared_nuts_workspace.continuation_delta_energy ≈
        [state.proposal_energy_error for state in gaussian_shared_nuts_workspace.column_continuation_states] atol=1e-8
    @test any(gaussian_shared_nuts_workspace.subtree_copy_left .| gaussian_shared_nuts_workspace.subtree_copy_right)
    @test gaussian_shared_nuts_workspace.subtree_merged_turning ==
        UncertainTea._batched_is_turning!(
            falses(length(gaussian_shared_nuts_workspace.subtree_merged_turning)),
            gaussian_shared_nuts_workspace.left_position,
            gaussian_shared_nuts_workspace.right_position,
            gaussian_shared_nuts_workspace.left_momentum,
            gaussian_shared_nuts_workspace.right_momentum,
            trues(length(gaussian_shared_nuts_workspace.subtree_merged_turning)),
        )
    gaussian_single_shared_params = gaussian_batch_params[:, 1:1]
    gaussian_single_shared_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_single_shared_params,
        (),
        choicemap((:y, 0.4)),
    )
    gaussian_single_shared_logjoint = batched_logjoint_unconstrained(
        gaussian_mean,
        gaussian_single_shared_params,
        (),
        choicemap((:y, 0.4)),
    )
    gaussian_single_shared_gradient = batched_logjoint_gradient_unconstrained(
        gaussian_mean,
        gaussian_single_shared_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_single_shared_nuts_workspace,
        gaussian_mean,
        gaussian_single_shared_params,
        gaussian_single_shared_logjoint,
        gaussian_single_shared_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(96),
    )
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_single_shared_nuts_workspace,
        gaussian_mean,
        gaussian_single_shared_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        3,
        1000.0,
        MersenneTwister(97),
    )
    @test gaussian_single_shared_nuts_workspace.tree_depths == [2]
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_single_shared_nuts_workspace,
        gaussian_mean,
        gaussian_single_shared_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        3,
        1000.0,
        MersenneTwister(98),
    )
    @test gaussian_single_shared_nuts_workspace.tree_depths == [3]
    @test gaussian_single_shared_nuts_workspace.integration_steps[1] >= 6
    gaussian_mixed_depth_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_mixed_depth_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(99),
    )
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_mixed_depth_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        4,
        1000.0,
        MersenneTwister(100),
    )
    @test all(depth == 2 for depth in gaussian_mixed_depth_nuts_workspace.tree_depths)
    gaussian_mixed_depth_nuts_workspace.continuation_turning[1] = true
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_mixed_depth_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        4,
        1000.0,
        MersenneTwister(101),
    )
    @test gaussian_mixed_depth_nuts_workspace.tree_depths[1] == 2
    @test gaussian_mixed_depth_nuts_workspace.tree_depths[2:3] == [3, 3]
    @test gaussian_mixed_depth_nuts_workspace.subtree_accept_stat_count[1] == 0
    @test gaussian_mixed_depth_nuts_workspace.subtree_candidate_log_weight[1] == -Inf
    @test !gaussian_mixed_depth_nuts_workspace.subtree_copy_left[1]
    @test !gaussian_mixed_depth_nuts_workspace.subtree_copy_right[1]
    @test !gaussian_mixed_depth_nuts_workspace.subtree_select_proposal[1]
    gaussian_mixed_depth_nuts_workspace.continuation_turning[1] = false
    @test UncertainTea._continue_batched_nuts_batched_subtree!(
        gaussian_mixed_depth_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        4,
        1000.0,
        MersenneTwister(102),
    )
    @test gaussian_mixed_depth_nuts_workspace.tree_depths[1] == 2
    @test gaussian_mixed_depth_nuts_workspace.tree_depths[2:3] == [4, 4]
    gaussian_finalized_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._batched_nuts_proposals!(
        gaussian_finalized_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.05,
        3,
        1000.0,
        MersenneTwister(103),
    )
    @test gaussian_finalized_nuts_workspace.proposed_logjoint ≈
        gaussian_finalized_nuts_workspace.continuation_proposal_logjoint atol=1e-8
    @test gaussian_finalized_nuts_workspace.proposed_energy ≈
        gaussian_finalized_nuts_workspace.continuation_proposed_energy atol=1e-8
    @test gaussian_finalized_nuts_workspace.accept_prob ≈
        UncertainTea._mean_acceptance_stats!(
            similar(gaussian_finalized_nuts_workspace.accept_prob),
            gaussian_finalized_nuts_workspace.continuation_accept_stat_sum,
            gaussian_finalized_nuts_workspace.continuation_accept_stat_count,
        ) atol=1e-8
    @test gaussian_finalized_nuts_workspace.energy_error ≈
        gaussian_finalized_nuts_workspace.continuation_delta_energy atol=1e-8
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
    ] atol=2e-8
    coin_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        coin_workspace,
        zeros(0, 3),
        (),
        [
            choicemap((:y, true)),
            choicemap((:y, false)),
            choicemap((:y, true)),
        ],
    )
    coin_workspace_env = coin_workspace.batched_environment[]
    coin_workspace_observed = coin_workspace_env.observed_values
    @test coin_workspace_values ≈ coin_batch_logjoint atol=2e-8
    @test UncertainTea._logjoint_with_batched_backend!(
        coin_workspace,
        zeros(0, 3),
        (),
        choicemap((:y, true)),
    ) ≈ fill(logjoint(observed_coin, Float64[], (), choicemap((:y, true))), 3) atol=2e-8
    @test coin_workspace.batched_environment[] === coin_workspace_env
    @test coin_workspace_env.observed_values === coin_workspace_observed
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
    @test UncertainTea._warmup_window_length(short_warmup_schedule, 1) == 7
    @test UncertainTea._warmup_window_length(long_warmup_schedule, 1) == 25
    @test UncertainTea._warmup_window_length(long_warmup_schedule, 2) == 50
    @test UncertainTea._warmup_window_length(long_warmup_schedule, 3) == 45
    @test UncertainTea._mean_batched_adaptation_probability([0.8, 0.6, 0.4], falses(3)) ≈ 0.6 atol=1e-8
    @test UncertainTea._mean_batched_adaptation_probability([0.8, 0.6, 0.4], BitVector([false, true, false])) ≈
        0.4 atol=1e-8
    early_window_state = UncertainTea._running_variance_state(1, 24)
    late_window_state = UncertainTea._running_variance_state(1, 24)
    late_window_state.count = 24
    @test UncertainTea._mass_adaptation_weight(early_window_state, true, 0.1, false) == 1.0
    @test UncertainTea._mass_adaptation_weight(early_window_state, false, 0.25, false) == 1.0
    @test UncertainTea._mass_adaptation_weight(late_window_state, false, 0.25, false) == 0.25
    @test UncertainTea._mass_adaptation_weight(early_window_state, false, 0.25, true) == 0.0
    mass_adaptation_weights = zeros(3)
    UncertainTea._mass_adaptation_weights!(
        late_window_state,
        mass_adaptation_weights,
        BitVector([true, false, false]),
        [0.2, 0.3, 0.4],
        BitVector([false, false, true]),
    )
    @test mass_adaptation_weights ≈ [1.0, 0.3, 0.0] atol=1e-8
    @test UncertainTea._running_variance_clip_scale(early_window_state) == 8.0
    early_window_state.count = 4
    @test UncertainTea._running_variance_clip_scale(early_window_state) == 8.0
    early_window_state.count = 14
    @test UncertainTea._running_variance_clip_scale(early_window_state) ≈ 6.5 atol=1e-8
    @test UncertainTea._running_variance_window_progress(early_window_state) ≈ 0.5 atol=1e-8
    @test UncertainTea._running_variance_clip_scale(late_window_state) == 5.0
    @test UncertainTea._running_variance_window_progress(late_window_state) == 1.0
    masked_variance_state = UncertainTea._running_variance_state(2)
    UncertainTea._update_running_variance!(
        masked_variance_state,
        [1.0 2.0 5.0; 10.0 20.0 50.0],
        BitVector([true, false, true]),
    )
    @test masked_variance_state.count == 2
    @test masked_variance_state.mean ≈ [3.0, 30.0] atol=1e-8
    @test masked_variance_state.m2 ≈ [8.0, 800.0] atol=1e-8
    @test masked_variance_state.weight_sum == 2.0
    @test masked_variance_state.weight_square_sum == 2.0
    @test UncertainTea._running_variance_effective_count(masked_variance_state) == 2.0
    weighted_variance_state = UncertainTea._running_variance_state(1)
    UncertainTea._update_running_variance!(weighted_variance_state, [0.0], 1.0)
    UncertainTea._update_running_variance!(weighted_variance_state, [10.0], 0.25)
    @test weighted_variance_state.count == 2
    @test weighted_variance_state.weight_sum ≈ 1.25 atol=1e-8
    @test weighted_variance_state.weight_square_sum ≈ 1.0625 atol=1e-8
    @test UncertainTea._running_variance_effective_count(weighted_variance_state) ≈
        (1.25^2 / 1.0625) atol=1e-8
    robust_variance_state = UncertainTea._running_variance_state(1)
    for value in (0.0, 0.05, -0.05, 0.1, 100.0)
        UncertainTea._update_running_variance!(robust_variance_state, [value])
    end
    @test robust_variance_state.count == 5
    @test robust_variance_state.mean[1] < 1.0
    @test robust_variance_state.m2[1] < 1.0
    @test UncertainTea._inverse_mass_matrix(robust_variance_state, 1e-3)[1] > 1.0

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
    gaussian_nuts_chain = nuts(
        gaussian_mean,
        (),
        constraints;
        num_samples=160,
        num_warmup=100,
        step_size=0.2,
        max_tree_depth=6,
        rng=MersenneTwister(60),
    )
    gaussian_nuts_baseline_chain = nuts(
        gaussian_mean,
        (),
        constraints;
        num_samples=20,
        num_warmup=0,
        step_size=0.2,
        max_tree_depth=5,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        rng=MersenneTwister(61),
    )
    gaussian_nuts_one_step_chain = nuts(
        gaussian_mean,
        (),
        constraints;
        num_samples=16,
        num_warmup=0,
        step_size=0.2,
        max_tree_depth=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        rng=MersenneTwister(64),
    )
    gaussian_nuts_one_step_chain_replay = nuts(
        gaussian_mean,
        (),
        constraints;
        num_samples=16,
        num_warmup=0,
        step_size=0.2,
        max_tree_depth=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        rng=MersenneTwister(64),
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
    gaussian_nuts_multichain = nuts_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=50,
        num_warmup=30,
        step_size=0.18,
        max_tree_depth=6,
        rng=MersenneTwister(62),
    )
    gaussian_nuts_multichain_replay = nuts_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=50,
        num_warmup=30,
        step_size=0.18,
        max_tree_depth=6,
        rng=MersenneTwister(62),
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
    gaussian_batched_baseline_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=40,
        num_warmup=20,
        step_size=0.18,
        num_leapfrog_steps=6,
        initial_params=gaussian_batch_params,
        adapt_step_size=false,
        adapt_mass_matrix=false,
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
    gaussian_batched_nuts_chain = batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=24,
        num_warmup=18,
        step_size=0.15,
        max_tree_depth=6,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(63),
    )
    gaussian_batched_nuts_chain_replay = batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=24,
        num_warmup=18,
        step_size=0.15,
        max_tree_depth=6,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(63),
    )
    gaussian_batched_nuts_one_step_chain = batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=16,
        num_warmup=0,
        step_size=0.15,
        max_tree_depth=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(64),
    )
    gaussian_batched_nuts_one_step_chain_replay = batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=16,
        num_warmup=0,
        step_size=0.15,
        max_tree_depth=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(64),
    )
    gaussian_batched_mass_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=20,
        num_warmup=20,
        step_size=0.18,
        num_leapfrog_steps=6,
        initial_params=gaussian_batch_params,
        adapt_step_size=false,
        adapt_mass_matrix=true,
        mass_matrix_min_samples=3,
        rng=MersenneTwister(55),
    )
    gaussian_batched_large_step_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=10,
        num_warmup=0,
        step_size=16.0,
        num_leapfrog_steps=4,
        initial_params=gaussian_batch_params,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        find_reasonable_step_size=true,
        rng=MersenneTwister(56),
    )
    gaussian_batched_divergence_adapt_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=10,
        num_warmup=15,
        step_size=2.0,
        num_leapfrog_steps=8,
        initial_params=gaussian_batch_params,
        adapt_step_size=true,
        adapt_mass_matrix=false,
        find_reasonable_step_size=false,
        divergence_threshold=1.0,
        rng=MersenneTwister(57),
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
    gaussian_nuts_rhat = rhat(gaussian_nuts_multichain)
    gaussian_nuts_ess = ess(gaussian_nuts_multichain)
    gaussian_nuts_summary = summarize(gaussian_nuts_multichain)
    gaussian_batched_nuts_summary = summarize(gaussian_batched_nuts_chain)
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
    @test length(massadaptationwindows(gaussian_chain)) == 3
    @test massadaptationwindows(gaussian_chain)[1] isa HMCMassAdaptationWindowSummary
    @test massadaptationwindows(gaussian_chain)[1].iteration_start == 16
    @test massadaptationwindows(gaussian_chain)[1].iteration_end == 40
    @test massadaptationwindows(gaussian_chain)[1].window_length == 25
    @test massadaptationwindows(gaussian_chain)[1].clip_scale_start == 8.0
    @test 5.0 <= massadaptationwindows(gaussian_chain)[1].clip_scale_end <= 8.0
    @test massadaptationwindows(gaussian_chain)[1].effective_count >= 3
    @test massadaptationwindows(gaussian_chain)[1].mass_min > 0
    @test massadaptationwindows(gaussian_chain)[1].mass_max >= massadaptationwindows(gaussian_chain)[1].mass_min
    @test any(gaussian_chain.accepted)
    @test gaussian_chain.step_size > 0
    @test gaussian_chain.mass_matrix[1] > 0
    @test gaussian_chain.target_accept == 0.8
    @test gaussian_baseline_chain.step_size == 0.25
    @test gaussian_baseline_chain.mass_matrix == [1.0]
    @test isempty(massadaptationwindows(gaussian_baseline_chain))
    @test 0 < gaussian_large_step_chain.step_size < 16.0
    @test gaussian_small_step_chain.step_size > 1e-6
    @test all(isfinite, gaussian_large_step_chain.logjoint_values)
    @test all(isfinite, gaussian_small_step_chain.logjoint_values)
    @test gaussian_windowed_mass_chain.step_size == 0.2
    @test gaussian_windowed_mass_chain.mass_matrix[1] != 1.0
    @test gaussian_nuts_chain.sampler == :nuts
    @test length(gaussian_nuts_chain) == 160
    @test size(gaussian_nuts_chain.unconstrained_samples) == (1, 160)
    @test size(gaussian_nuts_chain.constrained_samples) == (1, 160)
    @test all(isfinite, gaussian_nuts_chain.logjoint_values)
    @test all(isfinite, gaussian_nuts_chain.acceptance_stats)
    @test all(0.0 <= stat <= 1.0 for stat in gaussian_nuts_chain.acceptance_stats)
    @test all(1 <= depth <= gaussian_nuts_chain.max_tree_depth for depth in treedepths(gaussian_nuts_chain))
    @test all(1 <= steps <= (2 ^ gaussian_nuts_chain.max_tree_depth - 1) for steps in integrationsteps(gaussian_nuts_chain))
    @test gaussian_nuts_chain.num_leapfrog_steps == 0
    @test gaussian_nuts_chain.max_tree_depth == 6
    @test 0.0 <= acceptancerate(gaussian_nuts_chain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_nuts_chain) <= 1.0
    @test gaussian_nuts_chain.step_size > 0
    @test gaussian_nuts_chain.mass_matrix[1] > 0
    @test any(gaussian_nuts_chain.accepted)
    @test abs(sum(gaussian_nuts_chain.constrained_samples[1, :]) / size(gaussian_nuts_chain.constrained_samples, 2) - 0.15) < 0.25
    @test gaussian_nuts_baseline_chain.step_size == 0.2
    @test gaussian_nuts_baseline_chain.mass_matrix == [1.0]
    @test isempty(massadaptationwindows(gaussian_nuts_baseline_chain))
    @test gaussian_nuts_one_step_chain.max_tree_depth == 1
    @test gaussian_nuts_one_step_chain.step_size == 0.2
    @test gaussian_nuts_one_step_chain.mass_matrix == [1.0]
    @test isempty(massadaptationwindows(gaussian_nuts_one_step_chain))
    @test all(depth == 1 for depth in treedepths(gaussian_nuts_one_step_chain))
    @test all(0 <= steps <= 1 for steps in integrationsteps(gaussian_nuts_one_step_chain))
    @test gaussian_nuts_one_step_chain.unconstrained_samples == gaussian_nuts_one_step_chain_replay.unconstrained_samples
    @test gaussian_nuts_one_step_chain.accepted == gaussian_nuts_one_step_chain_replay.accepted
    @test nchains(gaussian_multichain) == 3
    @test numsamples(gaussian_multichain) == 60
    @test gaussian_multichain[1] isa HMCChain
    @test length(massadaptationwindows(gaussian_multichain)) == 3
    @test all(length(windows) == 1 for windows in massadaptationwindows(gaussian_multichain))
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
    @test gaussian_summary.diagnostics isa HMCDiagnosticsSummary
    @test acceptancerate(gaussian_summary) == acceptancerate(gaussian_multichain)
    @test divergencerate(gaussian_summary) == divergencerate(gaussian_multichain)
    @test gaussian_summary.diagnostics.mean_step_size ≈
        sum(chain.step_size for chain in gaussian_multichain) / nchains(gaussian_multichain) atol=1e-8
    @test length(gaussian_summary.diagnostics.step_sizes) == nchains(gaussian_multichain)
    @test length(massadaptationwindows(gaussian_summary)) == 1
    @test massadaptationwindows(gaussian_summary)[1] isa HMCMassAdaptationSummary
    @test massadaptationwindows(gaussian_summary)[1].chains == 3
    @test massadaptationwindows(gaussian_summary)[1].num_updated == 3
    @test massadaptationwindows(gaussian_summary)[1].iteration_start == 6
    @test massadaptationwindows(gaussian_summary)[1].iteration_end == 25
    @test massadaptationwindows(gaussian_summary)[1].mean_effective_count >= 3
    @test massadaptationwindows(gaussian_summary)[1].min_mass > 0
    gaussian_summary_text = repr(MIME"text/plain"(), gaussian_summary)
    @test occursin("HMCSummary(gaussian_mean)", gaussian_summary_text)
    @test occursin("diagnostics:", gaussian_summary_text)
    @test occursin("acceptance_rate:", gaussian_summary_text)
    @test occursin("mass_adaptation_windows:", gaussian_summary_text)
    @test occursin("mu @ mu:", gaussian_summary_text)
    diagnostics_text = repr(MIME"text/plain"(), gaussian_summary.diagnostics)
    @test occursin("HMCDiagnosticsSummary", diagnostics_text)
    @test occursin("step_size:", diagnostics_text)
    @test occursin("window 1 [6:25]", diagnostics_text)
    mass_window_text = repr(MIME"text/plain"(), massadaptationwindows(gaussian_summary)[1])
    @test occursin("HMCMassAdaptationSummary", mass_window_text)
    @test occursin("chains: 3 updated=3/3", mass_window_text)
    @test gaussian_summary[1].binding == :mu
    @test gaussian_summary[1].address == :mu
    @test gaussian_summary[1].mean ≈ gaussian_pooled_mean atol=1e-8
    @test gaussian_summary[1].sd > 0
    @test gaussian_summary[1].quantiles[1] <= gaussian_summary[1].quantiles[2] <= gaussian_summary[1].quantiles[3]
    @test gaussian_summary[1].rhat == gaussian_rhat[1]
    @test gaussian_summary[1].ess == gaussian_ess[1]
    @test gaussian_summary[1].mean ≈ gaussian_summary_unconstrained[1].mean atol=1e-8
    @test length(massadaptationwindows(gaussian_summary_unconstrained)) == 1
    @test gaussian_multichain[1].unconstrained_samples[:, 1] ==
        gaussian_multichain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_multichain[1].accepted == gaussian_multichain_replay[1].accepted
    @test gaussian_multichain[1].unconstrained_samples[:, 1] != gaussian_multichain[2].unconstrained_samples[:, 1]
    @test nchains(gaussian_nuts_multichain) == 3
    @test numsamples(gaussian_nuts_multichain) == 50
    @test all(chain.sampler == :nuts for chain in gaussian_nuts_multichain)
    @test all(length(treedepths(chain)) == 50 for chain in gaussian_nuts_multichain)
    @test all(all(1 <= depth <= chain.max_tree_depth for depth in treedepths(chain)) for chain in gaussian_nuts_multichain)
    @test all(all(1 <= steps <= (2 ^ chain.max_tree_depth - 1) for steps in integrationsteps(chain)) for chain in gaussian_nuts_multichain)
    @test length(gaussian_nuts_rhat) == 1
    @test length(gaussian_nuts_ess) == 1
    @test isfinite(gaussian_nuts_rhat[1])
    @test 1.0 <= gaussian_nuts_rhat[1] < 1.1
    @test 1.0 <= gaussian_nuts_ess[1] <= nchains(gaussian_nuts_multichain) * numsamples(gaussian_nuts_multichain)
    @test acceptancerate(gaussian_nuts_summary) == acceptancerate(gaussian_nuts_multichain)
    @test divergencerate(gaussian_nuts_summary) == divergencerate(gaussian_nuts_multichain)
    @test gaussian_nuts_multichain[1].unconstrained_samples[:, 1] ==
        gaussian_nuts_multichain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_nuts_multichain[1].tree_depths == gaussian_nuts_multichain_replay[1].tree_depths
    @test nchains(gaussian_batched_chain) == 3
    @test numsamples(gaussian_batched_chain) == 40
    @test gaussian_batched_chain.args == ()
    @test length(gaussian_batched_chain.constraints) == 3
    @test gaussian_batched_chain[1].constraints[:y] == gaussian_batch_constraints[1][:y]
    @test gaussian_batched_chain[2].constraints[:y] == gaussian_batch_constraints[2][:y]
    @test all(chain.step_size > 0 for chain in gaussian_batched_chain)
    @test all(chain.mass_matrix[1] > 0 for chain in gaussian_batched_chain)
    @test all(length(massadaptationwindows(chain)) == 1 for chain in gaussian_batched_chain)
    @test length(massadaptationwindows(gaussian_batched_chain)) == 3
    @test all(windows[1].window_length == 10 for windows in massadaptationwindows(gaussian_batched_chain))
    @test all(windows[1].iteration_start == 6 for windows in massadaptationwindows(gaussian_batched_chain))
    @test all(windows[1].iteration_end == 15 for windows in massadaptationwindows(gaussian_batched_chain))
    @test all(windows[1].updated for windows in massadaptationwindows(gaussian_batched_chain))
    @test all(chain.step_size == 0.18 for chain in gaussian_batched_baseline_chain)
    @test all(chain.mass_matrix == [1.0] for chain in gaussian_batched_baseline_chain)
    @test all(isempty(massadaptationwindows(chain)) for chain in gaussian_batched_baseline_chain)
    @test all(chain.step_size == 0.18 for chain in gaussian_batched_mass_chain)
    @test all(chain.mass_matrix[1] != 1.0 for chain in gaussian_batched_mass_chain)
    @test all(0 < chain.step_size < 16.0 for chain in gaussian_batched_large_step_chain)
    @test all(all(isfinite, chain.logjoint_values) for chain in gaussian_batched_large_step_chain)
    @test all(chain.step_size < 2.0 for chain in gaussian_batched_divergence_adapt_chain)
    gaussian_batched_summary = summarize(gaussian_batched_chain)
    @test acceptancerate(gaussian_batched_summary) == acceptancerate(gaussian_batched_chain)
    @test divergencerate(gaussian_batched_summary) == divergencerate(gaussian_batched_chain)
    @test length(massadaptationwindows(gaussian_batched_summary)) == 1
    @test massadaptationwindows(gaussian_batched_summary)[1].chains == 3
    @test massadaptationwindows(gaussian_batched_summary)[1].window_length == 10
    @test massadaptationwindows(gaussian_batched_summary)[1].num_updated == 3
    @test occursin("window 1 [6:15]", repr(MIME"text/plain"(), gaussian_batched_summary))
    @test !(isapprox(gaussian_batched_chain[1].step_size, gaussian_batched_baseline_chain[1].step_size; atol=1e-8) &&
        isapprox(gaussian_batched_chain[1].mass_matrix[1], gaussian_batched_baseline_chain[1].mass_matrix[1]; atol=1e-8))
    @test 0.0 <= acceptancerate(gaussian_batched_chain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_batched_chain) <= 1.0
    @test length(gaussian_batched_rhat) == 1
    @test isfinite(gaussian_batched_rhat[1])
    @test gaussian_batched_chain[1].unconstrained_samples[:, 1] ==
        gaussian_batched_chain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_batched_chain[1].accepted == gaussian_batched_chain_replay[1].accepted
    @test nchains(gaussian_batched_nuts_chain) == 3
    @test numsamples(gaussian_batched_nuts_chain) == 24
    @test gaussian_batched_nuts_chain.args == ()
    @test length(gaussian_batched_nuts_chain.constraints) == 3
    @test all(chain.sampler == :nuts for chain in gaussian_batched_nuts_chain)
    @test all(chain.step_size > 0 for chain in gaussian_batched_nuts_chain)
    @test all(chain.mass_matrix[1] > 0 for chain in gaussian_batched_nuts_chain)
    @test all(length(massadaptationwindows(chain)) == 1 for chain in gaussian_batched_nuts_chain)
    @test all(all(1 <= depth <= chain.max_tree_depth for depth in treedepths(chain)) for chain in gaussian_batched_nuts_chain)
    @test all(all(1 <= steps <= (2 ^ chain.max_tree_depth - 1) for steps in integrationsteps(chain)) for chain in gaussian_batched_nuts_chain)
    @test acceptancerate(gaussian_batched_nuts_summary) == acceptancerate(gaussian_batched_nuts_chain)
    @test divergencerate(gaussian_batched_nuts_summary) == divergencerate(gaussian_batched_nuts_chain)
    @test length(massadaptationwindows(gaussian_batched_nuts_summary)) == 1
    @test massadaptationwindows(gaussian_batched_nuts_summary)[1].chains == 3
    @test gaussian_batched_nuts_chain[1].unconstrained_samples[:, 1] ==
        gaussian_batched_nuts_chain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_batched_nuts_chain[1].tree_depths == gaussian_batched_nuts_chain_replay[1].tree_depths
    @test nchains(gaussian_batched_nuts_one_step_chain) == 3
    @test numsamples(gaussian_batched_nuts_one_step_chain) == 16
    @test all(chain.max_tree_depth == 1 for chain in gaussian_batched_nuts_one_step_chain)
    @test all(chain.step_size == 0.15 for chain in gaussian_batched_nuts_one_step_chain)
    @test all(chain.mass_matrix == [1.0] for chain in gaussian_batched_nuts_one_step_chain)
    @test all(isempty(massadaptationwindows(chain)) for chain in gaussian_batched_nuts_one_step_chain)
    @test all(all(depth == 1 for depth in treedepths(chain)) for chain in gaussian_batched_nuts_one_step_chain)
    @test all(all(0 <= steps <= 1 for steps in integrationsteps(chain)) for chain in gaussian_batched_nuts_one_step_chain)
    @test 0.0 <= acceptancerate(gaussian_batched_nuts_one_step_chain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_batched_nuts_one_step_chain) <= 1.0
    @test gaussian_batched_nuts_one_step_chain[1].unconstrained_samples ==
        gaussian_batched_nuts_one_step_chain_replay[1].unconstrained_samples
    @test gaussian_batched_nuts_one_step_chain[1].accepted ==
        gaussian_batched_nuts_one_step_chain_replay[1].accepted
    @test gaussian_batched_nuts_one_step_chain[1].tree_depths ==
        gaussian_batched_nuts_one_step_chain_replay[1].tree_depths
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
    @test all(chain.mass_matrix[1] > 0 for chain in positive_batched_chain)
    @test parameterchoicemap(observed_positive_step, positive_chain.constrained_samples[:, 1])[:state => :sigma] ==
        positive_chain.constrained_samples[1, 1]

    @tea static function observed_only()
        {:y} ~ bernoulli(0.5f0)
        return nothing
    end

    @test_throws DimensionMismatch parameterchoicemap(gaussian_mean, Float64[])
    @test_throws DimensionMismatch logjoint(iid_model, params2, (), repeated)
    @test_throws ArgumentError hmc(observed_only, (), choicemap((:y, true)); num_samples=10)
    @test_throws ArgumentError nuts(observed_only, (), choicemap((:y, true)); num_samples=10)
    @test_throws ArgumentError hmc(gaussian_mean, (), constraints; num_samples=10, divergence_threshold=0.0)
    @test_throws ArgumentError nuts(gaussian_mean, (), constraints; num_samples=10, max_tree_depth=0)
    @test_throws ArgumentError hmc_chains(gaussian_mean, (), constraints; num_chains=0, num_samples=10)
    @test_throws ArgumentError nuts_chains(gaussian_mean, (), constraints; num_chains=0, num_samples=10)
    @test_throws ArgumentError batched_hmc(gaussian_mean, (), gaussian_batch_constraints; num_chains=0, num_samples=10)
    @test_throws ArgumentError batched_nuts(gaussian_mean, (), gaussian_batch_constraints; num_chains=0, num_samples=10)
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
    @test_throws DimensionMismatch batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=10,
        initial_params=zeros(1, 2),
    )
end
