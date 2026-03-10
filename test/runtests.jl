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
    @test logjoint(gaussian_mean, params, (), constraints; rng=MersenneTwister(7)) ≈ expected_joint

    @tea static function iid_model(n)
        mu ~ normal(0.0f0, 1.0f0)
        for i in 1:n
            {:y => i} ~ normal(mu, 1.0f0)
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
    @test plan2.steps[1].parameter_slot == 1
    @test isnothing(plan2.steps[2].parameter_slot)
    @test length(plan2.steps[2].scopes) == 1

    params2 = [Float64(trace2[:mu])]
    expected_joint2 = UncertainTea.logpdf(normal(0.0f0, 1.0f0), trace2[:mu]) +
        sum(UncertainTea.logpdf(normal(trace2[:mu], 1.0f0), y) for y in ys)

    @test logjoint(iid_model, params2, (length(ys),), repeated; rng=MersenneTwister(8)) ≈ expected_joint2

    @tea static function step(prev)
        z ~ normal(prev, 1.0f0)
        return z
    end

    step_spec = modelspec(step)
    step_plan = executionplan(step)
    step_trace, _ = generate(step, (2.0f0,), choicemap(); rng=MersenneTwister(4))
    @test parametercount(step_spec.parameter_layout) == 1
    @test step_spec.parameter_layout.slots[1].binding == :z
    @test step_plan.steps[1].parameter_slot == 1
    @test logjoint(step, parameter_vector(step_trace), (2.0f0,), choicemap(); rng=MersenneTwister(9)) ≈
        UncertainTea.logpdf(normal(2.0f0, 1.0f0), step_trace[:z])

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
    @test spec3.choices[1].rhs.callee == :step
    @test spec3.choices[2].rhs isa GenerativeCallSpec
    @test !isrepeatedchoice(spec3.choices[1])
    @test isrepeatedchoice(spec3.choices[2])
    @test spec3.choices[2].scopes[1].iterator == :t
    @test spec3.choices[2].scopes[1].iterable == :(2:T)
    @test parametercount(spec3.parameter_layout) == 0
    @test isnothing(plan3.steps[1].parameter_slot)
    @test isnothing(plan3.steps[2].parameter_slot)
    @test length(plan3.steps[2].scopes) == 1
    @test parameter_vector(trace3) == Float64[]
    @test_throws ArgumentError logjoint(chain_model, Float64[], (3,), choicemap(); rng=MersenneTwister(10))

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
    @test length(plan4.steps[2].scopes) == 2

    @test_throws DimensionMismatch parameterchoicemap(gaussian_mean, Float64[])
end
