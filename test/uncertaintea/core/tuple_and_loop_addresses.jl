@testset "tuple_and_loop_addresses" begin
    @tea static function tuple_address_model(n)
        mu ~ normal(0.0f0, 1.0f0)
        for i = 1:n
            {(:y, i)} ~ normal(mu, 1.0f0)
        end
        return mu
    end

    tupleaddr_ys = Float32[0.2, -0.4, 0.7]
    tupleaddr_constraints = choicemap([((:y, i), tupleaddr_ys[i]) for i in eachindex(tupleaddr_ys)])
    tupleaddr_trace, tupleaddr_logw = generate(
        tuple_address_model,
        (3,),
        tupleaddr_constraints;
        rng=MersenneTwister(301),
    )
    tupleaddr_spec = modelspec(tuple_address_model)
    tupleaddr_params = parameter_vector(tupleaddr_trace)
    tupleaddr_expected =
        UncertainTea.logpdf(normal(0.0f0, 1.0f0), tupleaddr_trace[:mu]) +
        sum(UncertainTea.logpdf(normal(tupleaddr_trace[:mu], 1.0f0), y) for y in tupleaddr_ys)
    tupleaddr_full = choicemap(
        [(:mu, tupleaddr_trace[:mu]); [((:y, i), tupleaddr_ys[i]) for i in eachindex(tupleaddr_ys)]],
    )

    @test tupleaddr_trace[:y=>1] == tupleaddr_ys[1]
    @test tupleaddr_trace[(:y, 3)] == tupleaddr_ys[3]
    @test isaddresstemplate(tupleaddr_spec.choices[2].address)
    @test logjoint(tuple_address_model, tupleaddr_params, (3,), tupleaddr_constraints) ≈
          tupleaddr_expected atol=1e-6
    @test assess(tuple_address_model, (3,), tupleaddr_full) ≈ tupleaddr_expected atol=1e-6

    tupleaddr_batch_params = reshape(Float64[-0.3, 0.0, 0.5], 1, 3)
    tupleaddr_batch = batched_logjoint(
        tuple_address_model,
        tupleaddr_batch_params,
        (3,),
        tupleaddr_constraints,
    )
    @test tupleaddr_batch ≈ [
        logjoint(tuple_address_model, tupleaddr_batch_params[:, index], (3,), tupleaddr_constraints)
        for index = 1:3
    ] atol=1e-8

    @test_throws ArgumentError assess(tuple_address_model, (3,), tupleaddr_constraints)

    @tea static function loop_deterministic_model(n)
        mu ~ normal(0.0f0, 1.0f0)
        for i = 1:n
            scaled = mu * 2.0f0 + i
            {:y => i} ~ normal(scaled, 1.0f0)
        end
        return mu
    end

    loopdet_ys = Float32[1.4, 2.9, 3.6]
    loopdet_constraints = choicemap([(:y => i, loopdet_ys[i]) for i in eachindex(loopdet_ys)])
    loopdet_trace, _ = generate(
        loop_deterministic_model,
        (3,),
        loopdet_constraints;
        rng=MersenneTwister(302),
    )
    loopdet_plan = executionplan(loop_deterministic_model)
    loopdet_params = parameter_vector(loopdet_trace)
    loopdet_mu = loopdet_trace[:mu]
    loopdet_expected =
        UncertainTea.logpdf(normal(0.0f0, 1.0f0), loopdet_mu) +
        sum(UncertainTea.logpdf(normal(loopdet_mu * 2.0f0 + i, 1.0f0), loopdet_ys[i]) for i in eachindex(loopdet_ys))

    @test loopdet_plan.steps[2] isa LoopPlanStep
    @test length(loopdet_plan.steps[2].body) == 2
    @test loopdet_plan.steps[2].body[1] isa DeterministicPlanStep
    @test loopdet_plan.steps[2].body[1].binding == :scaled
    @test loopdet_plan.steps[2].body[2] isa ChoicePlanStep
    @test logjoint(loop_deterministic_model, loopdet_params, (3,), loopdet_constraints) ≈
          loopdet_expected atol=1e-6

    loopdet_batch_params = reshape(Float64[-0.2, 0.1, 0.4], 1, 3)
    loopdet_batch_gradient = batched_logjoint_gradient_unconstrained(
        loop_deterministic_model,
        loopdet_batch_params,
        (3,),
        loopdet_constraints,
    )
    @test loopdet_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                loop_deterministic_model,
                loopdet_batch_params[:, index],
                (3,),
                loopdet_constraints,
            ) for index = 1:3
        ]...,
    ) atol=1e-8

    @tea static function literal_range_model()
        mu ~ normal(0.0f0, 1.0f0)
        for i = 1:3
            {:y => i} ~ normal(mu, 1.0f0)
        end
        return mu
    end

    litrange_spec = modelspec(literal_range_model)
    @test litrange_spec.shape_specialized
    @test length(litrange_spec.choices[2].scopes) == 1
    @test !litrange_spec.choices[2].scopes[1].shape_specialized
    @test litrange_spec.choices[2].scopes[1].iterable == :(1:3)

    @tea static function pair_arg_address_model(addr)
        mu ~ normal(0.0f0, 1.0f0)
        {addr} ~ normal(mu, 1.0f0)
        return mu
    end

    pairarg_constraints = choicemap(((:obs, 2), 0.4f0))
    pairarg_trace, _ = generate(
        pair_arg_address_model,
        (:obs => 2,),
        pairarg_constraints;
        rng=MersenneTwister(303),
    )
    pairarg_params = parameter_vector(pairarg_trace)
    pairarg_expected =
        UncertainTea.logpdf(normal(0.0f0, 1.0f0), pairarg_trace[:mu]) +
        UncertainTea.logpdf(normal(pairarg_trace[:mu], 1.0f0), 0.4f0)
    @test pairarg_trace[:obs=>2] == 0.4f0
    @test logjoint(pair_arg_address_model, pairarg_params, (:obs => 2,), pairarg_constraints) ≈
          pairarg_expected atol=1e-6
end
