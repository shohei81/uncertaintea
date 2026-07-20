@testset "dist_exponential_poisson" begin
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
    @test logjoint(poisson_rate_model, poisson_params, (), poisson_constraints) ≈
          assess(
        poisson_rate_model,
        (),
        choicemap((:rate, poisson_trace[:rate]), (:y, 3)),
    ) atol=1e-6
    @test logjoint_unconstrained(poisson_rate_model, poisson_unconstrained, (), poisson_constraints) ≈
          logjoint(poisson_rate_model, poisson_params, (), poisson_constraints) +
          poisson_unconstrained[1] atol=1e-6
    @test poisson_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                poisson_rate_model,
                poisson_batch_params[:, index],
                (),
                poisson_batch_constraints[index],
            ) for index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(poisson_batch_cache.backend_cache)
    @test isnothing(poisson_batch_cache.flat_cache)
    @test isempty(poisson_batch_cache.column_caches)
    @test poisson_combined_values ≈ [
        logjoint_unconstrained(
            poisson_rate_model,
            poisson_batch_params[:, index],
            (),
            poisson_batch_constraints[index],
        ) for index = 1:3
    ] atol=1e-8
    @test poisson_combined_gradient === poisson_batch_cache.gradient_buffer
    @test poisson_combined_gradient ≈ poisson_batch_gradient atol=1e-8
end

# Issue #74: large-rate Poisson draws use the PTRS transformed rejection
# sampler; the Knuth product loop underflows exp(-lambda) and saturated draws
# around 745 regardless of the rate.
@testset "poisson_large_rate_sampler" begin
    ptrs_rng = MersenneTwister(11)
    ptrs_n = 200_000
    ptrs_draws = [rand(ptrs_rng, poisson(1000.0)) for _ = 1:ptrs_n]
    ptrs_mean = sum(ptrs_draws) / ptrs_n
    ptrs_sd = sqrt(sum((x - ptrs_mean)^2 for x in ptrs_draws) / (ptrs_n - 1))
    # mean 1000 +- 3 * sd / sqrt(n), sd = sqrt(1000) ~ 31.6
    @test abs(ptrs_mean - 1000.0) < 3 * sqrt(1000.0) / sqrt(ptrs_n)
    @test abs(ptrs_sd - sqrt(1000.0)) < 0.5
    # the old sampler could not exceed ~745; a healthy one covers mean +- 4 sd
    @test maximum(ptrs_draws) > 1100
    @test minimum(ptrs_draws) > 800
    @test all(x -> x >= 0, ptrs_draws)

    # both sides of the algorithm threshold stay unbiased
    for rate in (30.0, 30.5)
        edge_rng = MersenneTwister(12)
        edge_n = 50_000
        edge_mean = sum(rand(edge_rng, poisson(rate)) for _ = 1:edge_n) / edge_n
        @test abs(edge_mean - rate) < 3 * sqrt(rate) / sqrt(edge_n)
    end

    # small-lambda behavior is unchanged (Knuth path): pmf frequency check
    small_rng = MersenneTwister(13)
    small_n = 200_000
    small_draws = [rand(small_rng, poisson(1.0)) for _ = 1:small_n]
    for k = 0:3
        frequency = count(==(k), small_draws) / small_n
        pmf = exp(UncertainTea.logpdf(poisson(1.0), k))
        @test abs(frequency - pmf) < 0.006
    end

    # the Gamma-Poisson negativebinomial sampler inherits the fix:
    # r = 50, p = 0.05 has mean r(1-p)/p = 950, far past the old saturation
    negbin_rng = MersenneTwister(14)
    negbin_n = 50_000
    negbin_mean = sum(rand(negbin_rng, negativebinomial(50.0, 0.05)) for _ = 1:negbin_n) / negbin_n
    @test abs(negbin_mean - 950.0) < 3 * sqrt(50 * 0.95 / 0.05^2) / sqrt(negbin_n)
end
