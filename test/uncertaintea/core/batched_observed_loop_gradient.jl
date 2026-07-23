# Issue #140: the batched gradient walker's observed-loop shared-address fast
# path (src/batched/gradients/observed_loop.jl) must produce a gradient and
# logjoint BITWISE identical to the generic per-iteration body walk it
# replaces. The generic walk stays reachable through the
# `_BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH` test seam.

# The bench_gauss shape: two latents (one log-transformed), loop-addressed
# normal observations.
@tea static function bolg_gauss_model(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

# The bench_logistic shape reduced to a backend-lowerable form: loop-addressed
# Bernoulli observations whose probability is LOOP-VARYING (it reads the
# iterator), so the parameter expression is re-evaluated per iteration.
@tea static function bolg_logistic_model(n)
    alpha ~ normal(0.0, 2.5)
    beta ~ normal(0.0, 2.5)
    for i = 1:n
        {:y => i} ~ bernoulli(1.0 / (1.0 + exp(-(alpha + beta * (i - 25.0) / 25.0))))
    end
    return alpha
end

function bolg_gradient_pair(cache, params)
    destination = zeros(Float64, size(params, 2))
    UncertainTea._batched_logjoint_and_gradient_unconstrained!(destination, cache, params)
    return destination, copy(cache.gradient_buffer)
end

function bolg_generic_gradient_pair(cache, params)
    UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH[] = false
    try
        return bolg_gradient_pair(cache, params)
    finally
        UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH[] = true
    end
end

@testset "bolg_fast_path_bitwise_identity" begin
    observation_count = 50
    data_rng = MersenneTwister(7)
    gauss_constraints =
        choicemap(((:y => i, 0.4 + 0.8 * randn(data_rng)) for i = 1:observation_count)...)
    logistic_constraints = choicemap(
        (
            (:y => i, rand(data_rng) < 1 / (1 + exp(-0.7 * (i - 25.0) / 25.0))) for
            i = 1:observation_count
        )...,
    )

    cases = (
        (bolg_gauss_model, (observation_count,), gauss_constraints, 2),
        (bolg_logistic_model, (observation_count,), logistic_constraints, 2),
    )
    for (model, args, constraints, parameter_count) in cases
        for seed in (11, 23, 47), num_chains in (4, 64)
            rng = MersenneTwister(seed)
            params = randn(rng, parameter_count, num_chains) .* 0.4
            cache = BatchedLogjointGradientCache(model, params, args, constraints)
            # the fast path only exists on the analytic backend tier
            @test cache.backend_cache !== nothing
            fast_logjoint, fast_gradient = bolg_gradient_pair(cache, params)
            generic_logjoint, generic_gradient = bolg_generic_gradient_pair(cache, params)
            @test fast_logjoint == generic_logjoint
            @test fast_gradient == generic_gradient
        end
    end
end

@testset "bolg_fast_path_per_chain_constraints" begin
    # per-chain choicemaps take the vector-constraint lookup inside the fast
    # path; results must still match the generic walk bitwise
    observation_count = 20
    num_chains = 4
    data_rng = MersenneTwister(97)
    constraints = [
        choicemap(((:y => i, 0.2 * chain + randn(data_rng)) for i = 1:observation_count)...) for
        chain = 1:num_chains
    ]
    params = randn(MersenneTwister(5), 2, num_chains) .* 0.3
    cache = BatchedLogjointGradientCache(bolg_gauss_model, params, (observation_count,), constraints)
    @test cache.backend_cache !== nothing
    fast_logjoint, fast_gradient = bolg_gradient_pair(cache, params)
    generic_logjoint, generic_gradient = bolg_generic_gradient_pair(cache, params)
    @test fast_logjoint == generic_logjoint
    @test fast_gradient == generic_gradient
end
