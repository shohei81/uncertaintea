# Issue #140: the batched gradient walker's observed-loop shared-address fast
# path (src/batched/gradients/observed_loop.jl) must produce a gradient and
# logjoint BITWISE identical to the generic per-iteration body walk it
# replaces. Issue #141: staged observations plus the loop-invariant hoisted
# per-chain reduction reassociate the observation sum, so that tier is
# tolerance-equal instead, with identical NUTS tree depths on a fixed seed and
# a posterior that matches the conjugate analytic answer. Both replaced walks
# stay reachable through the `_BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH` /
# `_BATCHED_GRADIENT_OBSERVED_LOOP_HOIST` test seams.

# The bench_gauss shape: two latents (one log-transformed), loop-addressed
# normal observations with loop-invariant parameters (hoistable).
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
# iterator), so the parameter expression is re-evaluated per iteration and the
# hoisted tier does not apply.
@tea static function bolg_logistic_model(n)
    alpha ~ normal(0.0, 2.5)
    beta ~ normal(0.0, 2.5)
    for i = 1:n
        {:y => i} ~ bernoulli(1.0 / (1.0 + exp(-(alpha + beta * (i - 25.0) / 25.0))))
    end
    return alpha
end

# Conjugate check model: known unit observation scale, mu ~ N(0, 1) prior.
@tea static function bolg_conjugate_model(n)
    mu ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, 1.0)
    end
    return mu
end

function bolg_gradient_pair(cache, params)
    destination = zeros(Float64, size(params, 2))
    UncertainTea._batched_logjoint_and_gradient_unconstrained!(destination, cache, params)
    return destination, copy(cache.gradient_buffer)
end

# The issue #146 sufficient-statistics tier sits above the tiers under test
# here (see batched_observed_loop_suffstats.jl); pinning suffstats=false keeps
# each arm on exactly the tier its testset names.
function bolg_with_seams(body; fast::Bool=true, hoist::Bool=true, suffstats::Bool=true)
    previous_fast = UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH[]
    previous_hoist = UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_HOIST[]
    previous_suffstats = UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_SUFFSTATS[]
    UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH[] = fast
    UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_HOIST[] = hoist
    UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_SUFFSTATS[] = suffstats
    try
        return body()
    finally
        UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH[] = previous_fast
        UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_HOIST[] = previous_hoist
        UncertainTea._BATCHED_GRADIENT_OBSERVED_LOOP_SUFFSTATS[] = previous_suffstats
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
        (bolg_gauss_model, gauss_constraints, false),
        (bolg_logistic_model, logistic_constraints, true),
    )
    for (model, constraints, default_is_bitwise) in cases
        for seed in (11, 23, 47), num_chains in (4, 64)
            rng = MersenneTwister(seed)
            params = randn(rng, 2, num_chains) .* 0.4
            cache = BatchedLogjointGradientCache(model, params, (observation_count,), constraints)
            # the fast path only exists on the analytic backend tier
            @test cache.backend_cache !== nothing
            # #140 (+ #141 staged observations): the per-iteration fast path is
            # bitwise identical to the generic body walk
            fast_logjoint, fast_gradient =
                bolg_with_seams(() -> bolg_gradient_pair(cache, params); hoist=false, suffstats=false)
            generic_logjoint, generic_gradient =
                bolg_with_seams(() -> bolg_gradient_pair(cache, params); fast=false)
            @test fast_logjoint == generic_logjoint
            @test fast_gradient == generic_gradient
            if default_is_bitwise
                # loop-varying parameters: the default path stays on the
                # per-iteration tier, so it is bitwise as well
                default_logjoint, default_gradient = bolg_gradient_pair(cache, params)
                @test default_logjoint == generic_logjoint
                @test default_gradient == generic_gradient
            end
        end
    end
end

@testset "bolg_fast_path_per_chain_constraints" begin
    # per-chain choicemaps take the vector-constraint lookup inside the fast
    # path (no observation staging); results must still match the generic walk
    # bitwise
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
    generic_logjoint, generic_gradient =
        bolg_with_seams(() -> bolg_gradient_pair(cache, params); fast=false)
    @test fast_logjoint == generic_logjoint
    @test fast_gradient == generic_gradient
end

@testset "bolg_hoisted_reduction_tolerance" begin
    # #141: loop-invariant normal parameters take the hoisted per-chain
    # reduction; the reassociated observation sum is tolerance-equal to the
    # per-iteration tier, not bitwise
    observation_count = 200
    data_rng = MersenneTwister(31)
    constraints =
        choicemap(((:y => i, 0.4 + 0.8 * randn(data_rng)) for i = 1:observation_count)...)
    for seed in (3, 17), num_chains in (4, 64)
        params = randn(MersenneTwister(seed), 2, num_chains) .* 0.4
        cache = BatchedLogjointGradientCache(bolg_gauss_model, params, (observation_count,), constraints)
        @test cache.backend_cache !== nothing
        hoisted_logjoint, hoisted_gradient =
            bolg_with_seams(() -> bolg_gradient_pair(cache, params); suffstats=false)
        loop_logjoint, loop_gradient =
            bolg_with_seams(() -> bolg_gradient_pair(cache, params); hoist=false, suffstats=false)
        @test isapprox(hoisted_logjoint, loop_logjoint; rtol=1e-9)
        @test isapprox(hoisted_gradient, loop_gradient; rtol=1e-9)
        # repeated calls reuse the staged observation vector deterministically
        repeat_logjoint, repeat_gradient =
            bolg_with_seams(() -> bolg_gradient_pair(cache, params); suffstats=false)
        @test repeat_logjoint == hoisted_logjoint
        @test repeat_gradient == hoisted_gradient
    end
end

@testset "bolg_nuts_tree_depths_and_conjugate_posterior" begin
    observation_count = 100
    num_chains = 4
    data_rng = MersenneTwister(11)
    observations = [0.5 + randn(data_rng) for _ = 1:observation_count]
    constraints = choicemap(((:y => i, observations[i]) for i = 1:observation_count)...)
    run_nuts() = batched_nuts(
        bolg_conjugate_model,
        (observation_count,),
        constraints;
        num_chains=num_chains,
        num_samples=150,
        num_warmup=150,
        rng=MersenneTwister(42),
    )
    hoisted_chains = bolg_with_seams(run_nuts; suffstats=false)
    loop_chains = bolg_with_seams(run_nuts; hoist=false, suffstats=false)
    # a fixed-seed run through the hoisted gradient walks identical
    # trajectories: per-draw tree depths match the per-iteration tier exactly
    for (hoisted, loop) in zip(hoisted_chains.chains, loop_chains.chains)
        @test hoisted.tree_depths == loop.tree_depths
    end
    # conjugate posterior: mu | y ~ N(sum(y) / (n + 1), 1 / (n + 1))
    posterior_mean = sum(observations) / (observation_count + 1)
    posterior_sd = sqrt(1 / (observation_count + 1))
    sampled_mean =
        sum(chain -> sum(chain.constrained_samples[1, :]), hoisted_chains.chains) /
        (num_chains * 150)
    @test isapprox(sampled_mean, posterior_mean; atol=4 * posterior_sd / sqrt(num_chains * 150 / 4))
end
