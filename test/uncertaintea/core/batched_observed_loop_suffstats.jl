# Issue #146: sufficient-statistics fusion for iid exponential-family
# observation loops on the batched analytic gradient walker
# (src/batched/gradients/observed_loop.jl). A loop whose body is a single
# observed choice with loop-invariant parameters and shared constraints scores
# from a few numbers cached once per gradient cache: normal from the CENTERED
# stats (n, ybar, S2c), exponential from (n, sum y), poisson from (n, sum y,
# sum log y!). The fused reduction reassociates the observation sum, so it is
# tolerance-equal to the unfused tiers (which stay reachable through the
# `_BATCHED_GRADIENT_OBSERVED_LOOP_SUFFSTATS` seam), with identical NUTS tree
# depths on a fixed seed and a posterior matching the conjugate analytic
# answer. The centered normal form must hold in the |ybar| >> sigma regime
# where the naive power-sum form cancels catastrophically.

@tea static function bols_gauss_model(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

# |ybar| >> sigma: the observations sit near 1e6 with unit scale, so a
# power-sum reduction would lose ~11 digits; the centered form must not.
@tea static function bols_gauss_offset_model(n)
    mu ~ normal(1.0e6, 10.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

# Loop-varying mu: not fusable (and not hoistable), stays on the
# per-iteration tier.
@tea static function bols_gauss_varying_model(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu + 0.01 * i, s)
    end
    return mu
end

@tea static function bols_exponential_model(n)
    rate ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ exponential(rate)
    end
    return rate
end

@tea static function bols_poisson_model(n)
    lambda ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ poisson(lambda)
    end
    return lambda
end

# Conjugate check model: known unit observation scale, mu ~ N(0, 1) prior.
@tea static function bols_conjugate_model(n)
    mu ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, 1.0)
    end
    return mu
end

function bols_gradient_pair(cache, params)
    destination = zeros(Float64, size(params, 2))
    UncertainTea._batched_logjoint_and_gradient_unconstrained!(destination, cache, params)
    return destination, copy(cache.gradient_buffer)
end

function bols_with_seams(body; fast::Bool=true, hoist::Bool=true, suffstats::Bool=true)
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

@testset "bols_fused_matches_unfused_tiers" begin
    observation_count = 300
    data_rng = MersenneTwister(19)
    gauss_constraints =
        choicemap(((:y => i, 0.4 + 0.8 * randn(data_rng)) for i = 1:observation_count)...)
    exponential_constraints =
        choicemap(((:y => i, -log(rand(data_rng)) / 1.3) for i = 1:observation_count)...)
    poisson_constraints =
        choicemap(((:y => i, rand(data_rng, 0:9)) for i = 1:observation_count)...)

    cases = (
        (bols_gauss_model, gauss_constraints, 2),
        (bols_exponential_model, exponential_constraints, 1),
        (bols_poisson_model, poisson_constraints, 1),
    )
    for (model, constraints, parameter_count) in cases
        # parameter sweep: several seeds and unconstrained-scale magnitudes
        for seed in (11, 23, 47), scale in (0.1, 0.5, 1.5), num_chains in (4, 64)
            params = randn(MersenneTwister(seed), parameter_count, num_chains) .* scale
            cache = BatchedLogjointGradientCache(model, params, (observation_count,), constraints)
            # the fused tier only exists on the analytic backend tier
            @test cache.backend_cache !== nothing
            fused_logjoint, fused_gradient = bols_gradient_pair(cache, params)
            unfused_logjoint, unfused_gradient =
                bols_with_seams(() -> bols_gradient_pair(cache, params); suffstats=false)
            loop_logjoint, loop_gradient =
                bols_with_seams(() -> bols_gradient_pair(cache, params); suffstats=false, hoist=false)
            @test isapprox(fused_logjoint, unfused_logjoint; rtol=1e-10)
            @test isapprox(fused_gradient, unfused_gradient; rtol=1e-10)
            @test isapprox(fused_logjoint, loop_logjoint; rtol=1e-10)
            @test isapprox(fused_gradient, loop_gradient; rtol=1e-10)
            # repeated calls reuse the cached statistics deterministically
            repeat_logjoint, repeat_gradient = bols_gradient_pair(cache, params)
            @test repeat_logjoint == fused_logjoint
            @test repeat_gradient == fused_gradient
        end
    end
end

@testset "bols_centered_form_cancellation_regime" begin
    # ybar = 1e6, sigma ~ 1: the centered stats keep full precision against
    # the per-iteration tier (which computes each z = (y - mu) / sigma
    # directly and never forms a power sum)
    observation_count = 1000
    data_rng = MersenneTwister(29)
    constraints =
        choicemap(((:y => i, 1.0e6 + randn(data_rng)) for i = 1:observation_count)...)
    for seed in (5, 13), num_chains in (4, 32)
        rng = MersenneTwister(seed)
        params = vcat(1.0e6 .+ 0.7 .* randn(rng, 1, num_chains), 0.3 .* randn(rng, 1, num_chains))
        cache = BatchedLogjointGradientCache(bols_gauss_offset_model, params, (observation_count,), constraints)
        @test cache.backend_cache !== nothing
        fused_logjoint, fused_gradient = bols_gradient_pair(cache, params)
        loop_logjoint, loop_gradient =
            bols_with_seams(() -> bols_gradient_pair(cache, params); suffstats=false, hoist=false)
        @test isapprox(fused_logjoint, loop_logjoint; rtol=1e-10)
        @test isapprox(fused_gradient, loop_gradient; rtol=1e-10)
        @test all(isfinite, fused_logjoint)
        @test all(isfinite, fused_gradient)
    end
end

@testset "bols_unfusable_loops_fall_back" begin
    observation_count = 40
    num_chains = 8
    data_rng = MersenneTwister(43)

    # loop-varying mu: the fusable gate rejects the loop, so the default walk
    # is the per-iteration tier and matches the seam-off walk bitwise
    varying_constraints =
        choicemap(((:y => i, 0.01 * i + randn(data_rng)) for i = 1:observation_count)...)
    params = randn(MersenneTwister(3), 2, num_chains) .* 0.4
    cache = BatchedLogjointGradientCache(
        bols_gauss_varying_model,
        params,
        (observation_count,),
        varying_constraints,
    )
    @test cache.backend_cache !== nothing
    default_logjoint, default_gradient = bols_gradient_pair(cache, params)
    unfused_logjoint, unfused_gradient =
        bols_with_seams(() -> bols_gradient_pair(cache, params); suffstats=false)
    @test default_logjoint == unfused_logjoint
    @test default_gradient == unfused_gradient

    # data the closed form cannot represent: a negative exponential
    # observation / a non-count poisson observation score -Inf per
    # observation, so staging records `nothing` and the walk keeps the
    # per-iteration tier (bitwise identical to the seam-off walk)
    invalid_cases = (
        (
            bols_exponential_model,
            choicemap(
                ((:y => i, i == 7 ? -0.5 : -log(rand(data_rng))) for i = 1:observation_count)...,
            ),
        ),
        (
            bols_poisson_model,
            choicemap(((:y => i, i == 7 ? 2.5 : float(rand(data_rng, 0:5))) for i = 1:observation_count)...),
        ),
    )
    for (model, constraints) in invalid_cases
        params = randn(MersenneTwister(9), 1, num_chains) .* 0.4
        cache = BatchedLogjointGradientCache(model, params, (observation_count,), constraints)
        @test cache.backend_cache !== nothing
        default_logjoint, default_gradient = bols_gradient_pair(cache, params)
        unfused_logjoint, unfused_gradient =
            bols_with_seams(() -> bols_gradient_pair(cache, params); suffstats=false)
        @test all(==(-Inf), default_logjoint)
        @test default_logjoint == unfused_logjoint
        @test default_gradient == unfused_gradient
    end

    # empty loop: no statistics to fuse over; the walk stays finite and
    # matches the seam-off walk bitwise
    params = randn(MersenneTwister(21), 2, num_chains) .* 0.4
    cache = BatchedLogjointGradientCache(bols_gauss_model, params, (0,), choicemap())
    @test cache.backend_cache !== nothing
    default_logjoint, default_gradient = bols_gradient_pair(cache, params)
    unfused_logjoint, unfused_gradient =
        bols_with_seams(() -> bols_gradient_pair(cache, params); suffstats=false)
    @test all(isfinite, default_logjoint)
    @test default_logjoint == unfused_logjoint
    @test default_gradient == unfused_gradient
end

@testset "bols_nuts_tree_depths_and_conjugate_posterior" begin
    observation_count = 100
    num_chains = 4
    data_rng = MersenneTwister(11)
    observations = [0.5 + randn(data_rng) for _ = 1:observation_count]
    constraints = choicemap(((:y => i, observations[i]) for i = 1:observation_count)...)
    run_nuts() = batched_nuts(
        bols_conjugate_model,
        (observation_count,),
        constraints;
        num_chains=num_chains,
        num_samples=150,
        num_warmup=150,
        rng=MersenneTwister(42),
    )
    fused_chains = run_nuts()
    unfused_chains = bols_with_seams(run_nuts; suffstats=false)
    # a fixed-seed run through the fused gradient walks identical
    # trajectories: per-draw tree depths match the unfused tier exactly
    for (fused, unfused) in zip(fused_chains.chains, unfused_chains.chains)
        @test fused.tree_depths == unfused.tree_depths
    end
    # conjugate posterior: mu | y ~ N(sum(y) / (n + 1), 1 / (n + 1))
    posterior_mean = sum(observations) / (observation_count + 1)
    posterior_sd = sqrt(1 / (observation_count + 1))
    sampled_mean =
        sum(chain -> sum(chain.constrained_samples[1, :]), fused_chains.chains) /
        (num_chains * 150)
    @test isapprox(sampled_mean, posterior_mean; atol=4 * posterior_sd / sqrt(num_chains * 150 / 4))
end
