# PR 6.4: mask-based iterative-doubling batched NUTS (tree_strategy=:masked).
#
# The masked path (src/inference/nuts/masked_doubling.jl) runs every chain
# through the same doubling round in lockstep with active masks, so every
# leapfrog step is one full-width batched gradient call. It is statistically
# equivalent to -- not bitwise identical to -- the default :hybrid path; the
# tests below compare posterior analytics within tolerance and pin the masked
# path's own determinism under a seed.

# Local mean/std helpers (Statistics is not imported by the test harness).
mnuts_mean(x) = sum(x) / length(x)
function mnuts_std(x)
    m = mnuts_mean(x)
    accumulator = 0.0
    for value in x
        accumulator += (value - m)^2
    end
    return sqrt(accumulator / (length(x) - 1))
end

# Conjugate gaussian: mu ~ N(0,1), y ~ N(mu,1) observed at y = 0.3.
# Posterior mean 0.15, variance 0.5.
@tea static function mnuts_conjugate_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# Two-parameter model: location and log-scale.
@tea static function mnuts_two_param()
    mu ~ normal(0.0, 1.0)
    log_sigma ~ normal(0.0, 0.5)
    {:y} ~ normal(mu, exp(log_sigma))
    return mu
end

function mnuts_run(model, constraints, strategy, seed; kwargs...)
    return batched_nuts(
        model,
        (),
        constraints;
        num_chains=4,
        num_samples=300,
        num_warmup=200,
        tree_strategy=strategy,
        rng=MersenneTwister(seed),
        kwargs...,
    )
end

@testset "mnuts_conjugate_gauss_hybrid_vs_masked" begin
    constraints = choicemap((:y, 0.3))
    hybrid = mnuts_run(mnuts_conjugate_gauss, constraints, :hybrid, 481)
    masked = mnuts_run(mnuts_conjugate_gauss, constraints, :masked, 481)

    hybrid_draws = posterior_array(hybrid)
    masked_draws = posterior_array(masked)
    @test all(isfinite, masked_draws)
    @test isapprox(mnuts_mean(masked_draws), 0.15; atol=0.1)
    @test isapprox(mnuts_std(masked_draws), sqrt(0.5); atol=0.15)
    @test abs(mnuts_mean(masked_draws) - mnuts_mean(hybrid_draws)) < 0.1
    @test abs(mnuts_std(masked_draws) - mnuts_std(hybrid_draws)) < 0.15
    @test all(<(1.2), rhat(hybrid))
    @test all(<(1.2), rhat(masked))

    hybrid_depths = reduce(vcat, treedepths(hybrid))
    masked_depths = reduce(vcat, treedepths(masked))
    @test abs(mnuts_mean(masked_depths) - mnuts_mean(hybrid_depths)) < 1.0
end

@testset "mnuts_two_param_hybrid_vs_masked" begin
    constraints = choicemap((:y, 0.7))
    hybrid = mnuts_run(mnuts_two_param, constraints, :hybrid, 482)
    masked = mnuts_run(mnuts_two_param, constraints, :masked, 482)

    hybrid_draws = posterior_array(hybrid)
    masked_draws = posterior_array(masked)
    @test all(isfinite, masked_draws)
    for parameter_index = 1:2
        hybrid_param = hybrid_draws[:, :, parameter_index]
        masked_param = masked_draws[:, :, parameter_index]
        @test abs(mnuts_mean(masked_param) - mnuts_mean(hybrid_param)) < 0.1
        @test abs(mnuts_std(masked_param) - mnuts_std(hybrid_param)) < 0.15
    end
    @test all(<(1.2), rhat(hybrid))
    @test all(<(1.2), rhat(masked))

    hybrid_depths = reduce(vcat, treedepths(hybrid))
    masked_depths = reduce(vcat, treedepths(masked))
    @test abs(mnuts_mean(masked_depths) - mnuts_mean(hybrid_depths)) < 1.0
end

@testset "mnuts_masked_deterministic" begin
    constraints = choicemap((:y, 0.3))
    first_run = mnuts_run(mnuts_conjugate_gauss, constraints, :masked, 483)
    second_run = mnuts_run(mnuts_conjugate_gauss, constraints, :masked, 483)
    for chain_index in eachindex(first_run.chains)
        first_chain = first_run.chains[chain_index]
        second_chain = second_run.chains[chain_index]
        @test first_chain.unconstrained_samples == second_chain.unconstrained_samples
        @test first_chain.logjoint_values == second_chain.logjoint_values
        @test first_chain.acceptance_stats == second_chain.acceptance_stats
        @test first_chain.divergent == second_chain.divergent
        @test first_chain.tree_depths == second_chain.tree_depths
        @test first_chain.integration_steps == second_chain.integration_steps
    end
end

@testset "mnuts_masked_diagnostics_coherent" begin
    constraints = choicemap((:y, 0.3))
    max_tree_depth = 6
    masked = batched_nuts(
        mnuts_conjugate_gauss,
        (),
        constraints;
        num_chains=4,
        num_samples=300,
        num_warmup=200,
        max_tree_depth=max_tree_depth,
        tree_strategy=:masked,
        rng=MersenneTwister(484),
    )
    depths = reduce(vcat, treedepths(masked))
    @test all(d -> 1 <= d <= max_tree_depth, depths)
    divergent_flags = reduce(vcat, [chain.divergent for chain in masked.chains])
    @test mnuts_mean(Float64.(divergent_flags)) < 0.2
end

@testset "mnuts_masked_per_chain_adaptation" begin
    constraints = choicemap((:y, 0.7))
    masked = batched_nuts(
        mnuts_two_param,
        (),
        constraints;
        num_chains=4,
        num_samples=200,
        num_warmup=150,
        per_chain_adaptation=true,
        tree_strategy=:masked,
        rng=MersenneTwister(485),
    )
    @test all(isfinite, posterior_array(masked))
    @test all(<(1.2), rhat(masked))
end

@testset "mnuts_tree_strategy_validation" begin
    constraints = choicemap((:y, 0.3))
    @test_throws ArgumentError batched_nuts(
        mnuts_conjugate_gauss,
        (),
        constraints;
        num_chains=2,
        num_samples=5,
        tree_strategy=:bogus,
    )
end
