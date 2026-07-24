@testset "shared warmup driver regression" begin
    @tea static function warmup_driver_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 1.0f0)
        return mu
    end

    warmup_driver_constraints = choicemap((:y, 0.3f0))

    warmup_driver_hmc = hmc(
        warmup_driver_model,
        (),
        warmup_driver_constraints;
        num_samples=20,
        num_warmup=30,
        rng=MersenneTwister(101),
    )
    if adaptation_pins_exact
        @test warmup_driver_hmc.step_size ≈ 0.80242751762314313 atol = 1e-12
    else
        @test 0.05 < warmup_driver_hmc.step_size < 10.0
    end
    @test length(warmup_driver_hmc.mass_matrix) == 1
    if adaptation_pins_exact
        @test warmup_driver_hmc.mass_matrix[1] ≈ 0.74579902097242412 atol = 1e-12
    else
        @test 0.01 < warmup_driver_hmc.mass_matrix[1] < 10.0
    end

    # per_chain_adaptation=false: this regression pins the SHARED driver;
    # since issue #137 the host default is per-chain adaptation.
    warmup_driver_bn = batched_nuts(
        warmup_driver_model,
        (),
        warmup_driver_constraints;
        num_chains=3,
        num_samples=20,
        num_warmup=30,
        per_chain_adaptation=false,
        rng=MersenneTwister(404),
    )
    warmup_driver_bn_chain = warmup_driver_bn.chains[1]
    # Values re-pinned after the issue #159 biased progressive merge: the
    # doubling-merge proposal swap is now Stan's biased
    # min(1, w_subtree/w_continuation), so the same RNG draws select
    # different proposals and the warmup trajectory shifts. (Previous
    # re-pins: issue #93/#81 canonical invalid-subtree discard and
    # first-crossing step-size search; batched-NUTS merge-cohort
    # stale-select fix, PR 6.4.)
    if adaptation_pins_exact
        @test warmup_driver_bn_chain.step_size ≈ 1.2243640004920302 atol = 1e-12
    else
        @test 0.05 < warmup_driver_bn_chain.step_size < 10.0
    end
    @test length(warmup_driver_bn_chain.mass_matrix) == 1
    if adaptation_pins_exact
        @test warmup_driver_bn_chain.mass_matrix[1] ≈ 0.3968463881646532 atol = 1e-12
    else
        @test 0.01 < warmup_driver_bn_chain.mass_matrix[1] < 10.0
    end
    # Version-independent: the shared driver adapts one step size / mass matrix
    # for the whole batch, so every chain reports identical values.
    for other_chain in warmup_driver_bn.chains
        @test other_chain.step_size == warmup_driver_bn_chain.step_size
        @test other_chain.mass_matrix == warmup_driver_bn_chain.mass_matrix
    end

    # Issue #81: the shared batched reasonable-step-size search must stop at
    # the FIRST crossing of the 0.5 mean-acceptance target, keeping the
    # direction chosen on the initial trial (like the per-chain variant).
    # Before the fix the crossing condition was unreachable, the loop ran to
    # the iteration cap while oscillating between the two bracketing values,
    # and the result depended on the parity of the cap: searches started from
    # s0 and 2*s0 (same momentum draw) returned different brackets. Post-fix
    # both must land on the same first-crossing step size.
    stepsearch_num_chains = 2
    stepsearch_num_params = UncertainTea.parametercount(
        UncertainTea.parameterlayout(warmup_driver_model),
    )
    stepsearch_position = randn(
        MersenneTwister(1),
        stepsearch_num_params,
        stepsearch_num_chains,
    )
    stepsearch_gradient = Matrix{Float64}(undef, stepsearch_num_params, stepsearch_num_chains)
    stepsearch_logjoint = Vector{Float64}(undef, stepsearch_num_chains)
    for stepsearch_chain = 1:stepsearch_num_chains
        stepsearch_cache = UncertainTea._logjoint_gradient_cache(
            warmup_driver_model,
            stepsearch_position[:, stepsearch_chain],
            (),
            warmup_driver_constraints,
        )
        stepsearch_gradient[:, stepsearch_chain] = UncertainTea._logjoint_gradient!(
            stepsearch_cache,
            stepsearch_position[:, stepsearch_chain],
        )
        stepsearch_logjoint[stepsearch_chain] = logjoint_unconstrained(
            warmup_driver_model,
            stepsearch_position[:, stepsearch_chain],
            (),
            warmup_driver_constraints,
        )
    end
    stepsearch_workspace = UncertainTea.BatchedHMCWorkspace(
        warmup_driver_model,
        stepsearch_position,
        (),
        warmup_driver_constraints,
    )
    stepsearch(initial_step) = UncertainTea._find_reasonable_batched_step_size(
        stepsearch_workspace,
        warmup_driver_model,
        stepsearch_position,
        stepsearch_logjoint,
        stepsearch_gradient,
        ones(stepsearch_num_params),
        (),
        warmup_driver_constraints,
        initial_step,
        1000.0,
        MersenneTwister(44),
    )
    stepsearch_down = stepsearch(10.0)
    stepsearch_down_shifted = stepsearch(20.0)
    # Same crossing regardless of the start parity, reached by halving.
    @test stepsearch_down == stepsearch_down_shifted
    @test isinteger(log2(10.0 / stepsearch_down))
    @test isinteger(log2(20.0 / stepsearch_down_shifted))
    @test stepsearch_down < 10.0
    stepsearch_up = stepsearch(1.0e-4)
    stepsearch_up_shifted = stepsearch(2.0e-4)
    @test stepsearch_up == stepsearch_up_shifted
    @test isinteger(log2(stepsearch_up / 1.0e-4))
    @test stepsearch_up > 1.0e-4
    # The two directions bracket each other within the doubling grid.
    @test stepsearch_up / 4 <= stepsearch_down <= stepsearch_up * 4
end
