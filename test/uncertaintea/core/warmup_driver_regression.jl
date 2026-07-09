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

    warmup_driver_bn = batched_nuts(
        warmup_driver_model,
        (),
        warmup_driver_constraints;
        num_chains=3,
        num_samples=20,
        num_warmup=30,
        rng=MersenneTwister(404),
    )
    warmup_driver_bn_chain = warmup_driver_bn.chains[1]
    # Values re-pinned after the batched-NUTS merge-cohort stale-select fix
    # (PR 6.4): the previous pins had captured runs where stale proposal
    # selections replayed old tree-proposal columns into the live proposals.
    if adaptation_pins_exact
        @test warmup_driver_bn_chain.step_size ≈ 1.2168785742992647 atol = 1e-12
    else
        @test 0.05 < warmup_driver_bn_chain.step_size < 10.0
    end
    @test length(warmup_driver_bn_chain.mass_matrix) == 1
    if adaptation_pins_exact
        @test warmup_driver_bn_chain.mass_matrix[1] ≈ 0.5636619744202114 atol = 1e-12
    else
        @test 0.01 < warmup_driver_bn_chain.mass_matrix[1] < 10.0
    end
    # Version-independent: the shared driver adapts one step size / mass matrix
    # for the whole batch, so every chain reports identical values.
    for other_chain in warmup_driver_bn.chains
        @test other_chain.step_size == warmup_driver_bn_chain.step_size
        @test other_chain.mass_matrix == warmup_driver_bn_chain.mass_matrix
    end
end
