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
    @test warmup_driver_hmc.step_size ≈ 0.80242751762314313 atol = 1e-12
    @test length(warmup_driver_hmc.mass_matrix) == 1
    @test warmup_driver_hmc.mass_matrix[1] ≈ 0.74579902097242412 atol = 1e-12

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
    @test warmup_driver_bn_chain.step_size ≈ 1.4269329081220101 atol = 1e-12
    @test length(warmup_driver_bn_chain.mass_matrix) == 1
    @test warmup_driver_bn_chain.mass_matrix[1] ≈ 0.54987846326449419 atol = 1e-12
end
