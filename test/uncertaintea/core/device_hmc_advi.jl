# PR 46: device-resident batched HMC and ADVI inner loops.
#
# All tests run on KernelAbstractions.CPU() (the authoritative reference backend);
# the GPU smoke test under test/gpu/ mirrors a subset on a MetalBackend at Float32.
# The device path keeps the RNG host-side, so its results are STATISTICALLY
# equivalent to -- not bitwise identical to -- the CPU `batched_hmc`/`batched_advi`
# path; the tests below compare posterior/variational analytics within tolerance.

using KernelAbstractions: CPU

# Local mean/std helpers (Statistics is not imported by the test harness).
devh_mean(x) = sum(x) / length(x)
function devh_std(x)
    m = devh_mean(x)
    accumulator = 0.0
    for value in x
        accumulator += (value - m)^2
    end
    return sqrt(accumulator / (length(x) - 1))
end

# Conjugate gaussian: mu ~ N(0,1), y ~ N(mu,1) observed at y = 0.3.
# Posterior precision = 2 -> variance 0.5 (sd sqrt(0.5)), mean = 0.3/2 = 0.15.
@tea static function devh_conjugate_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# dirichlet latent -> simplex transform: intentionally NOT device-lowerable.
@tea static function devh_dirichlet_model()
    theta ~ dirichlet([1.0, 1.0, 1.0])
    return theta
end

@testset "devh_hmc_conjugate" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    chains = batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        # 4 chains x 600/300 rather than 2 x 300/150: the posterior-std smoke
        # check below must hold for every supported randn stream (the ziggurat
        # changed in Julia 1.12), and shorter seeded runs leave the std
        # estimate of the autocorrelated chains too noisy -- the pre-1.12
        # stream under-adapts the step size at 150 warmup (acceptance pins at
        # 1.0, std comes out at half its true value) and single short chains
        # can under-disperse on any stream. Exact device-vs-host parity is
        # covered separately by devh_hmc_vs_cpu.
        num_chains=4,
        num_samples=600,
        num_warmup=300,
        backend=backend,
        precision=Float64,
        rng=MersenneTwister(46),
    )

    samples = posterior_array(chains)
    @test all(isfinite, samples)
    @test isapprox(devh_mean(samples), 0.15; atol=0.1)
    @test isapprox(devh_std(samples), sqrt(0.5); atol=0.15)
    @test all(<(1.2), rhat(chains))
    accept = acceptancerate(chains)
    @test 0.5 < accept < 1.0
end

@testset "devh_hmc_vs_cpu" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    device_chains = batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        num_chains=2,
        num_samples=300,
        num_warmup=150,
        backend=backend,
        precision=Float64,
        rng=MersenneTwister(52),
    )
    cpu_chains = batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        num_chains=2,
        num_samples=300,
        num_warmup=150,
        rng=MersenneTwister(52),
    )
    @test isapprox(devh_mean(posterior_array(device_chains)), devh_mean(posterior_array(cpu_chains)); atol=0.1)
end

@testset "devh_hmc_float32" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    chains = batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        num_chains=2,
        num_samples=300,
        num_warmup=150,
        backend=backend,
        precision=Float32,
        rng=MersenneTwister(46),
    )
    samples = posterior_array(chains)
    # HMCChain stores Float64; the device ran in Float32 and downloads convert up.
    @test eltype(samples) === Float64
    @test all(isfinite, samples)
    @test isapprox(devh_mean(samples), 0.15; atol=0.1)
end

@testset "devh_advi_conjugate" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    result = batched_advi(
        devh_conjugate_gauss,
        (),
        constraints;
        num_steps=300,
        backend=backend,
        rng=MersenneTwister(46),
    )
    mean_estimate = variational_mean(result)
    @test length(mean_estimate) == 1
    @test isapprox(mean_estimate[1], 0.15; atol=0.1)
    @test all(isfinite, result.elbo_history)
    quartile = length(result.elbo_history) ÷ 4
    first_quartile = devh_mean(result.elbo_history[1:quartile])
    last_quartile = devh_mean(result.elbo_history[(end - quartile + 1):end])
    @test last_quartile > first_quartile
end

@testset "devh_device_errors" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    # Per-chain adaptation is unsupported on the device backend.
    @test_throws ArgumentError batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        num_chains=2,
        num_samples=10,
        per_chain_adaptation=true,
        backend=backend,
    )
    # Unsupported (dirichlet/simplex latent) model raises the lowering ArgumentError.
    dirichlet_error = try
        batched_hmc(devh_dirichlet_model, (), choicemap(); num_chains=2, num_samples=10, backend=backend)
        nothing
    catch err
        err
    end
    @test dirichlet_error isa ArgumentError
    @test occursin("device_lowering_report", sprint(showerror, dirichlet_error))

    advi_error = try
        batched_advi(devh_dirichlet_model, (), choicemap(); num_steps=10, backend=backend)
        nothing
    catch err
        err
    end
    @test advi_error isa ArgumentError
    @test occursin("device_lowering_report", sprint(showerror, advi_error))
end
