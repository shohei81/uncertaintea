# Manual Metal GPU smoke test for the device-resident batched logjoint.
#
# NOT part of the package test target and NOT run in CI. Requires a functional Metal
# GPU (Apple Silicon). See test/gpu/Project.toml for setup instructions.
#
# It runs the same parity checks as test/uncertaintea/core/part42.jl, but on a
# Metal.MetalBackend at Float32, and compares against the authoritative CPU
# `batched_logjoint_unconstrained` reference computed in Float64.

using Test
using UncertainTea
using Metal

@tea static function gpu_gauss_model(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i in 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

@tea static function gpu_bernoulli_model(n)
    p ~ beta(2.0, 3.0)
    for i in 1:n
        {:z => i} ~ bernoulli(p)
    end
    return p
end

function gpu_check_float32(dev32, ref)
    ok = true
    for (d, r) in zip(dev32, ref)
        ok &= isapprox(Float64(d), r; rtol=1e-3, atol=1e-3 * max(1.0, abs(r)))
    end
    return ok
end

@testset "device Metal GPU smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()

    # Float64 must be rejected on Metal via the precision guard.
    @test_throws ArgumentError DeviceBatchedWorkspace(gpu_gauss_model, 2; backend=backend, precision=Float64)

    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i in 1:5)
    params = [0.5 -0.3 1.2 0.05; 0.1 0.7 -0.2 0.4]
    ref = batched_logjoint_unconstrained(gpu_gauss_model, params, (5,), cm)
    dev = device_batched_logjoint(gpu_gauss_model, Float32.(params), (5,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(dev, ref)

    zs = [1.0, 0.0, 1.0, 1.0]
    cmb = choicemap((:z => i, zs[i]) for i in 1:4)
    pb = reshape([0.3, -0.8, 1.5], 1, 3)
    refb = batched_logjoint_unconstrained(gpu_bernoulli_model, pb, (4,), cmb)
    devb = device_batched_logjoint(gpu_bernoulli_model, Float32.(pb), (4,), cmb; backend=backend, precision=Float32)
    @test gpu_check_float32(devb, refb)
end

@testset "device Metal GPU gradient smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU gradient smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()

    # gaussian + gamma(log) latent with loop observations.
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i in 1:5)
    params = [0.5 -0.3 1.2 0.05; 0.1 0.7 -0.2 0.4]
    gref = batched_logjoint_gradient_unconstrained(gpu_gauss_model, params, (5,), cm)
    vref = batched_logjoint_unconstrained(gpu_gauss_model, params, (5,), cm)
    v, g = device_batched_logjoint_gradient(gpu_gauss_model, Float32.(params), (5,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(vec(Float64.(g)), vec(gref))
    @test gpu_check_float32(Float64.(v), vref)

    # bernoulli + beta(logit) latent.
    zs = [1.0, 0.0, 1.0, 1.0]
    cmb = choicemap((:z => i, zs[i]) for i in 1:4)
    pb = reshape([0.3, -0.8, 1.5], 1, 3)
    grefb = batched_logjoint_gradient_unconstrained(gpu_bernoulli_model, pb, (4,), cmb)
    _, gb = device_batched_logjoint_gradient(gpu_bernoulli_model, Float32.(pb), (4,), cmb; backend=backend, precision=Float32)
    @test gpu_check_float32(vec(Float64.(gb)), vec(grefb))
end
