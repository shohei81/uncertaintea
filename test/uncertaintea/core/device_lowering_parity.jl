# PR 42: first real device-execution phase -- device lowering + a device-resident
# batched (unconstrained) logjoint fused into a single KernelAbstractions kernel.
# All tests here run on KernelAbstractions.CPU() (the default backend); the GPU
# smoke test under test/gpu/ mirrors these on a MetalBackend and is intentionally
# not wired into CI.
#
# The device path takes UNCONSTRAINED parameters and folds the transform log-abs-det
# in-kernel, so the authoritative CPU counterpart is `batched_logjoint_unconstrained`.

# --- shared device-lowerable models -----------------------------------------
# gaussian mean/scale with a log-transformed (gamma) latent and loop observations.
@tea static function dev_gauss_model(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

# lognormal + gamma latents with an exp-derived scale argument.
@tea static function dev_lognormal_gamma_model(n)
    a ~ lognormal(0.0, 1.0)
    b ~ gamma(3.0, 2.0)
    scale = exp(a)
    for i = 1:n
        {:y => i} ~ normal(b, scale)
    end
    return a
end

# bernoulli observations governed by a logit-transformed (beta) latent probability.
@tea static function dev_bernoulli_model(n)
    p ~ beta(2.0, 3.0)
    for i = 1:n
        {:z => i} ~ bernoulli(p)
    end
    return p
end

# dirichlet latent -> simplex (vector) transform: intentionally NOT supported.
@tea static function dev_dirichlet_model()
    theta ~ dirichlet([1.0, 1.0, 1.0])
    return theta
end

# issue #12 group 1 (continuous): studentt (identity latent), inversegamma and
# weibull (log latents), studentt loop observations with a derived scale.
@tea static function dev_heavytail_model(n)
    loc ~ studentt(4.0, 0.0, 1.0)
    s ~ inversegamma(3.0, 2.0)
    w ~ weibull(2.0, 1.5)
    for i = 1:n
        {:y => i} ~ studentt(5.0, loc, s + w)
    end
    return loc
end

# issue #12 group 1 (discrete): binomial/geometric/negativebinomial observations and
# categorical loop observations, all driven by one beta (logit) latent probability.
@tea static function dev_count_model(n)
    p ~ beta(2.0, 2.0)
    {:k} ~ binomial(20, p)
    {:g} ~ geometric(p)
    {:nb} ~ negativebinomial(4.0, p)
    for i = 1:n
        {:c => i} ~ categorical(p / 2.0, p / 2.0, 1.0 - p)
    end
    return p
end

# helper: assert Float32 device output matches a Float64 reference with the
# rtol 1e-4 / atol 1e-4*max(1,|ref|) contract, elementwise.
function dev_check_float32(dev32, ref)
    ok = true
    for (d, r) in zip(dev32, ref)
        ok &= isapprox(Float64(d), r; rtol=1e-4, atol=1e-4 * max(1.0, abs(r)))
    end
    return ok
end

@testset "dev_lowering_report" begin
    supported, issues = device_lowering_report(dev_gauss_model)
    @test supported
    @test isempty(issues)

    supported_ln, _ = device_lowering_report(dev_lognormal_gamma_model)
    @test supported_ln
    supported_bern, _ = device_lowering_report(dev_bernoulli_model)
    @test supported_bern

    dir_supported, dir_issues = device_lowering_report(dev_dirichlet_model)
    @test !dir_supported
    @test !isempty(dir_issues)
    @test any(occursin("dirichlet", lowercase(issue)) for issue in dir_issues)
end

@testset "dev_plan_isbits" begin
    ys = [0.4, -0.7, 1.1, 0.2]
    cm = choicemap((:y => i, ys[i]) for i = 1:4)
    ws = DeviceBatchedWorkspace(dev_gauss_model, 3; args=(4,), constraints=cm)
    @test isbits(ws.plan)
    @test ws.plan isa DeviceExecutionPlan{Float64}

    ws32 = DeviceBatchedWorkspace(dev_gauss_model, 3; args=(4,), constraints=cm, precision=Float32)
    @test isbits(ws32.plan)
end

@testset "dev_device_loggamma_grid" begin
    max_rel64 = 0.0
    max_rel32 = 0.0
    for x = 0.05:0.05:20.0
        ref = UncertainTea.loggamma(x)
        d64 = UncertainTea._device_loggamma(x)
        max_rel64 = max(max_rel64, abs(d64 - ref) / max(1.0, abs(ref)))
        d32 = UncertainTea._device_loggamma(Float32(x))
        max_rel32 = max(max_rel32, abs(Float64(d32) - ref) / max(1.0, abs(ref)))
    end
    @test max_rel64 < 1e-10
    # Float32 accuracy is bounded by ~1 ulp of Float32 (~1.2e-7) accumulated over
    # the Lanczos sum; the achievable worst-case relative error is just above 1e-6.
    @test max_rel32 < 2e-6
end

@testset "dev_numerical_parity" begin
    # --- model 1: gaussian with loop observations, log latent ---
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2 0.05; 0.1 0.7 -0.2 0.4]
    dev = device_batched_logjoint(dev_gauss_model, params, (5,), cm)
    ref = batched_logjoint_unconstrained(dev_gauss_model, params, (5,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_gauss_model, Float32.(params), (5,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    # --- model 2: lognormal + gamma latents, exp-derived arg ---
    ys2 = [0.4, 1.1, -0.7]
    cm2 = choicemap((:y => i, ys2[i]) for i = 1:3)
    params2 = [0.5 -0.3 0.9; 0.1 0.7 -0.2]
    dev2 = device_batched_logjoint(dev_lognormal_gamma_model, params2, (3,), cm2)
    ref2 = batched_logjoint_unconstrained(dev_lognormal_gamma_model, params2, (3,), cm2)
    @test dev2 ≈ ref2 rtol = 1e-12
    dev2_32 = device_batched_logjoint(dev_lognormal_gamma_model, Float32.(params2), (3,), cm2; precision=Float32)
    @test dev_check_float32(dev2_32, ref2)

    # --- model 3: bernoulli observations, logit latent ---
    zs = [1.0, 0.0, 1.0, 1.0]
    cm3 = choicemap((:z => i, zs[i]) for i = 1:4)
    params3 = reshape([0.3, -0.8, 1.5], 1, 3)
    dev3 = device_batched_logjoint(dev_bernoulli_model, params3, (4,), cm3)
    ref3 = batched_logjoint_unconstrained(dev_bernoulli_model, params3, (4,), cm3)
    @test dev3 ≈ ref3 rtol = 1e-12
    dev3_32 = device_batched_logjoint(dev_bernoulli_model, Float32.(params3), (4,), cm3; precision=Float32)
    @test dev_check_float32(dev3_32, ref3)
end

@testset "dev_numerical_parity_group1" begin
    supported_ht, ht_issues = device_lowering_report(dev_heavytail_model)
    @test supported_ht
    @test isempty(ht_issues)
    supported_ct, ct_issues = device_lowering_report(dev_count_model)
    @test supported_ct
    @test isempty(ct_issues)

    # --- continuous: studentt / inversegamma / weibull ---
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2; -0.4 0.2 0.6]
    dev = device_batched_logjoint(dev_heavytail_model, params, (5,), cm)
    ref = batched_logjoint_unconstrained(dev_heavytail_model, params, (5,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_heavytail_model, Float32.(params), (5,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    # --- discrete: binomial / geometric / negativebinomial / categorical ---
    cs = [1.0, 3.0, 2.0, 3.0]
    cm2 = choicemap((:k, 7.0), (:g, 3.0), (:nb, 5.0), ((:c => i, cs[i]) for i = 1:4)...)
    params2 = reshape([0.3, -0.8, 1.5], 1, 3)
    dev2 = device_batched_logjoint(dev_count_model, params2, (4,), cm2)
    ref2 = batched_logjoint_unconstrained(dev_count_model, params2, (4,), cm2)
    @test dev2 ≈ ref2 rtol = 1e-12
    dev2_32 = device_batched_logjoint(dev_count_model, Float32.(params2), (4,), cm2; precision=Float32)
    @test dev_check_float32(dev2_32, ref2)

    # out-of-support discrete observations must land on -Inf, matching the CPU path.
    cm_oob = choicemap((:k, 25.0), (:g, 3.0), (:nb, 5.0), ((:c => i, cs[i]) for i = 1:4)...)
    dev_oob = device_batched_logjoint(dev_count_model, params2, (4,), cm_oob)
    @test all(==(-Inf), dev_oob)
    cm_cat_oob = choicemap((:k, 7.0), (:g, 3.0), (:nb, 5.0), (:c => 1, 4.0), ((:c => i, cs[i]) for i = 2:4)...)
    dev_cat_oob = device_batched_logjoint(dev_count_model, params2, (4,), cm_cat_oob)
    @test all(==(-Inf), dev_cat_oob)
end

# binomial trials via a named deterministic binding: the symbol is classified as an
# index slot, which the kernel must materialize on device (codex review of issue #12
# group 1; previously read uninitialized scratch).
@tea static function dev_named_trials_model(n)
    p ~ beta(2.0, 2.0)
    trials = n + 1
    {:k} ~ binomial(trials, p)
    return p
end

@testset "dev_named_trials_index_slot" begin
    supported, issues = device_lowering_report(dev_named_trials_model)
    @test supported
    @test isempty(issues)

    cm = choicemap((:k, 7.0))
    params = reshape([0.3, -0.8, 1.5], 1, 3)
    dev = device_batched_logjoint(dev_named_trials_model, params, (10,), cm)
    ref = batched_logjoint_unconstrained(dev_named_trials_model, params, (10,), cm)
    @test dev ≈ ref rtol = 1e-12
end

@testset "dev_workspace_reuse" begin
    ys = [0.4, -0.7, 1.1, 0.2]
    cm = choicemap((:y => i, ys[i]) for i = 1:4)
    ws = DeviceBatchedWorkspace(dev_gauss_model, 3; args=(4,), constraints=cm)

    params_a = [0.5 -0.3 1.2; 0.1 0.7 -0.2]
    params_b = params_a .+ 0.75
    result_a = device_batched_logjoint!(ws, params_a)
    result_b = device_batched_logjoint!(ws, params_b)
    # a third call with params_a again must reproduce result_a exactly (no stale state).
    result_a2 = device_batched_logjoint!(ws, params_a)

    ref_a = batched_logjoint_unconstrained(dev_gauss_model, params_a, (4,), cm)
    ref_b = batched_logjoint_unconstrained(dev_gauss_model, params_b, (4,), cm)
    @test result_a ≈ ref_a rtol = 1e-12
    @test result_b ≈ ref_b rtol = 1e-12
    @test result_a2 == result_a
end

@testset "dev_batched_constraints" begin
    ys = [0.4, -0.7, 1.1, 0.2]
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2]

    # identical-structure constraint vectors (per-column observation values).
    cm_vec = [choicemap((:y => i, ys[i] + 0.1 * b) for i = 1:4) for b = 1:3]
    dev = device_batched_logjoint(dev_gauss_model, params, (4,), cm_vec)
    ref = batched_logjoint_unconstrained(dev_gauss_model, params, (4,), cm_vec)
    @test dev ≈ ref rtol = 1e-12

    # mismatched structure (a column missing an address) must throw during staging.
    cm_bad = [
        choicemap((:y => i, ys[i]) for i = 1:4),
        choicemap((:y => i, ys[i]) for i = 1:2),
        choicemap((:y => i, ys[i]) for i = 1:4),
    ]
    @test_throws Exception device_batched_logjoint(dev_gauss_model, params, (4,), cm_bad)
end

@testset "dev_unsupported_fallback" begin
    # A device-unsupported model must raise a clear error pointing at the report.
    err = try
        device_batched_logjoint(dev_dirichlet_model, reshape([0.1, 0.2], 2, 1), ())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("device_lowering_report", err.msg)
end
