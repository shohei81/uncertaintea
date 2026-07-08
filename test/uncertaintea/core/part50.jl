# PR 50: backend-native batched scoring + analytic gradients for recently-added
# distribution families.
#
# `truncatednormal`, `mixture` (all-normal components), and `mvnormaldense` gain
# GPU-backend-native batched scoring and manual gradients, so `backend_report`
# now reports them supported and the batched gradient cache uses the
# backend-native tier (`backend_cache !== nothing`, no `flat_cache`, no
# `column_caches`). Correctness is validated against the compiled logjoint and the
# per-column ForwardDiff reference at atol 1e-8.
#
# Analytic gradient formulas implemented:
#   truncatednormal(mu, sigma, lower, upper): logpdf = normal_logpdf - log Z,
#     Z = Phi(zb) - Phi(za), za=(lower-mu)/sigma, zb=(upper-mu)/sigma. Derivatives
#     combine the base normal partials with the -log Z partials
#     d/dmu logZ = (phi(za)-phi(zb))/(sigma Z), d/dsigma logZ = (za*phi(za) -
#     zb*phi(zb))/(sigma Z), d/dlower logZ = -phi(za)/(sigma Z), d/dupper logZ =
#     phi(zb)/(sigma Z) (phi at +-Inf, and z*phi at +-Inf, taken as 0).
#   mixture(weights, normal components): responsibilities r_k = exp(log w_k +
#     lp_k - logpdf); d/dparam via r_k * d lp_k / dparam; d/dw_k = exp(lp_k -
#     logpdf).
#   mvnormaldense(mu, scale_tril=L): z = L^{-1}(x-mu), w = L^{-T} z; d/dmu = w,
#     d/dx = -w (the constant scale factor contributes no gradient).
#   truncatedstudentt(nu, mu, sigma, lower, upper) with a CONSTANT nu: logpdf =
#     studentt_logpdf - log Z, Z = T_cdf(zb) - T_cdf(za). The nu-derivative of the
#     incomplete-beta normalizer is intractable, so backend support is restricted
#     (at lowering) to a literal nu — its d/dnu term is then genuinely zero and
#     safely omitted. The remaining partials reuse the truncated-normal structure
#     with the Student-t pdf `p`/CDF: d/dmu logZ = (p(za)-p(zb))/(sigma Z),
#     d/dsigma logZ = (za p(za) - zb p(zb))/(sigma Z), d/dlower logZ =
#     -p(za)/(sigma Z), d/dupper logZ = p(zb)/(sigma Z).
#
# Deferred (still honestly reported unsupported, run via the fallback):
#   truncatedstudentt with a latent- or argument-flowing nu — the intractable
#     d/dnu term cannot be omitted, so lowering falls back to the compiled logjoint.
#   latent truncatednormal / truncatedstudentt — draw through a bounded parameter
#     transform the batched-backend unconstrained-transform layer does not implement.

@testset "bg_truncatednormal_backend_native" begin
    # Observation with a latent-flowing scale (exp(s)) and an unbounded upper side.
    @tea static function bg_tn_scale_model()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ truncatednormal(0.5f0, exp(s), 0.0, Inf)
        return s
    end

    bg_tn_scale_plan = backend_execution_plan(bg_tn_scale_model)
    @test backend_report(bg_tn_scale_model).supported == true
    @test bg_tn_scale_plan.steps[2] isa UncertainTea.BackendTruncatedNormalChoicePlanStep

    bg_tn_scale_params = reshape(Float64[-0.4, 0.1, 0.6], 1, 3)
    bg_tn_scale_constraints = [choicemap((:y, 0.3)), choicemap((:y, 1.2)), choicemap((:y, 2.0))]
    @test batched_logjoint(bg_tn_scale_model, bg_tn_scale_params, (), bg_tn_scale_constraints) ≈ [
        logjoint(bg_tn_scale_model, bg_tn_scale_params[:, i], (), bg_tn_scale_constraints[i]) for i in 1:3
    ] atol = 1e-8
    @test batched_logjoint_gradient_unconstrained(bg_tn_scale_model, bg_tn_scale_params, (), bg_tn_scale_constraints) ≈ hcat([
        logjoint_gradient_unconstrained(bg_tn_scale_model, bg_tn_scale_params[:, i], (), bg_tn_scale_constraints[i]) for i in 1:3
    ]...) atol = 1e-8

    bg_tn_scale_cache = BatchedLogjointGradientCache(bg_tn_scale_model, bg_tn_scale_params, (), bg_tn_scale_constraints)
    @test !isnothing(bg_tn_scale_cache.backend_cache)
    @test isnothing(bg_tn_scale_cache.flat_cache)
    @test isempty(bg_tn_scale_cache.column_caches)

    # Observation with a latent-flowing mean and finite bounds, exercising the
    # finite-normalizer d/dmu path.
    @tea static function bg_tn_mean_model()
        m ~ normal(0.0f0, 1.0f0)
        {:y} ~ truncatednormal(m, 1.0, -2.0, 2.0)
        return m
    end

    @test backend_report(bg_tn_mean_model).supported == true
    @test backend_execution_plan(bg_tn_mean_model).steps[2] isa UncertainTea.BackendTruncatedNormalChoicePlanStep

    bg_tn_mean_params = reshape(Float64[-0.4, 0.1, 0.6], 1, 3)
    bg_tn_mean_constraints = [choicemap((:y, 0.1)), choicemap((:y, 0.5)), choicemap((:y, -0.3))]
    @test batched_logjoint(bg_tn_mean_model, bg_tn_mean_params, (), bg_tn_mean_constraints) ≈ [
        logjoint(bg_tn_mean_model, bg_tn_mean_params[:, i], (), bg_tn_mean_constraints[i]) for i in 1:3
    ] atol = 1e-8
    @test batched_logjoint_gradient_unconstrained(bg_tn_mean_model, bg_tn_mean_params, (), bg_tn_mean_constraints) ≈ hcat([
        logjoint_gradient_unconstrained(bg_tn_mean_model, bg_tn_mean_params[:, i], (), bg_tn_mean_constraints[i]) for i in 1:3
    ]...) atol = 1e-8

    bg_tn_mean_cache = BatchedLogjointGradientCache(bg_tn_mean_model, bg_tn_mean_params, (), bg_tn_mean_constraints)
    @test !isnothing(bg_tn_mean_cache.backend_cache)
    @test isnothing(bg_tn_mean_cache.flat_cache)
    @test isempty(bg_tn_mean_cache.column_caches)
end

@testset "bg_mixture_backend_native" begin
    # Observation with a latent component mean.
    @tea static function bg_mix_mean_model()
        m ~ normal(0.0f0, 1.0f0)
        {:y} ~ mixture((0.3, 0.7), normal(m, 1.0f0), normal(2.0f0, 0.5f0))
        return m
    end

    bg_mix_mean_plan = backend_execution_plan(bg_mix_mean_model)
    @test backend_report(bg_mix_mean_model).supported == true
    @test bg_mix_mean_plan.steps[2] isa UncertainTea.BackendMixtureNormalChoicePlanStep

    bg_mix_mean_params = reshape(Float64[-0.4, 0.1, 0.6], 1, 3)
    bg_mix_mean_constraints = [choicemap((:y, 0.3)), choicemap((:y, 1.2)), choicemap((:y, 2.0))]
    @test batched_logjoint(bg_mix_mean_model, bg_mix_mean_params, (), bg_mix_mean_constraints) ≈ [
        logjoint(bg_mix_mean_model, bg_mix_mean_params[:, i], (), bg_mix_mean_constraints[i]) for i in 1:3
    ] atol = 1e-8
    @test batched_logjoint_gradient_unconstrained(bg_mix_mean_model, bg_mix_mean_params, (), bg_mix_mean_constraints) ≈ hcat([
        logjoint_gradient_unconstrained(bg_mix_mean_model, bg_mix_mean_params[:, i], (), bg_mix_mean_constraints[i]) for i in 1:3
    ]...) atol = 1e-8

    bg_mix_mean_cache = BatchedLogjointGradientCache(bg_mix_mean_model, bg_mix_mean_params, (), bg_mix_mean_constraints)
    @test !isnothing(bg_mix_mean_cache.backend_cache)
    @test isnothing(bg_mix_mean_cache.flat_cache)
    @test isempty(bg_mix_mean_cache.column_caches)

    # Latent mixture value (IdentityTransform) consumed by a downstream normal,
    # exercising the mixture d/dvalue (responsibility-weighted) partial.
    @tea static function bg_mix_latent_model()
        x ~ mixture((0.5f0, 0.5f0), normal(-2.0f0, 0.5f0), normal(2.0f0, 0.5f0))
        {:y} ~ normal(x, 0.5f0)
        return x
    end

    @test backend_report(bg_mix_latent_model).supported == true
    @test backend_execution_plan(bg_mix_latent_model).steps[1] isa UncertainTea.BackendMixtureNormalChoicePlanStep

    bg_mix_latent_params = reshape(Float64[-1.0, 0.2, 1.5], 1, 3)
    bg_mix_latent_constraints = [choicemap((:y, -1.5)), choicemap((:y, 0.4)), choicemap((:y, 2.1))]
    @test batched_logjoint_gradient_unconstrained(bg_mix_latent_model, bg_mix_latent_params, (), bg_mix_latent_constraints) ≈ hcat([
        logjoint_gradient_unconstrained(bg_mix_latent_model, bg_mix_latent_params[:, i], (), bg_mix_latent_constraints[i]) for i in 1:3
    ]...) atol = 1e-8

    bg_mix_latent_cache = BatchedLogjointGradientCache(bg_mix_latent_model, bg_mix_latent_params, (), bg_mix_latent_constraints)
    @test !isnothing(bg_mix_latent_cache.backend_cache)
    @test isnothing(bg_mix_latent_cache.flat_cache)
    @test isempty(bg_mix_latent_cache.column_caches)
end

@testset "bg_mvnormaldense_backend_native" begin
    # Observation with a latent scalar flowing into the mean vector; the scale
    # factor is a constant matrix passed as a model argument.
    @tea static function bg_mvd_mean_model(Larg)
        m ~ normal(0.0f0, 1.0f0)
        {:y} ~ mvnormaldense([m, m], Larg)
        return m
    end

    bg_mvd_L = [1.0 0.0; 0.5 2.0]
    bg_mvd_plan = backend_execution_plan(bg_mvd_mean_model)
    @test backend_report(bg_mvd_mean_model).supported == true
    @test bg_mvd_plan.steps[2] isa UncertainTea.BackendMvNormalDenseChoicePlanStep

    bg_mvd_params = reshape(Float64[-0.4, 0.1, 0.6], 1, 3)
    bg_mvd_constraints = [
        choicemap((:y, [0.3, 0.5])),
        choicemap((:y, [1.2, -0.4])),
        choicemap((:y, [2.0, 1.1])),
    ]
    @test batched_logjoint(bg_mvd_mean_model, bg_mvd_params, (bg_mvd_L,), bg_mvd_constraints) ≈ [
        logjoint(bg_mvd_mean_model, bg_mvd_params[:, i], (bg_mvd_L,), bg_mvd_constraints[i]) for i in 1:3
    ] atol = 1e-8
    @test batched_logjoint_gradient_unconstrained(bg_mvd_mean_model, bg_mvd_params, (bg_mvd_L,), bg_mvd_constraints) ≈ hcat([
        logjoint_gradient_unconstrained(bg_mvd_mean_model, bg_mvd_params[:, i], (bg_mvd_L,), bg_mvd_constraints[i]) for i in 1:3
    ]...) atol = 1e-8

    bg_mvd_cache = BatchedLogjointGradientCache(bg_mvd_mean_model, bg_mvd_params, (bg_mvd_L,), bg_mvd_constraints)
    @test !isnothing(bg_mvd_cache.backend_cache)
    @test isnothing(bg_mvd_cache.flat_cache)
    @test isempty(bg_mvd_cache.column_caches)
end

@testset "bg_truncatedstudentt_backend_native" begin
    # Observation with a latent-flowing scale (exp(s)), a constant nu, and an
    # unbounded upper side, exercising the infinite-bound (p(zb)=0) path.
    @tea static function bg_ts_scale_model()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ truncatedstudentt(5.0, 0.5f0, exp(s), 0.0, Inf)
        return s
    end

    bg_ts_scale_plan = backend_execution_plan(bg_ts_scale_model)
    @test backend_report(bg_ts_scale_model).supported == true
    @test bg_ts_scale_plan.steps[2] isa UncertainTea.BackendTruncatedStudentTChoicePlanStep

    bg_ts_scale_params = reshape(Float64[-0.4, 0.1, 0.6], 1, 3)
    bg_ts_scale_constraints = [choicemap((:y, 0.3)), choicemap((:y, 1.2)), choicemap((:y, 2.0))]
    @test batched_logjoint(bg_ts_scale_model, bg_ts_scale_params, (), bg_ts_scale_constraints) ≈ [
        logjoint(bg_ts_scale_model, bg_ts_scale_params[:, i], (), bg_ts_scale_constraints[i]) for i in 1:3
    ] atol = 1e-8
    @test batched_logjoint_gradient_unconstrained(bg_ts_scale_model, bg_ts_scale_params, (), bg_ts_scale_constraints) ≈ hcat([
        logjoint_gradient_unconstrained(bg_ts_scale_model, bg_ts_scale_params[:, i], (), bg_ts_scale_constraints[i]) for i in 1:3
    ]...) atol = 1e-8

    bg_ts_scale_cache = BatchedLogjointGradientCache(bg_ts_scale_model, bg_ts_scale_params, (), bg_ts_scale_constraints)
    @test !isnothing(bg_ts_scale_cache.backend_cache)
    @test isnothing(bg_ts_scale_cache.flat_cache)
    @test isempty(bg_ts_scale_cache.column_caches)

    # Observation with a latent-flowing mean and finite bounds, exercising the
    # finite-normalizer d/dmu and both d/dlower, d/dupper paths.
    @tea static function bg_ts_mean_model()
        m ~ normal(0.0f0, 1.0f0)
        {:y} ~ truncatedstudentt(4.0, m, 1.0, -2.0, 2.0)
        return m
    end

    @test backend_report(bg_ts_mean_model).supported == true
    @test backend_execution_plan(bg_ts_mean_model).steps[2] isa UncertainTea.BackendTruncatedStudentTChoicePlanStep

    bg_ts_mean_params = reshape(Float64[-0.4, 0.1, 0.6], 1, 3)
    bg_ts_mean_constraints = [choicemap((:y, 0.1)), choicemap((:y, 0.5)), choicemap((:y, -0.3))]
    @test batched_logjoint(bg_ts_mean_model, bg_ts_mean_params, (), bg_ts_mean_constraints) ≈ [
        logjoint(bg_ts_mean_model, bg_ts_mean_params[:, i], (), bg_ts_mean_constraints[i]) for i in 1:3
    ] atol = 1e-8
    @test batched_logjoint_gradient_unconstrained(bg_ts_mean_model, bg_ts_mean_params, (), bg_ts_mean_constraints) ≈ hcat([
        logjoint_gradient_unconstrained(bg_ts_mean_model, bg_ts_mean_params[:, i], (), bg_ts_mean_constraints[i]) for i in 1:3
    ]...) atol = 1e-8

    bg_ts_mean_cache = BatchedLogjointGradientCache(bg_ts_mean_model, bg_ts_mean_params, (), bg_ts_mean_constraints)
    @test !isnothing(bg_ts_mean_cache.backend_cache)
    @test isnothing(bg_ts_mean_cache.flat_cache)
    @test isempty(bg_ts_mean_cache.column_caches)
end

@testset "bg_deferred_families_fall_back" begin
    # A latent-flowing nu keeps truncatedstudentt deferred: the intractable d/dnu
    # term cannot be omitted, so lowering falls back to the compiled logjoint. The
    # logjoint VALUE still matches per column via the fallback.
    @tea static function bg_ts_latent_nu_model()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ truncatedstudentt(exp(s) + 3.0, 0.5f0, 1.0, 0.0, Inf)
        return s
    end

    @test backend_report(bg_ts_latent_nu_model).supported == false
    bg_ts_params = reshape(Float64[-0.4, 0.1, 0.6], 1, 3)
    bg_ts_constraints = [choicemap((:y, 0.3)), choicemap((:y, 1.2)), choicemap((:y, 2.0))]
    @test batched_logjoint(bg_ts_latent_nu_model, bg_ts_params, (), bg_ts_constraints) ≈ [
        logjoint(bg_ts_latent_nu_model, bg_ts_params[:, i], (), bg_ts_constraints[i]) for i in 1:3
    ] atol = 1e-8

    # A latent nu with BOTH bounds infinite still has a valid CPU gradient: the
    # normalizer Z = T_cdf(+Inf) - T_cdf(-Inf) = 1 is constant, so no intractable
    # d/dnu term arises and the infinite-bound guard must fire before the
    # constant-nu check. The gradient matches the equivalent untruncated studentt.
    @tea static function bg_ts_latent_nu_open_model()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ truncatedstudentt(exp(s) + 3.0, 0.5f0, 1.0, -Inf, Inf)
        return s
    end
    @tea static function bg_ts_latent_nu_ref_model()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ studentt(exp(s) + 3.0, 0.5f0, 1.0)
        return s
    end
    for i in 1:3
        @test logjoint_gradient_unconstrained(bg_ts_latent_nu_open_model, bg_ts_params[:, i], (), bg_ts_constraints[i]) ≈
              logjoint_gradient_unconstrained(bg_ts_latent_nu_ref_model, bg_ts_params[:, i], (), bg_ts_constraints[i]) atol = 1e-8
        @test all(isfinite, logjoint_gradient_unconstrained(bg_ts_latent_nu_open_model, bg_ts_params[:, i], (), bg_ts_constraints[i]))
    end

    # A latent nu with a finite bound remains genuinely unsupported: the CDF's
    # d/dnu term is intractable, so the CPU gradient throws honestly.
    @test_throws ArgumentError logjoint_gradient_unconstrained(
        bg_ts_latent_nu_model, bg_ts_params[:, 1], (), bg_ts_constraints[1],
    )

    # Latent truncatednormal is deferred: the bounded parameter transform is not
    # implemented in the batched-backend transform layer, so it runs via the
    # ForwardDiff column-cache fallback. The value still matches per column.
    @tea static function bg_tn_latent_model()
        x ~ truncatednormal(0.0f0, 1.0, -2.0, 2.0)
        {:y} ~ normal(x, 0.5f0)
        return x
    end

    @test backend_report(bg_tn_latent_model).supported == false
    bg_tn_latent_trace, _ = generate(bg_tn_latent_model, (), choicemap((:y, 0.3)); rng = MersenneTwister(150))
    bg_tn_latent_pv = transform_to_unconstrained(bg_tn_latent_trace)
    bg_tn_latent_params = hcat(bg_tn_latent_pv, bg_tn_latent_pv .+ 0.3, bg_tn_latent_pv .- 0.4)
    bg_tn_latent_constraints = [choicemap((:y, 0.1)), choicemap((:y, 0.5)), choicemap((:y, -0.3))]
    @test batched_logjoint_unconstrained(bg_tn_latent_model, bg_tn_latent_params, (), bg_tn_latent_constraints) ≈ [
        logjoint_unconstrained(bg_tn_latent_model, bg_tn_latent_params[:, i], (), bg_tn_latent_constraints[i]) for i in 1:3
    ] atol = 1e-8

    bg_tn_latent_cache = BatchedLogjointGradientCache(bg_tn_latent_model, bg_tn_latent_params, (), bg_tn_latent_constraints)
    @test isnothing(bg_tn_latent_cache.backend_cache)
    @test !isempty(bg_tn_latent_cache.column_caches)
end
