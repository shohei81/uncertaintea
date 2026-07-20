# Issue #105: the logit-family log-Jacobian log(c) + log1p(-c) with
# c = sigmoid(u) collapses to -Inf once c rounds to exactly 1 (u ~ 36.7 in
# Float64, ~16.6 in Float32) while the exact value -|u| - 2*log1p(exp(-|u|))
# stays finite; the negative side was exact, so the density was asymmetric
# under u -> -u. Covers LogitTransform, VectorLogitTransform, BoundedTransform,
# the batched pre-pass, and the device `_device_transform` LOGIT mirror
# (value and DeviceDual gradient channels).

tls_exact(u) = -abs(u) - 2 * log1p(exp(-abs(u)))

@testset "transform_logit_saturation" begin
    # Float64 CPU: exact and symmetric across the old cliff at u ~ 36.7.
    for tls_u = 35.0:0.5:40.0
        tls_pos = UncertainTea.logabsdetjac(LogitTransform(), tls_u)
        tls_neg = UncertainTea.logabsdetjac(LogitTransform(), -tls_u)
        @test isfinite(tls_pos)
        @test tls_pos == tls_exact(tls_u)
        @test tls_pos == tls_neg
    end
    @test UncertainTea.logabsdetjac(LogitTransform(), 37.0) == -37.0

    # Vector and bounded variants share the stable form.
    tls_v = [36.5, -37.0, 40.0]
    @test UncertainTea.logabsdetjac(VectorLogitTransform(3), tls_v) == sum(tls_exact, tls_v)
    tls_b = BoundedTransform(-2.0, 3.0)
    @test UncertainTea.logabsdetjac(tls_b, 38.0) == log(5.0) + tls_exact(38.0)
    @test UncertainTea.logabsdetjac(tls_b, 38.0) == UncertainTea.logabsdetjac(tls_b, -38.0)

    # Moderate |u|: matches the naive log(c) + log1p(-c) form to rounding.
    for tls_um = -5.0:0.25:5.0
        tls_c = inv(1 + exp(-tls_um))
        tls_naive = log(tls_c) + log1p(-tls_c)
        @test UncertainTea.logabsdetjac(LogitTransform(), tls_um) ≈ tls_naive atol = 1e-13
    end

    # The batched transform pre-pass reuses the same Jacobian. u = 36.5 sits in
    # the old error region (the naive form was off by ~0.46 nats there) while
    # beta's own density is still finite; parity with the single path holds and
    # the Jacobian contribution is the exact stable value.
    @tea static function tls_logit_model()
        p ~ beta(2.0, 3.0)
        {:y} ~ bernoulli(p)
    end
    tls_batch = reshape([36.5, -36.5, 1.25], 1, 3)
    tls_values =
        batched_logjoint_unconstrained(tls_logit_model, tls_batch, (), choicemap(:y => true))
    @test all(isfinite, tls_values)
    tls_single = [
        logjoint_unconstrained(tls_logit_model, [tls_batch[1, k]], (), choicemap(:y => true))
        for k = 1:3
    ]
    @test maximum(abs.(tls_values .- tls_single)) < 1e-12
    tls_c36 = inv(1 + exp(-36.5))
    tls_expected1 =
        UncertainTea.logpdf(beta(2.0, 3.0), tls_c36) +
        tls_exact(36.5) +
        UncertainTea.logpdf(bernoulli(tls_c36), true)
    @test tls_values[1] ≈ tls_expected1 atol = 1e-10

    # Float32 device mirror: finite past the old u ~ 16.6 cliff, exact, and the
    # DeviceDual channel carries the finite gradient -tanh(u/2).
    for tls_uf in Float32[16.0, 17.0, 20.0, -17.0, -20.0]
        tls_c32, tls_lad32 = UncertainTea._device_transform(Int32(2), tls_uf)
        @test isfinite(tls_lad32)
        @test tls_lad32 ≈ Float32(tls_exact(Float64(tls_uf))) atol = 2 * eps(abs(tls_uf))
        tls_dual = UncertainTea.DeviceDual(tls_uf, one(Float32))
        tls_cd, tls_ld = UncertainTea._device_transform(Int32(2), tls_dual)
        @test isfinite(tls_ld.value)
        @test isfinite(tls_ld.deriv)
        @test tls_ld.deriv ≈ -tanh(tls_uf / 2) atol = 1.0f-5
        @test tls_cd.value ≈ tls_c32 atol = 1.0f-7
    end
    @test UncertainTea._device_transform(Int32(2), 17.0f0)[2] == -17.0f0
end
