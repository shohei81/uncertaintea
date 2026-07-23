# Issue #148: `_logfactorial_like` computes log-factorials of data counts via
# Float64 `loggamma` instead of summing `log(k)` in the caller's (possibly
# dual) arithmetic. These tests pin equality with the old O(n) summation, the
# exact-cancellation behavior of the binomial coefficient at k=0 / k=n, and
# that gradients through latent rate/p are unchanged (the count term is a data
# constant with zero derivative).
@testset "dist_count_logfactorial" begin
    lf_FD = UncertainTea.ForwardDiff

    # the old O(count) summation, kept here as the reference
    lf_reference = function (value, n::Integer)
        total = zero(value)
        unit = one(value)
        for k = 2:n
            total += log(unit * k)
        end
        return total
    end

    lf_counts = (0, 1, 2, 10, 250, 10_000)

    for n in lf_counts
        lf_new = UncertainTea._logfactorial_like(0.7, n)
        lf_old = lf_reference(0.7, n)
        if n < 2
            @test lf_new == 0.0
            @test lf_old == 0.0
        else
            @test lf_new ≈ lf_old rtol = 1e-12
        end
    end

    # element type follows the caller's arithmetic (Float32 stays Float32,
    # duals stay duals with zero count-derivative)
    @test UncertainTea._logfactorial_like(0.5f0, 7) isa Float32
    @test UncertainTea.logpdf(poisson(2.0f0), 5) isa Float32
    lf_dual = UncertainTea._logfactorial_like(lf_FD.Dual(0.5, 1.0), 12)
    @test lf_FD.value(lf_dual) ≈ lf_reference(0.5, 12) rtol = 1e-12
    @test lf_FD.partials(lf_dual, 1) == 0.0

    # public logpdfs and the backend scalar scorers agree with the values
    # rebuilt from the reference summation
    lf_lambda = 3.25
    lf_trials = 10_000
    lf_p = 0.37
    lf_successes = 2.5
    for k in lf_counts
        lf_poisson_expected = k * log(lf_lambda) - lf_lambda - lf_reference(lf_lambda, k)
        @test UncertainTea.logpdf(poisson(lf_lambda), k) ≈ lf_poisson_expected rtol = 1e-12
        @test UncertainTea._backend_poisson_logpdf(lf_lambda, k) ≈ lf_poisson_expected rtol = 1e-12

        lf_binomial_expected =
            lf_reference(lf_p, lf_trials) - lf_reference(lf_p, k) - lf_reference(lf_p, lf_trials - k) +
            (k == 0 ? zero(lf_p) : k * log(lf_p)) +
            (k == lf_trials ? zero(lf_p) : (lf_trials - k) * log1p(-lf_p))
        @test UncertainTea.logpdf(UncertainTea.binomial(lf_trials, lf_p), k) ≈ lf_binomial_expected rtol = 1e-12
        @test UncertainTea._backend_binomial_logpdf(lf_trials, lf_p, k) ≈ lf_binomial_expected rtol = 1e-12

        lf_negbin_expected =
            UncertainTea.loggamma(k + lf_successes) - UncertainTea.loggamma(lf_successes) -
            lf_reference(lf_p, k) + lf_successes * log(lf_p) + k * log1p(-lf_p)
        @test UncertainTea.logpdf(negativebinomial(lf_successes, lf_p), k) ≈ lf_negbin_expected rtol = 1e-12
        @test UncertainTea._backend_negativebinomial_logpdf(lf_successes, lf_p, k) ≈ lf_negbin_expected rtol = 1e-12
    end

    # binomial coefficient at extreme counts: log n! - log k! - log (n-k)! must
    # cancel exactly at k=0 and k=n even for huge n, and the full logpdf stays
    # the pure tail term
    lf_huge = 10^6
    @test UncertainTea._logbinomial_like(0.3, lf_huge, 0) == 0.0
    @test UncertainTea._logbinomial_like(0.3, lf_huge, lf_huge) == 0.0
    @test UncertainTea.logpdf(UncertainTea.binomial(lf_huge, 0.3), 0) ≈ lf_huge * log1p(-0.3) rtol = 1e-12
    @test UncertainTea.logpdf(UncertainTea.binomial(lf_huge, 0.3), lf_huge) ≈ lf_huge * log(0.3) rtol = 1e-12
    @test isfinite(UncertainTea._logbinomial_like(0.3, lf_huge, lf_huge ÷ 2))

    # gradients through a latent rate: ForwardDiff through the full CPU logjoint
    # must match ForwardDiff through the old-summation reference density
    @tea static function lf_poisson_model()
        rate ~ exponential(1.0f0)
        {:y} ~ poisson(rate)
        return rate
    end

    lf_poisson_constraints = choicemap((:y, 250))
    lf_poisson_reference =
        rate -> -rate + 250 * log(rate) - rate - lf_reference(rate, 250)
    for rate in (0.5, 2.5, 300.0)
        lf_grad_new = lf_FD.derivative(r -> logjoint(lf_poisson_model, [r], (), lf_poisson_constraints), rate)
        lf_grad_old = lf_FD.derivative(lf_poisson_reference, rate)
        @test lf_grad_new ≈ lf_grad_old rtol = 1e-12
        @test lf_grad_new ≈ 250 / rate - 2 rtol = 1e-12
    end

    # gradients through a latent binomial success probability
    @tea static function lf_binomial_model()
        p ~ beta(2.0f0, 2.0f0)
        {:y} ~ binomial(1000, p)
        return p
    end

    lf_binomial_constraints = choicemap((:y, 500))
    lf_binomial_reference =
        p ->
            log(6.0) + log(p) + log1p(-p) + lf_reference(p, 1000) - lf_reference(p, 500) -
            lf_reference(p, 500) + 500 * log(p) + 500 * log1p(-p)
    for p in (0.05, 0.4, 0.95)
        lf_grad_new = lf_FD.derivative(q -> logjoint(lf_binomial_model, [q], (), lf_binomial_constraints), p)
        lf_grad_old = lf_FD.derivative(lf_binomial_reference, p)
        @test lf_grad_new ≈ lf_grad_old rtol = 1e-12
    end
end
