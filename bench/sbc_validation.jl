# Release-grade simulation-based calibration run (issue #18). Not part of the
# CI suite -- run manually when validating a new sampler, adaptation change,
# or distribution family:
#
#   julia --project=. bench/sbc_validation.jl
#
# Runtime is dominated by num_simulations * (num_warmup + num_draws * thin)
# NUTS iterations per model; the defaults below take a few minutes.

using UncertainTea
using Random

@tea static function sbc_bench_conjugate()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

@tea static function sbc_bench_hierarchical()
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 0.5)
    for i = 1:4
        {:y => i} ~ normal(mu, sigma)
    end
    return mu
end

@tea static function sbc_bench_positive()
    rate ~ gamma(2.0, 2.0)
    for i = 1:3
        {:y => i} ~ exponential(rate)
    end
    return rate
end

const SBC_BENCH_MODELS = [
    ("conjugate normal-normal", sbc_bench_conjugate),
    ("hierarchical normal", sbc_bench_hierarchical),
    ("gamma-exponential", sbc_bench_positive),
]

rng = MersenneTwister(20260709)
for (name, model) in SBC_BENCH_MODELS
    result = sbc(
        model;
        num_simulations=300,
        num_posterior_draws=63,
        num_warmup=200,
        thin=2,
        rng=rng,
    )
    println("== ", name)
    show(stdout, MIME"text/plain"(), result)
    println()
end
