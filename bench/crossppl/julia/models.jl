# Canonical benchmark models, shared shapes with the Stan/NumPyro definitions.
# Priors follow the literature-standard forms so every framework states the
# identical joint density (see bench/crossppl/README.md).

# Eight schools (Rubin 1981): mu ~ N(0,5), tau ~ HalfCauchy(5),
# theta_j ~ N(mu, tau), y_j ~ N(theta_j, sigma_j).  J = 8 is literal because
# `iid` requires a literal count.
@tea static function bench_eight_schools_centered(sigma)
    mu ~ normal(0.0, 5.0)
    tau ~ truncatedstudentt(1.0, 0.0, 5.0, 0.0, Inf)
    theta ~ iid(normal(mu, tau), 8)
    for i = 1:8
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

@tea static function bench_eight_schools_noncentered(sigma)
    mu ~ normal(0.0, 5.0)
    tau ~ truncatedstudentt(1.0, 0.0, 5.0, 0.0, Inf)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for i = 1:8
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

# Logistic regression: alpha ~ N(0,2.5), beta_d ~ N(0,2.5),
# y_i ~ Bernoulli(logistic(alpha + x_i'beta)).  D = 8 is literal; the
# observation loop is loop-addressed because only `normal.` broadcasts.
@tea static function bench_logistic(X, n)
    alpha ~ normal(0.0, 2.5)
    beta ~ iid(normal(0.0, 2.5), 8)
    for i = 1:n
        {:y => i} ~ bernoulli(1.0 / (1.0 + exp(-(alpha + sum(beta .* X[:, i])))))
    end
    return alpha
end

# Gaussian mean/scale estimation with a loop-addressed observation vector —
# the device-supported form (same shape as test/gpu gpu_gauss_model) used for
# the chain-count scaling sweep.  Broadcast-normal observations and indexed
# covariates do not lower to the device path yet, so regression-style models
# cannot ride it (see issues filed from this benchmark work).
@tea static function bench_gauss(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end
