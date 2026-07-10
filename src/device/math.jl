# Device-safe, T-generic scalar log-densities used by the fused KernelAbstractions
# logjoint kernel. Every function here must be callable inside a GPU kernel:
#   * no exceptions / no error paths (out-of-support returns -Inf through ifelse),
#   * no allocations,
#   * no dependency on SpecialFunctions (a pure-Julia Lanczos loggamma is provided).
# Non-finite parameters (e.g. sigma <= 0) propagate as NaN naturally, matching the
# "garbage in -> NaN out" contract; the authoritative CPU backend still validates.

# Lanczos approximation coefficients (g = 7, n = 9). Stored as Float64 and cast to
# the working precision T on demand.
const _DEVICE_LANCZOS_G = 7.0
const _DEVICE_LANCZOS_COEFS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)

"""
    _device_loggamma(x::T) -> T

Pure-Julia log-gamma via the Lanczos approximation, valid on device. Uses the
reflection formula for `x < 0.5`. Coefficients are Float64 constants cast to `T`.
"""
@inline function _device_loggamma(x::T) where {T}
    if x < T(0.5)
        # reflection: loggamma(x) = log(pi / sin(pi x)) - loggamma(1 - x)
        return T(log(pi)) - log(abs(sin(T(pi) * x))) - _device_loggamma(one(T) - x)
    end
    xm1 = x - one(T)
    a = T(_DEVICE_LANCZOS_COEFS[1])
    t = xm1 + T(_DEVICE_LANCZOS_G) + T(0.5)
    a += T(_DEVICE_LANCZOS_COEFS[2]) / (xm1 + one(T))
    a += T(_DEVICE_LANCZOS_COEFS[3]) / (xm1 + T(2))
    a += T(_DEVICE_LANCZOS_COEFS[4]) / (xm1 + T(3))
    a += T(_DEVICE_LANCZOS_COEFS[5]) / (xm1 + T(4))
    a += T(_DEVICE_LANCZOS_COEFS[6]) / (xm1 + T(5))
    a += T(_DEVICE_LANCZOS_COEFS[7]) / (xm1 + T(6))
    a += T(_DEVICE_LANCZOS_COEFS[8]) / (xm1 + T(7))
    a += T(_DEVICE_LANCZOS_COEFS[9]) / (xm1 + T(8))
    half_log_2pi = T(0.9189385332046727) # 0.5*log(2*pi)
    return half_log_2pi + (xm1 + T(0.5)) * log(t) - t + log(a)
end

@inline _device_neginf(::Type{T}) where {T} = T(-Inf)

@inline function _device_normal_logpdf(mu::T, sigma::T, x::T) where {T}
    z = (x - mu) / sigma
    return -log(sigma) - T(0.9189385332046727) - z * z / T(2)
end

@inline function _device_lognormal_logpdf(mu::T, sigma::T, x::T) where {T}
    lx = log(x)
    base = -log(sigma) - T(0.9189385332046727) - ((lx - mu) / sigma)^2 / T(2) - lx
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_exponential_logpdf(rate::T, x::T) where {T}
    return ifelse(x >= zero(T), log(rate) - rate * x, _device_neginf(T))
end

@inline function _device_gamma_logpdf(shape::T, rate::T, x::T) where {T}
    base = shape * log(rate) - _device_loggamma(shape) +
           (shape - one(T)) * log(x) - rate * x
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_laplace_logpdf(loc::T, scale::T, x::T) where {T}
    return -log(T(2) * scale) - abs(x - loc) / scale
end

@inline function _device_beta_logpdf(alpha::T, beta::T, x::T) where {T}
    logbeta = _device_loggamma(alpha) + _device_loggamma(beta) - _device_loggamma(alpha + beta)
    base = (alpha - one(T)) * log(x) + (beta - one(T)) * log1p(-x) - logbeta
    return ifelse((x > zero(T)) & (x < one(T)), base, _device_neginf(T))
end

@inline function _device_studentt_logpdf(nu::T, mu::T, sigma::T, x::T) where {T}
    z = (x - mu) / sigma
    return _device_loggamma((nu + one(T)) / T(2)) - _device_loggamma(nu / T(2)) -
           (log(nu) + T(1.1447298858494002)) / T(2) - log(sigma) - # log(pi)
           (nu + one(T)) * log1p(z * z / nu) / T(2)
end

@inline function _device_inversegamma_logpdf(shape::T, scale::T, x::T) where {T}
    base = shape * log(scale) - _device_loggamma(shape) -
           (shape + one(T)) * log(x) - scale / x
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_weibull_logpdf(shape::T, scale::T, x::T) where {T}
    base = log(shape) + (shape - one(T)) * log(x) -
           shape * log(scale) - exp(shape * (log(x) - log(scale)))
    at_zero = ifelse(shape < one(T), T(Inf), ifelse(shape == one(T), -log(scale), _device_neginf(T)))
    return ifelse(x > zero(T), base, ifelse(x == zero(T), at_zero, _device_neginf(T)))
end

@inline function _device_bernoulli_logpdf(p::T, x::T) where {T}
    return ifelse(x > T(0.5), log(p), log1p(-p))
end

# Nonnegative near-integer support check shared by the count families; mirrors the
# CPU `_poisson_count` acceptance up to the device float tolerance.
@inline _device_count_ok(x::T, k::T) where {T} =
    (x >= zero(T)) & (abs(x - k) <= T(1e-6) * (one(T) + abs(x)))

@inline function _device_poisson_logpdf(lambda::T, x::T) where {T}
    k = round(x)
    base = k * log(lambda) - lambda - _device_loggamma(k + one(T))
    return ifelse(_device_count_ok(x, k), base, _device_neginf(T))
end

@inline function _device_binomial_logpdf(trials::T, p::T, x::T) where {T}
    n = round(trials)
    k = round(x)
    in_support = _device_count_ok(x, k) & _device_count_ok(trials, n) & (k <= n)
    log_combination = _device_loggamma(n + one(T)) - _device_loggamma(k + one(T)) -
                      _device_loggamma(n - k + one(T))
    # guard the k == 0 / k == n corners so p in {0, 1} cannot produce 0 * -Inf
    base = log_combination + ifelse(k > zero(T), k * log(p), zero(T)) +
           ifelse(k < n, (n - k) * log1p(-p), zero(T))
    return ifelse(in_support, base, _device_neginf(T))
end

@inline function _device_geometric_logpdf(p::T, x::T) where {T}
    k = round(x)
    base = log(p) + ifelse(k > zero(T), k * log1p(-p), zero(T))
    return ifelse(_device_count_ok(x, k), base, _device_neginf(T))
end

@inline function _device_negativebinomial_logpdf(successes::T, p::T, x::T) where {T}
    k = round(x)
    base =
        _device_loggamma(k + successes) - _device_loggamma(successes) -
        _device_loggamma(k + one(T)) + successes * log(p) +
        ifelse(k > zero(T), k * log1p(-p), zero(T))
    return ifelse(_device_count_ok(x, k), base, _device_neginf(T))
end

# Compile-time unrolled selection of log(probabilities[k]); unmatched entries
# contribute an exact zero so heterogeneous (literal/dual) tuples promote safely.
@inline _device_categorical_pick(::Tuple{}, k, index::Int32, acc) = acc
@inline function _device_categorical_pick(probabilities::Tuple, k, index::Int32, acc)
    logp = log(first(probabilities))
    term = ifelse(k == index, logp, zero(logp))
    return _device_categorical_pick(Base.tail(probabilities), k, index + Int32(1), acc + term)
end

@inline function _device_categorical_logpdf(probabilities::Tuple, x::T) where {T}
    k = round(x)
    in_support = (x >= one(T)) & (x <= T(length(probabilities))) &
                 (abs(x - k) <= T(1e-6) * (one(T) + abs(x)))
    total = _device_categorical_pick(probabilities, k, Int32(1), zero(T))
    return ifelse(in_support, total, oftype(total, -Inf))
end

# Parameter transform codes shared by lowering, kernel, and staging.
const DEVICE_TRANSFORM_IDENTITY = Int32(0)
const DEVICE_TRANSFORM_LOG = Int32(1)
const DEVICE_TRANSFORM_LOGIT = Int32(2)

# Applies an unconstrained -> constrained transform, returning
# (constrained_value, logabsdet_jacobian). Mirrors src/parameters.jl and the
# batched transform branch in src/batched/workspace.jl for the supported set.
@inline function _device_transform(tcode::Int32, u::T) where {T}
    if tcode == DEVICE_TRANSFORM_LOG
        return (exp(u), u)
    elseif tcode == DEVICE_TRANSFORM_LOGIT
        c = inv(one(T) + exp(-u))
        return (c, log(c) + log1p(-c))
    else # DEVICE_TRANSFORM_IDENTITY
        return (u, zero(T))
    end
end
