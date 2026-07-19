# CPU-reference distributions (structs, builders, logpdf, rand): continuous scalar families (normal, lognormal, laplace, exponential, gamma, inversegamma, weibull, beta, studentt).

struct NormalDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    sigma::T

    function NormalDist(mu::T, sigma::T) where {T<:Real}
        sigma > zero(T) || throw(ArgumentError("normal requires sigma > 0"))
        new{T}(mu, sigma)
    end
end

struct LaplaceDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    scale::T

    function LaplaceDist(mu::T, scale::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("laplace requires scale > 0"))
        new{T}(mu, scale)
    end
end

struct ExponentialDist{T<:Real} <: AbstractTeaDistribution
    rate::T

    function ExponentialDist(rate::T) where {T<:Real}
        rate > zero(T) || throw(ArgumentError("exponential requires rate > 0"))
        new{T}(rate)
    end
end

struct GammaDist{T<:Real} <: AbstractTeaDistribution
    shape::T
    rate::T

    function GammaDist(shape::T, rate::T) where {T<:Real}
        shape > zero(T) || throw(ArgumentError("gamma requires shape > 0"))
        rate > zero(T) || throw(ArgumentError("gamma requires rate > 0"))
        new{T}(shape, rate)
    end
end

struct InverseGammaDist{T<:Real} <: AbstractTeaDistribution
    shape::T
    scale::T

    function InverseGammaDist(shape::T, scale::T) where {T<:Real}
        shape > zero(T) || throw(ArgumentError("inversegamma requires shape > 0"))
        scale > zero(T) || throw(ArgumentError("inversegamma requires scale > 0"))
        new{T}(shape, scale)
    end
end

struct WeibullDist{T<:Real} <: AbstractTeaDistribution
    shape::T
    scale::T

    function WeibullDist(shape::T, scale::T) where {T<:Real}
        shape > zero(T) || throw(ArgumentError("weibull requires shape > 0"))
        scale > zero(T) || throw(ArgumentError("weibull requires scale > 0"))
        new{T}(shape, scale)
    end
end

struct BetaDist{T<:Real} <: AbstractTeaDistribution
    alpha::T
    beta::T

    function BetaDist(alpha::T, beta::T) where {T<:Real}
        alpha > zero(T) || throw(ArgumentError("beta requires alpha > 0"))
        beta > zero(T) || throw(ArgumentError("beta requires beta > 0"))
        new{T}(alpha, beta)
    end
end

struct LogNormalDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    sigma::T

    function LogNormalDist(mu::T, sigma::T) where {T<:Real}
        sigma > zero(T) || throw(ArgumentError("lognormal requires sigma > 0"))
        new{T}(mu, sigma)
    end
end

struct StudentTDist{T<:Real} <: AbstractTeaDistribution
    nu::T
    mu::T
    sigma::T

    function StudentTDist(nu::T, mu::T, sigma::T) where {T<:Real}
        nu > zero(T) || throw(ArgumentError("studentt requires nu > 0"))
        sigma > zero(T) || throw(ArgumentError("studentt requires sigma > 0"))
        new{T}(nu, mu, sigma)
    end
end

# Builders normalize parameters through `float` so integer (or other non-float
# real) literals reach the samplers as float storage (issue #73); `float` keeps
# ForwardDiff Duals intact.
function normal(mu, sigma)
    promoted_mu, promoted_sigma = promote(float(mu), float(sigma))
    return NormalDist(promoted_mu, promoted_sigma)
end

function laplace(mu, scale)
    promoted_mu, promoted_scale = promote(float(mu), float(scale))
    return LaplaceDist(promoted_mu, promoted_scale)
end

function exponential(rate)
    return ExponentialDist(float(rate))
end

function gamma(shape, rate)
    promoted_shape, promoted_rate = promote(float(shape), float(rate))
    return GammaDist(promoted_shape, promoted_rate)
end

function inversegamma(shape, scale)
    promoted_shape, promoted_scale = promote(float(shape), float(scale))
    return InverseGammaDist(promoted_shape, promoted_scale)
end

function weibull(shape, scale)
    promoted_shape, promoted_scale = promote(float(shape), float(scale))
    return WeibullDist(promoted_shape, promoted_scale)
end

function beta(alpha, beta_parameter)
    promoted_alpha, promoted_beta = promote(float(alpha), float(beta_parameter))
    return BetaDist(promoted_alpha, promoted_beta)
end

function lognormal(mu, sigma)
    promoted_mu, promoted_sigma = promote(float(mu), float(sigma))
    return LogNormalDist(promoted_mu, promoted_sigma)
end

function studentt(nu, mu, sigma)
    promoted_nu, promoted_mu, promoted_sigma = promote(float(nu), float(mu), float(sigma))
    return StudentTDist(promoted_nu, promoted_mu, promoted_sigma)
end

function Random.rand(rng::AbstractRNG, dist::NormalDist{T}) where {T<:AbstractFloat}
    return dist.mu + dist.sigma * randn(rng, T)
end

function Random.rand(rng::AbstractRNG, dist::LaplaceDist)
    scale = float(dist.scale)
    threshold = rand(rng, typeof(scale)) - oftype(scale, 0.5)
    noise = threshold < 0 ? log1p(2 * threshold) : -log1p(-2 * threshold)
    return float(dist.mu) + scale * noise
end

function Random.rand(rng::AbstractRNG, dist::ExponentialDist)
    rate = float(dist.rate)
    return randexp(rng, typeof(rate)) / rate
end

function _rand_gamma_marsaglia(rng::AbstractRNG, shape::T, rate::T) where {T<:AbstractFloat}
    if shape < one(T)
        return _rand_gamma_marsaglia(rng, shape + one(T), rate) * (rand(rng, T) ^ (inv(shape)))
    end

    d = shape - T(1 / 3)
    c = inv(sqrt(T(9) * d))
    while true
        x = randn(rng, T)
        v = one(T) + c * x
        v > zero(T) || continue
        v3 = v * v * v
        u = rand(rng, T)
        if u < one(T) - T(0.0331) * (x * x) * (x * x) ||
           log(u) < T(0.5) * x * x + d * (one(T) - v3 + log(v3))
            return d * v3 / rate
        end
    end
end

function Random.rand(rng::AbstractRNG, dist::GammaDist)
    shape = float(dist.shape)
    rate = float(dist.rate)
    return _rand_gamma_marsaglia(rng, shape, rate)
end

function Random.rand(rng::AbstractRNG, dist::InverseGammaDist)
    shape = float(dist.shape)
    scale = float(dist.scale)
    return inv(_rand_gamma_marsaglia(rng, shape, scale))
end

function Random.rand(rng::AbstractRNG, dist::WeibullDist)
    shape = float(dist.shape)
    scale = float(dist.scale)
    return scale * (randexp(rng, typeof(scale)) ^ inv(shape))
end

function Random.rand(rng::AbstractRNG, dist::BetaDist)
    alpha = float(dist.alpha)
    beta_parameter = float(dist.beta)
    x = _rand_gamma_marsaglia(rng, alpha, one(alpha))
    y = _rand_gamma_marsaglia(rng, beta_parameter, one(beta_parameter))
    return x / (x + y)
end

function Random.rand(rng::AbstractRNG, dist::LogNormalDist{T}) where {T<:AbstractFloat}
    return exp(dist.mu + dist.sigma * randn(rng, T))
end

function Random.rand(rng::AbstractRNG, dist::StudentTDist)
    nu = float(dist.nu)
    mu = float(dist.mu)
    sigma = float(dist.sigma)
    scale = randn(rng, typeof(mu)) / sqrt(_rand_gamma_marsaglia(rng, nu / 2, nu / 2))
    return mu + sigma * scale
end

function _std_t_cdf(z, nu)
    zz = float(z)
    # Guard infinite arguments before arithmetic so ForwardDiff Duals carrying an
    # Inf value (with NaN partials) collapse to a finite constant.
    isinf(zz) && return zz > zero(zz) ? one(zz) : zero(zz)
    zz == zero(zz) && return oftype(zz, 0.5)
    x = nu / (nu + zz * zz)
    regularized = beta_inc(nu / 2, one(nu) / 2, x)[1]
    return zz > zero(zz) ? one(zz) - regularized / 2 : regularized / 2
end

# log(T_cdf(z; nu)) computed WITHOUT the `1 - regularized/2` cancellation: the
# regularized incomplete beta is small in the lower tail and arrives with
# relative accuracy, so its log stays finite where the plain CDF rounds to 0
# (or its complement rounds to 1); the upper side goes through log1p. This is
# the CPU port of the device `_device_std_t_log_cdf` (issue #43).
function _std_t_log_cdf(z, nu)
    zz = float(z)
    isinf(zz) && return zz > zero(zz) ? zero(zz) : oftype(zz, -Inf)
    zz == zero(zz) && return -log(oftype(zz, 2))
    # compute in at least Float64: a Float32 tail underflows the VALUE-space
    # incomplete beta (beta_inc -> 0.0f0) long before its log leaves the
    # Float32 range, so widen, take the log, and narrow the result
    W = promote_type(typeof(zz), Float64)
    zw = W(zz)
    nuw = W(nu)
    x = nuw / (nuw + zw * zw)
    regularized = beta_inc(nuw / 2, one(nuw) / 2, x)[1]
    zw < zero(zw) && return oftype(zz, log(regularized) - log(W(2)))
    return oftype(zz, log1p(-regularized / 2))
end

# log(T_cdf(zb) - T_cdf(za)): one-sided normalizers use the symmetry
# S(z) = cdf(-z) so a tail probability is computed directly rather than as
# 1 - cdf (which cancels at Float64 for light tails, e.g. nu = 1e5 with a 15
# sigma cutoff -- issue #43); finite intervals difference LOG CDFs through
# expm1, with the log-space midpoint form taking over once the difference
# sinks under sqrt(eps) (its own error is O(s^2) exactly there). Mirrors the
# device `_device_t_log_normalizer`.
function _t_log_normalizer(nu, za, zb)
    zaf, zbf = promote(float(za), float(zb))
    lower_infinite = isinf(zaf) && zaf < zero(zaf)
    upper_infinite = isinf(zbf) && zbf > zero(zbf)
    if lower_infinite && upper_infinite
        return zero(zaf)
    elseif lower_infinite
        return _std_t_log_cdf(zbf, nu)
    elseif upper_infinite
        return _std_t_log_cdf(-zaf, nu)
    end
    if zaf > zero(zaf) # right tail: S(za) - S(zb) = cdf(-za) - cdf(-zb)
        log_big = _std_t_log_cdf(-zaf, nu)
        log_small = _std_t_log_cdf(-zbf, nu)
    else # left tail and straddling: cdf(zb) - cdf(za)
        log_big = _std_t_log_cdf(zbf, nu)
        log_small = _std_t_log_cdf(zaf, nu)
    end
    s = log_small - log_big
    s < -sqrt(eps(float(one(s)))) && return log_big + log(-expm1(s))
    midpoint = (zaf + zbf) / 2
    return _std_t_log_pdf(midpoint, nu) + log(zbf - zaf)
end

# The nu-only Student-t normalizing constant
# loggamma((nu+1)/2) - loggamma(nu/2) - (log(nu) + log(pi))/2, computed in at
# least Float64: at Float32 the two ~nu*log(nu)-sized loggammas lose their
# O(log nu) difference to rounding (~0.03 absolute at nu = 1e5 -- issue #53).
# The z-dependent terms have no such cancellation and stay at input precision.
function _studentt_log_constant(nu)
    nuf = float(nu)
    W = promote_type(typeof(nuf), Float64)
    nuw = W(nuf)
    return oftype(nuf, loggamma((nuw + one(nuw)) / 2) - loggamma(nuw / 2) - (log(nuw) + log(W(pi))) / 2)
end

# Its nu-derivative, (digamma((nu+1)/2) - digamma(nu/2) - 1/nu) / 2, widened
# the same way: the digamma difference is ~1/nu against ~log(nu)-sized terms,
# so the Float32 analytic gradient would otherwise disagree with the
# Float64-widened value the ForwardDiff reference differentiates.
function _studentt_log_constant_dnu(nu)
    nuf = float(nu)
    W = promote_type(typeof(nuf), Float64)
    nuw = W(nuf)
    return oftype(nuf, (digamma((nuw + one(nuw)) / 2) - digamma(nuw / 2) - one(nuw) / nuw) / 2)
end

# Standard (unit-scale, zero-location) Student-t log-density; `_std_t_pdf`
# exponentiates it, and the truncated normalizer (midpoint fallback and the
# gradient's pdf/Z hazard ratios) uses it directly. Computed in at least
# Float64: at Float32 the loggamma((nu+1)/2) - loggamma(nu/2) difference loses
# ~0.05 to rounding for large nu, and the truncated gradient's cancellation
# structure (-k + pdf/Z, two nearly equal hazards) amplifies that into
# multiple-hundred-percent gradient errors.
function _std_t_log_pdf(z, nu)
    zz = float(z)
    W = promote_type(typeof(zz), Float64)
    zw = W(zz)
    nuw = W(nu)
    return oftype(
        zz,
        loggamma((nuw + one(nuw)) / 2) - loggamma(nuw / 2) -
        (log(nuw) + log(W(pi))) / 2 -
        (nuw + one(nuw)) * log1p((zw * zw) / nuw) / 2,
    )
end

# Standard (unit-scale, zero-location) Student-t density with `nu` degrees of
# freedom, guarded so an infinite standardized argument (an unbounded truncation
# side) yields a zero density.
function _std_t_pdf(z, nu)
    zz = float(z)
    isinf(zz) && return zero(zz)
    return exp(_std_t_log_pdf(zz, nu))
end

# Peel a degrees-of-freedom argument down to a plain value. `TruncatedStudentTDist`
# promotes every field to a common type, so a constant `nu` still arrives as a
# ForwardDiff Dual carrying zero partials — that is the tractable case. A `nu`
# with nonzero partials is a genuine latent dependence whose CDF derivative has no
# closed form, so it is rejected rather than silently mis-differentiated.
_constant_nu_value(nu::Real) = nu
function _constant_nu_value(nu::ForwardDiff.Dual)
    all(iszero, ForwardDiff.partials(nu)) || throw(
        ArgumentError(
            "truncatedstudentt gradient with respect to nu (degrees of freedom) is unsupported; nu must be a constant",
        ),
    )
    return _constant_nu_value(ForwardDiff.value(nu))
end

# ForwardDiff rule for the Student-t CDF differentiated through its `z` argument.
# `beta_inc` is not itself dual-differentiable, but d/dz T_cdf(z, nu) is exactly
# the Student-t density, a closed form. This keeps the CPU truncatedstudentt
# logpdf ForwardDiff-differentiable whenever `nu` is a constant (the only
# tractable case: the incomplete beta's nu-derivative has no closed form).
function _std_t_cdf(z::ForwardDiff.Dual{T}, nu::Real) where {T}
    zv = ForwardDiff.value(z)
    # An infinite standardized bound pins the CDF at 0/1 with a flat (zero)
    # derivative, independent of `nu`. Handle it before requiring a constant `nu`:
    # an unbounded truncation side needs no `d/dnu` term, so a latent `nu` stays
    # valid here (e.g. both bounds infinite). Skipping the pdf * partials product
    # also avoids an infinite partial surfacing as 0 * Inf = NaN.
    if isinf(zv)
        value = zv > zero(zv) ? one(zv) : zero(zv)
        return ForwardDiff.Dual{T}(value, zero(ForwardDiff.partials(z)))
    end
    nu_value = _constant_nu_value(nu)
    value = _std_t_cdf(zv, nu_value)
    derivative = _std_t_pdf(zv, nu_value)
    return ForwardDiff.Dual{T}(value, derivative * ForwardDiff.partials(z))
end

# The log-CDF analogue: d/dz log T_cdf(z) = exp(log pdf - log cdf), computed in
# log space so the ratio stays finite where the plain cdf underflows (the same
# rule the device dual path uses).
function _std_t_log_cdf(z::ForwardDiff.Dual{T}, nu::Real) where {T}
    zv = ForwardDiff.value(z)
    if isinf(zv)
        value = zv > zero(zv) ? zero(zv) : oftype(zv, -Inf)
        return ForwardDiff.Dual{T}(value, zero(ForwardDiff.partials(z)))
    end
    nu_value = _constant_nu_value(nu)
    value = _std_t_log_cdf(zv, nu_value)
    derivative = exp(_std_t_log_pdf(zv, nu_value) - value)
    return ForwardDiff.Dual{T}(value, derivative * ForwardDiff.partials(z))
end

function logpdf(dist::NormalDist, x)
    xx, mu, sigma = promote(x, dist.mu, dist.sigma)
    z = (xx - mu) / sigma
    return -log(sigma) - log(2 * pi) / 2 - z * z / 2
end

function logpdf(dist::LaplaceDist, x)
    xx, mu, scale = promote(x, dist.mu, dist.scale)
    return -log(2 * scale) - abs(xx - mu) / scale
end

function logpdf(dist::ExponentialDist, x)
    xx, rate = promote(x, dist.rate)
    xx >= zero(xx) || return oftype(xx, -Inf)
    return log(rate) - rate * xx
end

function logpdf(dist::GammaDist, x)
    xx, shape, rate = promote(x, dist.shape, dist.rate)
    xx > zero(xx) || return oftype(xx, -Inf)
    return shape * log(rate) - loggamma(shape) + (shape - one(shape)) * log(xx) - rate * xx
end

function logpdf(dist::InverseGammaDist, x)
    xx, shape, scale = promote(x, dist.shape, dist.scale)
    xx > zero(xx) || return oftype(xx, -Inf)
    return shape * log(scale) - loggamma(shape) - (shape + one(shape)) * log(xx) - scale / xx
end

function logpdf(dist::WeibullDist, x)
    xx, shape, scale = promote(x, dist.shape, dist.scale)
    xx < zero(xx) && return oftype(xx, -Inf)
    if xx == zero(xx)
        if shape < one(shape)
            return oftype(xx, Inf)
        elseif shape == one(shape)
            return -log(scale)
        end
        return oftype(xx, -Inf)
    end
    log_ratio = log(xx) - log(scale)
    return log(shape) + (shape - one(shape)) * log(xx) - shape * log(scale) - exp(shape * log_ratio)
end

function logpdf(dist::BetaDist, x)
    xx, alpha, beta_parameter = promote(x, dist.alpha, dist.beta)
    zero(xx) < xx < one(xx) || return oftype(xx, -Inf)
    return loggamma(alpha + beta_parameter) - loggamma(alpha) - loggamma(beta_parameter) +
           (alpha - one(alpha)) * log(xx) +
           (beta_parameter - one(beta_parameter)) * log1p(-xx)
end

function logpdf(dist::LogNormalDist, x)
    xx, mu, sigma = promote(x, dist.mu, dist.sigma)
    xx > zero(xx) || return oftype(xx, -Inf)
    return logpdf(normal(mu, sigma), log(xx)) - log(xx)
end

function logpdf(dist::StudentTDist, x)
    xx, nu, mu, sigma = promote(x, dist.nu, dist.mu, dist.sigma)
    z = (xx - mu) / sigma
    return _studentt_log_constant(nu) - log(sigma) -
           (nu + one(nu)) * log1p((z * z) / nu) / 2
end
