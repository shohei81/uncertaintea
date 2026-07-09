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

function normal(mu, sigma)
    promoted_mu, promoted_sigma = promote(mu, sigma)
    return NormalDist(promoted_mu, promoted_sigma)
end

function laplace(mu, scale)
    promoted_mu, promoted_scale = promote(mu, scale)
    return LaplaceDist(promoted_mu, promoted_scale)
end

function exponential(rate)
    return ExponentialDist(rate)
end

function gamma(shape, rate)
    promoted_shape, promoted_rate = promote(shape, rate)
    return GammaDist(promoted_shape, promoted_rate)
end

function inversegamma(shape, scale)
    promoted_shape, promoted_scale = promote(shape, scale)
    return InverseGammaDist(promoted_shape, promoted_scale)
end

function weibull(shape, scale)
    promoted_shape, promoted_scale = promote(shape, scale)
    return WeibullDist(promoted_shape, promoted_scale)
end

function beta(alpha, beta_parameter)
    promoted_alpha, promoted_beta = promote(alpha, beta_parameter)
    return BetaDist(promoted_alpha, promoted_beta)
end

function lognormal(mu, sigma)
    promoted_mu, promoted_sigma = promote(mu, sigma)
    return LogNormalDist(promoted_mu, promoted_sigma)
end

function studentt(nu, mu, sigma)
    promoted_nu, promoted_mu, promoted_sigma = promote(nu, mu, sigma)
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

# Standard (unit-scale, zero-location) Student-t density with `nu` degrees of
# freedom, guarded so an infinite standardized argument (an unbounded truncation
# side) yields a zero density.
function _std_t_pdf(z, nu)
    zz = float(z)
    isinf(zz) && return zero(zz)
    nu_ = oftype(zz, nu)
    return exp(
        loggamma((nu_ + one(nu_)) / 2) - loggamma(nu_ / 2) -
        (log(nu_) + log(oftype(zz, pi))) / 2 -
        (nu_ + one(nu_)) * log1p((zz * zz) / nu_) / 2,
    )
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
    return loggamma((nu + one(nu)) / 2) - loggamma(nu / 2) -
           (log(nu) + log(pi)) / 2 - log(sigma) -
           (nu + one(nu)) * log1p((z * z) / nu) / 2
end
