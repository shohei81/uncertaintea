# CPU-reference distributions (structs, builders, logpdf, rand): truncated families (truncatednormal, truncatedstudentt).

struct TruncatedNormalDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    sigma::T
    lower::T
    upper::T

    function TruncatedNormalDist(mu::T, sigma::T, lower::T, upper::T) where {T<:Real}
        sigma > zero(T) || throw(ArgumentError("truncatednormal requires sigma > 0"))
        lower < upper || throw(ArgumentError("truncatednormal requires lower < upper"))
        new{T}(mu, sigma, lower, upper)
    end
end

struct TruncatedStudentTDist{T<:Real} <: AbstractTeaDistribution
    nu::T
    mu::T
    sigma::T
    lower::T
    upper::T

    function TruncatedStudentTDist(nu::T, mu::T, sigma::T, lower::T, upper::T) where {T<:Real}
        nu > zero(T) || throw(ArgumentError("truncatedstudentt requires nu > 0"))
        sigma > zero(T) || throw(ArgumentError("truncatedstudentt requires sigma > 0"))
        lower < upper || throw(ArgumentError("truncatedstudentt requires lower < upper"))
        new{T}(nu, mu, sigma, lower, upper)
    end
end

function truncatednormal(mu, sigma, lower, upper)
    promoted_mu, promoted_sigma, promoted_lower, promoted_upper = promote(mu, sigma, lower, upper)
    return TruncatedNormalDist(promoted_mu, promoted_sigma, promoted_lower, promoted_upper)
end

function truncatedstudentt(nu, mu, sigma, lower, upper)
    promoted_nu, promoted_mu, promoted_sigma, promoted_lower, promoted_upper =
        promote(nu, mu, sigma, lower, upper)
    return TruncatedStudentTDist(
        promoted_nu,
        promoted_mu,
        promoted_sigma,
        promoted_lower,
        promoted_upper,
    )
end

_std_normal_cdf(z) = erfc(-z / sqrt(oftype(float(z), 2))) / 2
_std_normal_invcdf(u) = sqrt(oftype(float(u), 2)) * erfinv(2 * u - one(u))

function _log_normal_cdf_diff(za, zb)
    root2 = sqrt(oftype(float(za), 2))
    log2 = log(oftype(float(za), 2))
    lower_infinite = isinf(za) && za < zero(za)
    upper_infinite = isinf(zb) && zb > zero(zb)
    # Handle infinite bounds explicitly: Φ(-∞)=0, Φ(+∞)=1. This keeps erf/erfc off
    # arguments whose ForwardDiff partials would otherwise be NaN.
    if lower_infinite && upper_infinite
        return zero(float(za))
    elseif lower_infinite
        return log(erfc(-zb / root2)) - log2
    elseif upper_infinite
        return log(erfc(za / root2)) - log2
    elseif za > zero(za)
        return log(erfc(za / root2) - erfc(zb / root2)) - log2
    elseif zb < zero(zb)
        return log(erfc(-zb / root2) - erfc(-za / root2)) - log2
    end
    return log(erf(zb / root2) - erf(za / root2)) - log2
end

function Random.rand(rng::AbstractRNG, dist::TruncatedNormalDist)
    mu = float(dist.mu)
    sigma = float(dist.sigma)
    za = (float(dist.lower) - mu) / sigma
    zb = (float(dist.upper) - mu) / sigma
    lower_cdf = _std_normal_cdf(za)
    upper_cdf = _std_normal_cdf(zb)
    threshold = lower_cdf + rand(rng, typeof(mu)) * (upper_cdf - lower_cdf)
    threshold = min(max(threshold, floatmin(typeof(mu))), prevfloat(one(mu)))
    return mu + sigma * _std_normal_invcdf(threshold)
end

function Random.rand(rng::AbstractRNG, dist::TruncatedStudentTDist)
    nu = float(dist.nu)
    mu = float(dist.mu)
    sigma = float(dist.sigma)
    lower = float(dist.lower)
    upper = float(dist.upper)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    lower_cdf = _std_t_cdf(za, nu)
    upper_cdf = _std_t_cdf(zb, nu)
    mass = upper_cdf - lower_cdf

    if mass < oftype(mass, 1e-6)
        target = lower_cdf + rand(rng, typeof(mu)) * mass
        lo = isfinite(za) ? za : (isfinite(zb) ? zb - oftype(zb, 50) : oftype(nu, -50))
        hi = isfinite(zb) ? zb : (isfinite(za) ? za + oftype(za, 50) : oftype(nu, 50))
        for _ = 1:80
            mid = (lo + hi) / 2
            if _std_t_cdf(mid, nu) < target
                lo = mid
            else
                hi = mid
            end
        end
        return mu + sigma * (lo + hi) / 2
    end

    base = studentt(nu, mu, sigma)
    while true
        draw = rand(rng, base)
        (draw >= lower && draw <= upper) && return draw
    end
end

function logpdf(dist::TruncatedNormalDist, x)
    xx, mu, sigma, lower, upper = promote(x, dist.mu, dist.sigma, dist.lower, dist.upper)
    (xx < lower || xx > upper) && return oftype(xx, -Inf)
    base = logpdf(normal(mu, sigma), xx)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    return base - _log_normal_cdf_diff(za, zb)
end

function logpdf(dist::TruncatedStudentTDist, x)
    xx, nu, mu, sigma, lower, upper =
        promote(x, dist.nu, dist.mu, dist.sigma, dist.lower, dist.upper)
    (xx < lower || xx > upper) && return oftype(xx, -Inf)
    base = logpdf(studentt(nu, mu, sigma), xx)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    return base - log(_std_t_cdf(zb, nu) - _std_t_cdf(za, nu))
end
