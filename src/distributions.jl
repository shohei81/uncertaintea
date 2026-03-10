abstract type AbstractTeaDistribution end

struct NormalDist{T<:AbstractFloat} <: AbstractTeaDistribution
    mu::T
    sigma::T

    function NormalDist(mu::T, sigma::T) where {T<:AbstractFloat}
        sigma > zero(T) || throw(ArgumentError("normal requires sigma > 0"))
        new{T}(mu, sigma)
    end
end

struct BernoulliDist{T<:AbstractFloat} <: AbstractTeaDistribution
    p::T

    function BernoulliDist(p::T) where {T<:AbstractFloat}
        zero(T) <= p <= one(T) || throw(ArgumentError("bernoulli requires 0 <= p <= 1"))
        new{T}(p)
    end
end

struct LogNormalDist{T<:AbstractFloat} <: AbstractTeaDistribution
    mu::T
    sigma::T

    function LogNormalDist(mu::T, sigma::T) where {T<:AbstractFloat}
        sigma > zero(T) || throw(ArgumentError("lognormal requires sigma > 0"))
        new{T}(mu, sigma)
    end
end

function normal(mu, sigma)
    T = float(promote_type(typeof(mu), typeof(sigma)))
    return NormalDist(convert(T, mu), convert(T, sigma))
end

function lognormal(mu, sigma)
    T = float(promote_type(typeof(mu), typeof(sigma)))
    return LogNormalDist(convert(T, mu), convert(T, sigma))
end

function bernoulli(p)
    T = float(typeof(p))
    return BernoulliDist(convert(T, p))
end

function Random.rand(rng::AbstractRNG, dist::NormalDist{T}) where {T}
    return dist.mu + dist.sigma * randn(rng, T)
end

function Random.rand(rng::AbstractRNG, dist::BernoulliDist)
    return rand(rng) < dist.p
end

function Random.rand(rng::AbstractRNG, dist::LogNormalDist{T}) where {T}
    return exp(dist.mu + dist.sigma * randn(rng, T))
end

function logpdf(dist::NormalDist, x)
    T = float(promote_type(typeof(x), typeof(dist.mu), typeof(dist.sigma)))
    xx = convert(T, x)
    mu = convert(T, dist.mu)
    sigma = convert(T, dist.sigma)
    z = (xx - mu) / sigma
    return -log(sigma) - T(log(2 * pi)) / 2 - z * z / 2
end

function logpdf(dist::BernoulliDist, x)
    value = x isa Bool ? x : x != 0
    return value ? log(dist.p) : log1p(-dist.p)
end

function logpdf(dist::LogNormalDist, x)
    T = float(promote_type(typeof(x), typeof(dist.mu), typeof(dist.sigma)))
    xx = convert(T, x)
    xx > zero(T) || return T(-Inf)
    return logpdf(normal(dist.mu, dist.sigma), log(xx)) - log(xx)
end
