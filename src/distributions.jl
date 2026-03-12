abstract type AbstractTeaDistribution end

struct NormalDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    sigma::T

    function NormalDist(mu::T, sigma::T) where {T<:Real}
        sigma > zero(T) || throw(ArgumentError("normal requires sigma > 0"))
        new{T}(mu, sigma)
    end
end

struct ExponentialDist{T<:Real} <: AbstractTeaDistribution
    rate::T

    function ExponentialDist(rate::T) where {T<:Real}
        rate > zero(T) || throw(ArgumentError("exponential requires rate > 0"))
        new{T}(rate)
    end
end

struct BernoulliDist{T<:Real} <: AbstractTeaDistribution
    p::T

    function BernoulliDist(p::T) where {T<:Real}
        zero(T) <= p <= one(T) || throw(ArgumentError("bernoulli requires 0 <= p <= 1"))
        new{T}(p)
    end
end

struct PoissonDist{T<:Real} <: AbstractTeaDistribution
    lambda::T

    function PoissonDist(lambda::T) where {T<:Real}
        lambda > zero(T) || throw(ArgumentError("poisson requires lambda > 0"))
        new{T}(lambda)
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

function normal(mu, sigma)
    promoted_mu, promoted_sigma = promote(mu, sigma)
    return NormalDist(promoted_mu, promoted_sigma)
end

function exponential(rate)
    return ExponentialDist(rate)
end

function lognormal(mu, sigma)
    promoted_mu, promoted_sigma = promote(mu, sigma)
    return LogNormalDist(promoted_mu, promoted_sigma)
end

function bernoulli(p)
    return BernoulliDist(p)
end

function poisson(lambda)
    return PoissonDist(lambda)
end

function Random.rand(rng::AbstractRNG, dist::NormalDist{T}) where {T<:AbstractFloat}
    return dist.mu + dist.sigma * randn(rng, T)
end

function Random.rand(rng::AbstractRNG, dist::ExponentialDist)
    rate = float(dist.rate)
    return randexp(rng, typeof(rate)) / rate
end

function Random.rand(rng::AbstractRNG, dist::BernoulliDist)
    return rand(rng) < dist.p
end

function Random.rand(rng::AbstractRNG, dist::PoissonDist)
    lambda = float(dist.lambda)
    lambda == zero(lambda) && return 0
    limit = exp(-lambda)
    product = one(lambda)
    count = 0
    while product > limit
        count += 1
        product *= rand(rng, typeof(lambda))
    end
    return count - 1
end

function Random.rand(rng::AbstractRNG, dist::LogNormalDist{T}) where {T<:AbstractFloat}
    return exp(dist.mu + dist.sigma * randn(rng, T))
end

function logpdf(dist::NormalDist, x)
    xx, mu, sigma = promote(x, dist.mu, dist.sigma)
    z = (xx - mu) / sigma
    return -log(sigma) - log(2 * pi) / 2 - z * z / 2
end

function logpdf(dist::ExponentialDist, x)
    xx, rate = promote(x, dist.rate)
    xx >= zero(xx) || return oftype(xx, -Inf)
    return log(rate) - rate * xx
end

function logpdf(dist::BernoulliDist, x)
    value = x isa Bool ? x : x != 0
    return value ? log(dist.p) : log1p(-dist.p)
end

function _poisson_count(x)
    if x isa Integer
        return x >= 0 ? Int(x) : nothing
    elseif x isa Real && isfinite(x)
        truncated = trunc(x)
        return x >= zero(x) && x == truncated ? Int(truncated) : nothing
    end
    return nothing
end

function _logfactorial_like(value, n::Integer)
    total = zero(value)
    unit = one(value)
    for k in 2:n
        total += log(unit * k)
    end
    return total
end

function logpdf(dist::PoissonDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.lambda), -Inf)
    lambda = dist.lambda
    return count * log(lambda) - lambda - _logfactorial_like(lambda, count)
end

function logpdf(dist::LogNormalDist, x)
    xx, mu, sigma = promote(x, dist.mu, dist.sigma)
    xx > zero(xx) || return oftype(xx, -Inf)
    return logpdf(normal(mu, sigma), log(xx)) - log(xx)
end
