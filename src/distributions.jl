abstract type AbstractTeaDistribution end

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

struct DirichletDist{T<:Real} <: AbstractTeaDistribution
    alpha::Vector{T}

    function DirichletDist(alpha::Vector{T}) where {T<:Real}
        length(alpha) >= 2 || throw(ArgumentError("dirichlet requires at least 2 concentration parameters"))
        for value in alpha
            value > zero(T) || throw(ArgumentError("dirichlet requires alpha > 0"))
        end
        return new{T}(alpha)
    end
end

struct BernoulliDist{T<:Real} <: AbstractTeaDistribution
    p::T

    function BernoulliDist(p::T) where {T<:Real}
        zero(T) <= p <= one(T) || throw(ArgumentError("bernoulli requires 0 <= p <= 1"))
        new{T}(p)
    end
end

struct GeometricDist{T<:Real} <: AbstractTeaDistribution
    p::T

    function GeometricDist(p::T) where {T<:Real}
        zero(T) < p <= one(T) || throw(ArgumentError("geometric requires 0 < p <= 1"))
        new{T}(p)
    end
end

struct BinomialDist{T<:Real} <: AbstractTeaDistribution
    trials::Int
    p::T

    function BinomialDist(trials::Int, p::T) where {T<:Real}
        trials >= 0 || throw(ArgumentError("binomial requires trials >= 0"))
        zero(T) <= p <= one(T) || throw(ArgumentError("binomial requires 0 <= p <= 1"))
        new{T}(trials, p)
    end
end

struct NegativeBinomialDist{T<:Real} <: AbstractTeaDistribution
    successes::T
    p::T

    function NegativeBinomialDist(successes::T, p::T) where {T<:Real}
        successes > zero(T) || throw(ArgumentError("negativebinomial requires successes > 0"))
        zero(T) < p <= one(T) || throw(ArgumentError("negativebinomial requires 0 < p <= 1"))
        new{T}(successes, p)
    end
end

struct CategoricalDist{T<:Real} <: AbstractTeaDistribution
    probabilities::Vector{T}

    function CategoricalDist(probabilities::Vector{T}) where {T<:Real}
        isempty(probabilities) && throw(ArgumentError("categorical requires at least one probability"))
        total = zero(T)
        for probability in probabilities
            zero(T) <= probability <= one(T) || throw(ArgumentError("categorical requires 0 <= p <= 1"))
            total += probability
        end
        tolerance = sqrt(eps(float(total))) * max(length(probabilities), 1) * 8
        abs(total - one(total)) <= tolerance || throw(ArgumentError("categorical probabilities must sum to 1"))
        new{T}(probabilities)
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

function dirichlet(alpha::AbstractVector)
    promoted = map(float, collect(alpha))
    return DirichletDist(promoted)
end

function dirichlet(alpha::Vararg{Real})
    return DirichletDist(collect(promote(alpha...)))
end

function lognormal(mu, sigma)
    promoted_mu, promoted_sigma = promote(mu, sigma)
    return LogNormalDist(promoted_mu, promoted_sigma)
end

function bernoulli(p)
    return BernoulliDist(p)
end

function geometric(p)
    return GeometricDist(p)
end

function binomial(trials, p)
    count = _binomial_trials(trials)
    isnothing(count) && throw(ArgumentError("binomial requires integer trials >= 0"))
    return BinomialDist(count, p)
end

function negativebinomial(successes, p)
    promoted_successes, promoted_probability = promote(successes, p)
    return NegativeBinomialDist(promoted_successes, promoted_probability)
end

function categorical(probabilities::AbstractVector)
    promoted = map(float, collect(probabilities))
    return CategoricalDist(promoted)
end

function categorical(probabilities::Vararg{Real})
    promoted = collect(promote(probabilities...))
    return CategoricalDist(promoted)
end

function poisson(lambda)
    return PoissonDist(lambda)
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

function Random.rand(rng::AbstractRNG, dist::DirichletDist)
    draws = Vector{eltype(dist.alpha)}(undef, length(dist.alpha))
    total = zero(eltype(dist.alpha))
    for index in eachindex(dist.alpha)
        draw = _rand_gamma_marsaglia(rng, float(dist.alpha[index]), one(float(dist.alpha[index])))
        draws[index] = draw
        total += draw
    end
    for index in eachindex(draws)
        draws[index] /= total
    end
    return draws
end

function Random.rand(rng::AbstractRNG, dist::BernoulliDist)
    return rand(rng) < dist.p
end

function Random.rand(rng::AbstractRNG, dist::GeometricDist)
    probability = float(dist.p)
    probability == one(probability) && return 0
    threshold = max(rand(rng, typeof(probability)), floatmin(typeof(probability)))
    return floor(Int, log(threshold) / log1p(-probability))
end

function Random.rand(rng::AbstractRNG, dist::BinomialDist)
    successes = 0
    for _ in 1:dist.trials
        successes += rand(rng) < dist.p
    end
    return successes
end

function Random.rand(rng::AbstractRNG, dist::NegativeBinomialDist)
    probability = float(dist.p)
    probability == one(probability) && return 0
    rate = probability / (1 - probability)
    lambda = _rand_gamma_marsaglia(rng, float(dist.successes), rate)
    return rand(rng, poisson(lambda))
end

function Random.rand(rng::AbstractRNG, dist::CategoricalDist)
    threshold = rand(rng, eltype(dist.probabilities))
    cumulative = zero(threshold)
    for (index, probability) in enumerate(dist.probabilities)
        cumulative += probability
        threshold <= cumulative && return index
    end
    return length(dist.probabilities)
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

function Random.rand(rng::AbstractRNG, dist::StudentTDist)
    nu = float(dist.nu)
    mu = float(dist.mu)
    sigma = float(dist.sigma)
    scale = randn(rng, typeof(mu)) / sqrt(_rand_gamma_marsaglia(rng, nu / 2, nu / 2))
    return mu + sigma * scale
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

function logpdf(dist::DirichletDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("dirichlet logpdf expects a vector or tuple value"))
    length(values) == length(dist.alpha) || return -Inf

    promoted_values = map(float, collect(values))
    total = zero(eltype(promoted_values))
    accumulator = loggamma(sum(dist.alpha)) - sum(loggamma, dist.alpha)
    for (value, alpha) in zip(promoted_values, dist.alpha)
        value > zero(value) || return oftype(value, -Inf)
        total += value
        accumulator += (alpha - one(alpha)) * log(value)
    end
    abs(total - one(total)) <= sqrt(eps(float(total))) * length(promoted_values) * 16 || return oftype(total, -Inf)
    return accumulator
end

function logpdf(dist::BernoulliDist, x)
    value = x isa Bool ? x : x != 0
    return value ? log(dist.p) : log1p(-dist.p)
end

function logpdf(dist::GeometricDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.p), -Inf)
    if count == 0
        return log(dist.p)
    elseif dist.p == one(dist.p)
        return oftype(float(dist.p), -Inf)
    end
    return log(dist.p) + count * log1p(-dist.p)
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

_binomial_trials(x) = _poisson_count(x)

function _categorical_index(x, categories::Int)
    if x isa Integer
        return 1 <= x <= categories ? Int(x) : nothing
    elseif x isa Real && isfinite(x)
        truncated = trunc(x)
        return one(x) <= x <= categories && x == truncated ? Int(truncated) : nothing
    end
    return nothing
end

function logpdf(dist::CategoricalDist, x)
    index = _categorical_index(x, length(dist.probabilities))
    isnothing(index) && return oftype(float(dist.probabilities[1]), -Inf)
    return log(dist.probabilities[index])
end

function _logfactorial_like(value, n::Integer)
    total = zero(value)
    unit = one(value)
    for k in 2:n
        total += log(unit * k)
    end
    return total
end

function _logbinomial_like(value, n::Integer, k::Integer)
    return _logfactorial_like(value, n) -
           _logfactorial_like(value, k) -
           _logfactorial_like(value, n - k)
end

function logpdf(dist::BinomialDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.p), -Inf)
    count <= dist.trials || return oftype(float(dist.p), -Inf)
    probability = dist.p
    log_combination = _logbinomial_like(probability, dist.trials, count)
    if count == 0 && count == dist.trials
        return log_combination
    elseif count == 0
        return log_combination + dist.trials * log1p(-probability)
    elseif count == dist.trials
        return log_combination + count * log(probability)
    end
    return log_combination +
           count * log(probability) +
           (dist.trials - count) * log1p(-probability)
end

function logpdf(dist::NegativeBinomialDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.p), -Inf)
    successes, probability = promote(dist.successes, dist.p)
    if count == 0 && probability == one(probability)
        return zero(probability)
    elseif probability == one(probability)
        return oftype(probability, -Inf)
    end
    return loggamma(count + successes) - loggamma(successes) - _logfactorial_like(probability, count) +
           successes * log(probability) + count * log1p(-probability)
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

function logpdf(dist::StudentTDist, x)
    xx, nu, mu, sigma = promote(x, dist.nu, dist.mu, dist.sigma)
    z = (xx - mu) / sigma
    return loggamma((nu + one(nu)) / 2) - loggamma(nu / 2) -
           (log(nu) + log(pi)) / 2 - log(sigma) -
           (nu + one(nu)) * log1p((z * z) / nu) / 2
end
