# CPU-reference distributions (structs, builders, logpdf, rand): structured families (mvnormal, mvnormaldense, dirichlet, mixture, broadcast/iid vector machinery).

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

# T<:Real (not AbstractFloat) so the ForwardDiff fallback can score models
# whose mean/scale vectors carry Dual entries (e.g. a latent mean), matching
# NormalDist and DirichletDist; rand stays restricted to AbstractFloat.
struct MvNormalDist{T<:Real} <: AbstractTeaDistribution
    mu::Vector{T}
    sigma::Vector{T}

    function MvNormalDist(mu::Vector{T}, sigma::Vector{T}) where {T<:Real}
        isempty(mu) && throw(ArgumentError("mvnormal requires at least one dimension"))
        length(mu) == length(sigma) || throw(ArgumentError("mvnormal requires mean and scale vectors with the same length"))
        for value in sigma
            value > zero(T) || throw(ArgumentError("mvnormal requires sigma > 0 in every dimension"))
        end
        return new{T}(mu, sigma)
    end
end

# Finite marginalized mixture of AbstractTeaDistribution components. `weights` is
# kept generic (rather than pinned to Float64) so a latent simplex fed in through a
# dirichlet parameter slot arrives as ForwardDiff Duals and stays differentiable.
struct MixtureDist{W<:Real,C<:Tuple} <: AbstractTeaDistribution
    weights::Vector{W}
    components::C

    function MixtureDist(weights::Vector{W}, components::C) where {W<:Real,C<:Tuple}
        isempty(components) && throw(ArgumentError("mixture requires at least one component"))
        length(weights) == length(components) || throw(
            ArgumentError(
                "mixture requires one weight per component (got $(length(weights)) weights for $(length(components)) components)",
            ),
        )
        total = zero(W)
        for w in weights
            w >= zero(W) || throw(ArgumentError("mixture weights must be nonnegative"))
            total += w
        end
        abs(total - one(W)) <= oftype(total, 1e-8) ||
            throw(ArgumentError("mixture weights must sum to 1 (within 1e-8)"))
        for comp in components
            comp isa AbstractTeaDistribution ||
                throw(ArgumentError("mixture components must be UncertainTea distributions"))
        end
        return new{W,C}(weights, components)
    end
end

# Dense-covariance multivariate normal parameterized by a lower-triangular
# Cholesky factor `scale_tril` (covariance = L * L'). Element types stay generic
# (like MixtureDist) so a latent mean built from ForwardDiff Duals remains
# differentiable. Only the lower triangle of `scale_tril` is ever read — any
# upper-triangular content is ignored — which lets callers pass a full matrix
# without wrapping it in `LinearAlgebra.LowerTriangular`.
struct MvNormalDenseDist{M<:AbstractVector,S<:AbstractMatrix} <: AbstractTeaDistribution
    mu::M
    scale_tril::S

    function MvNormalDenseDist(mu::AbstractVector, scale_tril::AbstractMatrix)
        isempty(mu) && throw(ArgumentError("mvnormaldense requires at least one dimension"))
        size(scale_tril, 1) == size(scale_tril, 2) || throw(ArgumentError(
            "mvnormaldense requires a square scale_tril matrix, got size $(size(scale_tril))",
        ))
        size(scale_tril, 1) == length(mu) || throw(
            ArgumentError(
                "mvnormaldense requires scale_tril of size $(length(mu))x$(length(mu)) to match the mean length, got $(size(scale_tril))",
            ),
        )
        for index = 1:size(scale_tril, 1)
            scale_tril[index, index] > 0 || throw(ArgumentError(
                "mvnormaldense requires strictly positive scale_tril diagonal entries",
            ))
        end
        return new{typeof(mu),typeof(scale_tril)}(mu, scale_tril)
    end
end

# LKJ prior over the Cholesky factor of a d x d correlation matrix, scored on
# the column-major PACKED lower triangle (length d*(d+1)/2, diagonal included) —
# the same layout `CholeskyCorrTransform` produces. `eta` stays generic so a
# latent-dependent concentration remains differentiable.
struct LKJCholeskyDist{T<:Real} <: AbstractTeaDistribution
    d::Int
    eta::T

    function LKJCholeskyDist(d::Int, eta::Real)
        d >= 2 || throw(ArgumentError("lkjcholesky requires a dimension d >= 2"))
        eta > 0 || throw(ArgumentError("lkjcholesky requires a concentration eta > 0"))
        promoted_eta = float(eta)
        return new{typeof(promoted_eta)}(d, promoted_eta)
    end
end

# Broadcast (vectorized) normal observation. `mu` and `sigma` may each be a real
# scalar or an `AbstractVector`; a single vector-valued choice scores every element
# elementwise. This is the runtime counterpart of the `{:y} ~ normal.(mu, sigma)`
# dot-call DSL syntax and of the backend `BackendBroadcastNormalChoicePlanStep`.
struct BroadcastNormalDist{M,S} <: AbstractTeaDistribution
    mu::M
    sigma::S
end

BroadcastNormalDist(mu::Union{Real,AbstractVector}, sigma::Union{Real,AbstractVector}) =
    BroadcastNormalDist{typeof(mu),typeof(sigma)}(mu, sigma)

_broadcast_arg_length(::Real) = nothing
_broadcast_arg_length(v::AbstractVector) = length(v)
_broadcast_arg_getindex(x::Real, ::Int) = x
_broadcast_arg_getindex(v::AbstractVector, i::Int) = v[i]

function _broadcast_normal_length(mu, sigma)
    mu_length = _broadcast_arg_length(mu)
    sigma_length = _broadcast_arg_length(sigma)
    if !isnothing(mu_length) && !isnothing(sigma_length)
        mu_length == sigma_length ||
            throw(
                DimensionMismatch(
                    "broadcast normal requires vector arguments of equal length, got $(mu_length) and $(sigma_length)",
                ),
            )
        return mu_length
    end
    return something(mu_length, sigma_length, Some(nothing))
end

function _broadcast_normal_check_length(mu, sigma, n::Int)
    for arg in (mu, sigma)
        arg_length = _broadcast_arg_length(arg)
        isnothing(arg_length) && continue
        arg_length == n ||
            throw(DimensionMismatch("broadcast normal argument length $(arg_length) does not match value length $n"))
    end
    return nothing
end

function iid(base::AbstractTeaDistribution, n::Integer)
    n >= 1 || throw(ArgumentError("iid requires n >= 1"))
    return IIDDist(base, Int(n))
end

# `n` independent-and-identically-distributed draws from `base` under a single
# address. Runtime counterpart of the `eps ~ iid(dist_call, n)` DSL sugar.
struct IIDDist{D<:AbstractTeaDistribution} <: AbstractTeaDistribution
    base::D
    n::Int

    function IIDDist(base::D, n::Int) where {D<:AbstractTeaDistribution}
        n >= 1 || throw(ArgumentError("iid requires n >= 1"))
        return new{D}(base, n)
    end
end

function dirichlet(alpha::AbstractVector)
    promoted = map(float, collect(alpha))
    return DirichletDist(promoted)
end

function dirichlet(alpha::Vararg{Real})
    return DirichletDist(collect(promote(alpha...)))
end

function _mvnormal_vector(values)
    if values isa AbstractVector || values isa Tuple
        return collect(values)
    end
    throw(ArgumentError("mvnormal requires vector-like mean and scale arguments"))
end

function _mvnormal_promoted_type(vectors...)
    promoted_type = nothing
    for vector in vectors
        for value in vector
            promoted_type = isnothing(promoted_type) ? typeof(value) : promote_type(promoted_type, typeof(value))
        end
    end
    isnothing(promoted_type) && throw(ArgumentError("mvnormal requires at least one dimension"))
    return float(promoted_type)
end

function mvnormal(mu, sigma)
    mu_values = _mvnormal_vector(mu)
    sigma_values = _mvnormal_vector(sigma)
    length(mu_values) == length(sigma_values) ||
        throw(ArgumentError("mvnormal requires mean and scale vectors with the same length"))
    isempty(mu_values) && throw(ArgumentError("mvnormal requires at least one dimension"))
    promoted_type = _mvnormal_promoted_type(mu_values, sigma_values)
    return MvNormalDist(
        promoted_type[value for value in mu_values],
        promoted_type[value for value in sigma_values],
    )
end

function mvnormaldense(mu, scale_tril)
    (mu isa AbstractVector || mu isa Tuple) ||
        throw(ArgumentError("mvnormaldense requires a vector-like mean argument"))
    scale_tril isa AbstractMatrix ||
        throw(ArgumentError("mvnormaldense requires a matrix scale_tril argument"))
    # `map(identity, collect(...))` narrows a `Vector{Any}` (e.g. a compiled
    # `[mu, mu]` vector expression holding ForwardDiff Duals) to the promoted
    # element type so downstream arithmetic stays differentiable.
    mu_values = map(identity, collect(mu))
    return MvNormalDenseDist(mu_values, scale_tril)
end

# `d` accepts any integer-valued number: the compiled evaluator reconstructs the
# distribution from the `DistributionSpec` argument vector, where an integer
# literal may arrive promoted to Float64 (see `_lkjcholesky_static_size`).
function lkjcholesky(d, eta)
    (d isa Real && isinteger(d)) ||
        throw(ArgumentError("lkjcholesky requires an integer dimension d"))
    return LKJCholeskyDist(Int(d), eta)
end

# Log of the LKJ normalizing constant over correlation matrices (Lewandowski,
# Kurowicka & Joe 2009, cvine construction): the density over the free
# below-diagonal Cholesky coordinates is
#   prod_{i=2..d} L[i,i]^(d - i + 2*eta - 2) / c_d(eta)
# with
#   log c_d(eta) = sum_{k=1}^{d-1} (d - k) * [(2*eta - 2 + d - k) * log(2)
#                                             + log Beta(a_k, a_k)],
#   a_k = eta + (d - 1 - k) / 2,
# i.e. one symmetric Beta factor (rescaled to (-1, 1)) per canonical partial
# correlation at cvine tree level k, replicated (d - k) times.
function _lkj_log_normalizing_constant(d::Int, eta)
    eta_f = float(eta)
    log_c = zero(eta_f)
    for k = 1:(d-1)
        a = eta_f + (d - 1 - k) / 2
        log_beta = loggamma(a) + loggamma(a) - loggamma(2 * a)
        log_c += (d - k) * ((2 * eta_f - 2 + d - k) * log(oftype(eta_f, 2)) + log_beta)
    end
    return -log_c
end

# Un-pack a column-major packed correlation Cholesky factor and scale row i by
# `scales[i]`, producing the dense d x d lower-triangular `scale_tril` for
# `mvnormaldense` (covariance = diag(scales) * Omega * diag(scales)). Plain
# loops keep ForwardDiff Duals flowing through, and `map(identity, collect(...))`
# narrows `Vector{Any}` inputs from compiled expressions (mirroring
# `mvnormaldense`).
function scale_cholesky(scales::AbstractVector, packed_corr_chol::AbstractVector)
    isempty(scales) && throw(ArgumentError("scale_cholesky requires at least one scale"))
    scale_values = map(identity, collect(scales))
    packed_values = map(identity, collect(packed_corr_chol))
    d = length(scale_values)
    expected = (d * (d + 1)) ÷ 2
    length(packed_values) == expected || throw(
        DimensionMismatch(
            "scale_cholesky expected a packed lower triangle of length $expected for $d scales, got $(length(packed_values))",
        ),
    )
    T = promote_type(eltype(scale_values), eltype(packed_values))
    result = zeros(T, d, d)
    for col = 1:d
        for row = col:d
            result[row, col] = scale_values[row] * packed_values[_packed_lower_index(d, row, col)]
        end
    end
    return result
end

function mixture(weights, components...)
    isempty(components) && throw(ArgumentError("mixture requires at least one component"))
    return MixtureDist(collect(weights), components)
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

function Random.rand(rng::AbstractRNG, dist::MvNormalDist{T}) where {T<:AbstractFloat}
    draws = Vector{T}(undef, length(dist.mu))
    for index in eachindex(draws)
        draws[index] = dist.mu[index] + dist.sigma[index] * randn(rng, T)
    end
    return draws
end

# Draw mu + L * z with z ~ standard normal, reading only the lower triangle of
# `scale_tril`.
function Random.rand(rng::AbstractRNG, dist::MvNormalDenseDist)
    dimension = length(dist.mu)
    T = float(promote_type(typeof(float(dist.mu[1])), typeof(float(dist.scale_tril[1, 1]))))
    noise = randn(rng, T, dimension)
    draws = Vector{T}(undef, dimension)
    for row = 1:dimension
        accumulator = T(dist.mu[row])
        for col = 1:row
            accumulator += T(dist.scale_tril[row, col]) * noise[col]
        end
        draws[row] = accumulator
    end
    return draws
end

# LKJ cvine construction (Lewandowski, Kurowicka & Joe 2009): the canonical
# partial correlation feeding below-diagonal entry (i, j) sits at cvine tree
# level j and is distributed as w = 2 * Beta(a_j, a_j) - 1 with
# a_j = eta + (d - 1 - j) / 2; running Stan's `cholesky_corr_constrain` forward
# recursion on those w values yields the packed correlation Cholesky factor.
function Random.rand(rng::AbstractRNG, dist::LKJCholeskyDist)
    d = dist.d
    eta = Float64(dist.eta)
    packed = Vector{Float64}(undef, (d * (d + 1)) ÷ 2)
    packed[_packed_lower_index(d, 1, 1)] = 1.0
    for row = 2:d
        sum_sqs = 0.0
        for col = 1:(row-1)
            a = eta + (d - 1 - col) / 2
            w = 2.0 * rand(rng, BetaDist(a, a)) - 1.0
            entry = w * sqrt(1 - sum_sqs)
            packed[_packed_lower_index(d, row, col)] = entry
            sum_sqs += entry * entry
        end
        packed[_packed_lower_index(d, row, row)] = sqrt(1 - sum_sqs)
    end
    return packed
end

function Random.rand(rng::AbstractRNG, dist::BroadcastNormalDist)
    n = _broadcast_normal_length(dist.mu, dist.sigma)
    isnothing(n) && throw(
        ArgumentError(
            "broadcast normal with all-scalar arguments cannot infer a sample length; " *
            "constrain the observation or supply at least one vector argument",
        ),
    )
    mu1 = float(_broadcast_arg_getindex(dist.mu, 1))
    sigma1 = float(_broadcast_arg_getindex(dist.sigma, 1))
    T = float(promote_type(typeof(mu1), typeof(sigma1)))
    draws = Vector{T}(undef, n)
    for index = 1:n
        mu = _broadcast_arg_getindex(dist.mu, index)
        sigma = _broadcast_arg_getindex(dist.sigma, index)
        draws[index] = mu + sigma * randn(rng, T)
    end
    return draws
end

function Random.rand(rng::AbstractRNG, dist::IIDDist)
    first_draw = rand(rng, dist.base)
    draws = Vector{typeof(first_draw)}(undef, dist.n)
    draws[1] = first_draw
    for index = 2:dist.n
        draws[index] = rand(rng, dist.base)
    end
    return draws
end

function Random.rand(rng::AbstractRNG, dist::MixtureDist)
    threshold = rand(rng, Float64)
    cumulative = 0.0
    index = length(dist.components)
    for (k, w) in enumerate(dist.weights)
        cumulative += w
        if threshold <= cumulative
            index = k
            break
        end
    end
    return rand(rng, dist.components[index])
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

function logpdf(dist::MvNormalDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("mvnormal logpdf expects a vector or tuple value"))
    length(values) == length(dist.mu) || return -Inf

    accumulator = logpdf(normal(dist.mu[1], dist.sigma[1]), values[1])
    for index = 2:length(values)
        accumulator += logpdf(normal(dist.mu[index], dist.sigma[index]), values[index])
    end
    return accumulator
end

# Dense mvnormal log density via hand-rolled forward substitution solving
# L z = x - mu, reading only the lower triangle of `scale_tril`. Avoiding
# LinearAlgebra factorization objects keeps every operation a plain scalar
# loop that ForwardDiff Duals flow through:
#   logpdf = -sum_i log(L[i,i]) - z'z / 2 - d * log(2*pi) / 2.
function logpdf(dist::MvNormalDenseDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("mvnormaldense logpdf expects a vector or tuple value"))
    dimension = length(dist.mu)
    length(values) == dimension || return -Inf
    L = dist.scale_tril
    z1 = (values[1] - dist.mu[1]) / L[1, 1]
    solved = Vector{typeof(z1)}(undef, dimension)
    solved[1] = z1
    log_det = log(L[1, 1])
    quadratic = z1 * z1
    for row = 2:dimension
        residual = values[row] - dist.mu[row]
        for col = 1:(row-1)
            residual -= L[row, col] * solved[col]
        end
        z = residual / L[row, row]
        solved[row] = z
        log_det += log(L[row, row])
        quadratic += z * z
    end
    return -log_det - quadratic / 2 - dimension * log(2 * pi) / 2
end

# LKJ log density over the free (below-diagonal) coordinates of the packed
# correlation Cholesky factor:
#   lpdf = sum_{i=2..d} (d - i + 2*eta - 2) * log(L[i,i]) + log(1 / c_d(eta)).
# Support: packed length d*(d+1)/2, strictly positive diagonal entries, and
# every row of the unpacked factor a unit-or-shorter vector (else -Inf).
function logpdf(dist::LKJCholeskyDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("lkjcholesky logpdf expects a vector or tuple value"))
    d = dist.d
    expected = (d * (d + 1)) ÷ 2
    length(values) == expected || return -Inf

    accumulator = _lkj_log_normalizing_constant(d, dist.eta) + zero(float(values[firstindex(values)]))
    tolerance = sqrt(eps(Float64)) * d * 16
    for row = 1:d
        diagonal = values[_packed_lower_index(d, row, row)]
        diagonal > zero(diagonal) || return oftype(accumulator, -Inf)
        sum_sqs = zero(float(diagonal))
        for col = 1:row
            entry = values[_packed_lower_index(d, row, col)]
            sum_sqs += entry * entry
        end
        sum_sqs <= 1 + tolerance || return oftype(accumulator, -Inf)
        if row >= 2
            accumulator += (d - row + 2 * dist.eta - 2) * log(diagonal)
        end
    end
    return accumulator
end

function logpdf(dist::BroadcastNormalDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("broadcast normal logpdf expects a vector or tuple value"))
    n = length(values)
    n >= 1 || throw(ArgumentError("broadcast normal requires a non-empty value"))
    _broadcast_normal_check_length(dist.mu, dist.sigma, n)
    accumulator = logpdf(
        normal(_broadcast_arg_getindex(dist.mu, 1), _broadcast_arg_getindex(dist.sigma, 1)),
        values[1],
    )
    for index = 2:n
        accumulator += logpdf(
            normal(_broadcast_arg_getindex(dist.mu, index), _broadcast_arg_getindex(dist.sigma, index)),
            values[index],
        )
    end
    return accumulator
end

function logpdf(dist::IIDDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("iid logpdf expects a vector or tuple value"))
    length(values) == dist.n ||
        throw(DimensionMismatch("iid expects a value of length $(dist.n), got $(length(values))"))
    accumulator = logpdf(dist.base, values[1])
    for index = 2:dist.n
        accumulator += logpdf(dist.base, values[index])
    end
    return accumulator
end

# Per-component terms log(w_k) + logpdf(component_k, x) as a tuple. A zero weight
# yields log(0) = -Inf, which drops out of the max-shifted logsumexp below, so no
# explicit skipping is needed. Recursion keeps the tuple type-stable.
function _mixture_log_terms(weights, components::Tuple, x, index::Int)
    isempty(components) && return ()
    term = log(weights[index]) + logpdf(first(components), x)
    return (term, _mixture_log_terms(weights, Base.tail(components), x, index + 1)...)
end

function logpdf(dist::MixtureDist, x)
    terms = _mixture_log_terms(dist.weights, dist.components, x, 1)
    m = maximum(terms)
    isfinite(m) || return oftype(m, -Inf)
    total = zero(m)
    for term in terms
        total += exp(term - m)
    end
    return m + log(total)
end
