# Mass-metric abstraction for the single-chain CPU HMC/NUTS samplers.
#
# The legacy scalar code threads a diagonal inverse mass matrix as a plain
# `Vector{Float64}` (imm[i] plays the role of M^{-1} in the leapfrog drift,
# kinetic energy, and momentum sampling). That path is left completely untouched
# for bitwise reproducibility; production `metric=:diag` runs never construct a
# `MassMetric`. The types below add an optional dense metric for `metric=:dense`.
#
# Convention note: a `MassMetric`'s stored `inverse_mass` IS M^{-1} (the leapfrog
# drift coefficient / kinetic-energy weight), and momentum is drawn from N(0, M)
# where M = inv(M^{-1}). `DiagonalMetric(v)` and `DenseMetric(Diagonal(v))` are
# therefore interchangeable at the operation level. The dense *adaptation*
# estimates M^{-1} = posterior covariance Sigma (the standard Euclidean-HMC
# choice), so a dense run whitens correlated targets.

abstract type MassMetric end

struct DiagonalMetric <: MassMetric
    inverse_mass::Vector{Float64}
end

struct DenseMetric{U<:UpperTriangular{Float64}} <: MassMetric
    inverse_mass::Matrix{Float64}   # M^{-1} = estimated covariance Sigma
    chol_upper::U                   # U with Sigma = U'U; momentum p = U \ z ~ N(0, inv(Sigma)) = N(0, M)
end

# Build a dense metric from an inverse mass matrix (covariance) Sigma. We factor
# Sigma = L L' (cholesky), store the upper factor U = L'. Drawing z ~ N(0, I) and
# solving p = U \ z gives cov(p) = (U'U)^{-1} = Sigma^{-1} = M, exactly the
# momentum distribution N(0, M) required by the leapfrog integrator.
function DenseMetric(sigma::AbstractMatrix)
    n = LinearAlgebra.checksquare(sigma)
    symmetric = Matrix{Float64}(undef, n, n)
    @inbounds for j = 1:n, i = 1:n
        symmetric[i, j] = 0.5 * (Float64(sigma[i, j]) + Float64(sigma[j, i]))
    end
    factor = cholesky(Symmetric(symmetric))
    return DenseMetric(symmetric, factor.U)
end

# --- momentum sampling ------------------------------------------------------

# Diagonal metric: reproduce the legacy `randn(rng, n) ./ sqrt.(imm)` element
# order exactly so a `DiagonalMetric` draw matches the raw-Vector draw bitwise.
function sample_momentum!(p::AbstractVector, metric::DiagonalMetric, rng::AbstractRNG)
    z = randn(rng, length(p))
    p .= z ./ sqrt.(metric.inverse_mass)
    return p
end

function sample_momentum!(p::AbstractVector, metric::DenseMetric, rng::AbstractRNG)
    z = randn(rng, length(p))
    copyto!(p, z)
    ldiv!(metric.chol_upper, p)   # p = U \ z
    return p
end

# --- M^{-1} application (leapfrog drift coefficient) ------------------------

function apply_inverse_mass!(out::AbstractVector, metric::DiagonalMetric, p::AbstractVector)
    out .= metric.inverse_mass .* p
    return out
end

function apply_inverse_mass!(out::AbstractVector, metric::DenseMetric, p::AbstractVector)
    mul!(out, metric.inverse_mass, p)
    return out
end

# --- kinetic energy 0.5 p' M^{-1} p ----------------------------------------

# Diagonal metric keeps the exact `_kinetic_energy` expression for bitwise match.
kinetic_energy(metric::DiagonalMetric, p::AbstractVector) =
    sum((p .^ 2) .* metric.inverse_mass) / 2

function kinetic_energy(metric::DenseMetric, p::AbstractVector)
    tmp = metric.inverse_mass * p
    return dot(p, tmp) / 2
end

# --- scalar-sampler integration leaves --------------------------------------
# These bridge the metric abstraction into the legacy scalar helpers. The raw
# `Vector{Float64}` methods elsewhere are untouched; these only fire for a
# `MassMetric` argument (i.e. `metric=:dense`).

_sample_momentum(rng::AbstractRNG, metric::MassMetric) =
    sample_momentum!(Vector{Float64}(undef, size(metric.inverse_mass, 1)), metric, rng)

_kinetic_energy(momentum::AbstractVector, metric::MassMetric) = kinetic_energy(metric, momentum)

_hamiltonian(logjoint_value::Float64, momentum::AbstractVector, metric::MassMetric) =
    -logjoint_value + kinetic_energy(metric, momentum)

# Leapfrog drift term `M^{-1} p`. The raw-Vector method returns `imm .* p`, the
# exact legacy expression, so diagonal runs stay bitwise identical; the metric
# method dispatches to `apply_inverse_mass!`.
_mass_drift(inverse_mass_matrix::AbstractVector, p::AbstractVector) = inverse_mass_matrix .* p
_mass_drift(metric::MassMetric, p::AbstractVector) =
    apply_inverse_mass!(similar(p), metric, p)

# --- dense running covariance adaptation ------------------------------------
# Mirrors `RunningVarianceState` (weighted Welford) but accumulates a full M2
# matrix. The clip / weighting / effective-count policy is shared with the
# diagonal path so a dense run tracks the same samples; only the produced metric
# differs (full covariance instead of per-dimension variance).

mutable struct DenseRunningCovarianceState
    mean::Vector{Float64}
    m2::Matrix{Float64}
    clipped_sample::Vector{Float64}
    window_length::Int
    count::Int
    weight_sum::Float64
    weight_square_sum::Float64
end

function _dense_running_covariance_state(
    num_params::Int,
    window_length::Int=_RUNNING_VARIANCE_CLIP_START + 16,
)
    window_length > 0 || throw(ArgumentError("dense running covariance state requires window_length > 0"))
    return DenseRunningCovarianceState(
        zeros(num_params),
        zeros(num_params, num_params),
        zeros(num_params),
        window_length,
        0,
        0.0,
        0.0,
    )
end

_running_variance_effective_count(state::DenseRunningCovarianceState) =
    state.weight_square_sum <= 0 ? 0.0 : state.weight_sum^2 / state.weight_square_sum

# Per-dimension clip identical to `_running_variance_sample!` but reading the
# diagonal of the covariance M2 for the per-dimension variance estimate.
function _dense_running_covariance_sample!(
    state::DenseRunningCovarianceState,
    sample::AbstractVector,
)
    clipped_sample = state.clipped_sample
    if state.count < _RUNNING_VARIANCE_CLIP_START
        copyto!(clipped_sample, sample)
        return clipped_sample
    end

    clip_scale = _running_variance_clip_scale(state.count, state.window_length)
    @inbounds for index in eachindex(clipped_sample, sample, state.mean)
        variance = state.m2[index, index] / max(state.count - 1, 1)
        bound = clip_scale * sqrt(max(variance, _RUNNING_VARIANCE_FLOOR))
        delta = sample[index] - state.mean[index]
        clipped_sample[index] = state.mean[index] + clamp(delta, -bound, bound)
    end
    return clipped_sample
end

function _update_dense_covariance!(
    state::DenseRunningCovarianceState,
    sample::AbstractVector,
    weight::Real,
)
    weight_value = Float64(weight)
    0 <= weight_value <= 1 || throw(ArgumentError("dense covariance weight must lie in [0, 1], got $weight"))
    iszero(weight_value) && return nothing

    update_sample = _dense_running_covariance_sample!(state, sample)
    state.count += 1
    new_weight_sum = state.weight_sum + weight_value
    n = length(state.mean)
    delta = update_sample .- state.mean
    state.mean .+= (weight_value / new_weight_sum) .* delta
    @inbounds for j = 1:n
        delta2j = update_sample[j] - state.mean[j]
        for i = 1:n
            state.m2[i, j] += weight_value * delta[i] * delta2j
        end
    end
    state.weight_sum = new_weight_sum
    state.weight_square_sum += weight_value^2
    return nothing
end

function _update_dense_covariance!(
    state::DenseRunningCovarianceState,
    samples::AbstractMatrix,
    weights::AbstractVector,
)
    size(samples, 2) == length(weights) ||
        throw(DimensionMismatch("expected $(size(samples, 2)) dense covariance weights, got $(length(weights))"))
    for column_index in axes(samples, 2)
        _update_dense_covariance!(state, view(samples, :, column_index), weights[column_index])
    end
    return nothing
end

# Dense analogue of `_inverse_mass_matrix`: estimate the posterior covariance,
# shrink it toward the identity with the same n/(n+5) weight, floor the spectrum
# at `regularization`, and return M^{-1} = Sigma. On diagonal data this reduces
# to `Diagonal(shrink*var + (1-shrink))`, which matches the diagonal path exactly
# (both store the regularized variance as M^{-1}, i.e. the standard windowed /
# Euclidean-HMC choice).
function _dense_inverse_mass_matrix(state::DenseRunningCovarianceState, regularization::Float64)
    n = length(state.mean)
    identity = Matrix{Float64}(I, n, n)
    effective_count = _running_variance_effective_count(state)
    effective_count < 2 && return identity

    variance_denom = state.weight_sum - state.weight_square_sum / state.weight_sum
    variance_denom <= 0 && return identity

    covariance = state.m2 ./ variance_denom
    shrinkage = effective_count / (effective_count + 5.0)
    regularized = shrinkage .* covariance .+ (1 - shrinkage) .* identity
    regularized .= 0.5 .* (regularized .+ transpose(regularized))

    decomposition = eigen(Symmetric(regularized))
    values = max.(decomposition.values, regularization)
    reconstructed = decomposition.vectors * Diagonal(values) * transpose(decomposition.vectors)
    reconstructed .= 0.5 .* (reconstructed .+ transpose(reconstructed))
    return reconstructed
end
