struct ImportanceSamplingResult
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    proposal_loc::Vector{Float64}
    proposal_log_scale::Vector{Float64}
    unconstrained_particles::Matrix{Float64}
    constrained_particles::Matrix{Float64}
    logjoint_values::Vector{Float64}
    logweights::Vector{Float64}
    normalized_weights::Vector{Float64}
    log_evidence_estimate::Float64
    effective_sample_size::Float64
    evaluation_backend::Symbol
end

struct SIRResult
    importance::ImportanceSamplingResult
    ancestors::Vector{Int}
    unconstrained_samples::Matrix{Float64}
    constrained_samples::Matrix{Float64}
end

struct SMCStageSummary
    stage_index::Int
    beta_start::Float64
    beta_end::Float64
    effective_sample_size::Float64
    log_normalizer_increment::Float64
    resampled::Bool
    move_kernel::Symbol
    move_steps::Int
    move_acceptance_rate::Float64
end

struct SMCResult
    importance::ImportanceSamplingResult
    stages::Vector{SMCStageSummary}
    ancestor_history::Vector{Vector{Int}}
end

function ess(result::ImportanceSamplingResult)
    return result.effective_sample_size
end

function ess(result::SMCResult)
    return ess(result.importance)
end

function numsamples(result::SIRResult)
    return size(result.unconstrained_samples, 2)
end

function numsamples(result::SMCResult)
    return size(result.importance.unconstrained_particles, 2)
end

function numstages(result::SMCResult)
    return length(result.stages)
end

function _batched_evaluation_backend(model::TeaModel)
    return isnothing(_backend_execution_plan(model)) ? :compiled_cpu : :backend_native
end

function _logsumexp(values::AbstractVector)
    isempty(values) && throw(ArgumentError("logsumexp requires a non-empty vector"))
    max_value = maximum(values)
    total = 0.0
    for value in values
        total += exp(value - max_value)
    end
    return max_value + log(total)
end

function _normalized_logweights(logweights::AbstractVector)
    log_total = _logsumexp(logweights)
    normalized = Vector{Float64}(undef, length(logweights))
    for index in eachindex(logweights)
        normalized[index] = exp(logweights[index] - log_total)
    end
    return normalized, log_total
end

function _effective_sample_size(normalized_weights::AbstractVector)
    total = 0.0
    for weight in normalized_weights
        total += weight * weight
    end
    return inv(total)
end

function _adaptive_tempering_beta(
    log_ratio::AbstractVector,
    beta_start::Float64,
    min_effective_sample_size::Float64,
    scratch_logweights::AbstractVector,
)
    beta_start < 1.0 || return 1.0
    length(log_ratio) == length(scratch_logweights) ||
        throw(DimensionMismatch("expected tempering scratch of length $(length(log_ratio)), got $(length(scratch_logweights))"))

    function effective_sample_size_at(beta_end::Float64)
        delta = beta_end - beta_start
        for index in eachindex(log_ratio, scratch_logweights)
            scratch_logweights[index] = delta * log_ratio[index]
        end
        normalized_weights, _ = _normalized_logweights(scratch_logweights)
        return _effective_sample_size(normalized_weights)
    end

    full_ess = effective_sample_size_at(1.0)
    full_ess >= min_effective_sample_size && return 1.0

    lower = beta_start
    upper = 1.0
    for _ in 1:32
        midpoint = (lower + upper) / 2
        midpoint_ess = effective_sample_size_at(midpoint)
        if midpoint_ess >= min_effective_sample_size
            lower = midpoint
        else
            upper = midpoint
        end
    end

    return max(lower, min(1.0, beta_start + 1e-6))
end

function _incremental_tempering_weights!(
    destination::AbstractVector,
    log_ratio::AbstractVector,
    beta_start::Float64,
    beta_end::Float64,
)
    length(destination) == length(log_ratio) ||
        throw(DimensionMismatch("expected tempering destination of length $(length(log_ratio)), got $(length(destination))"))

    delta = beta_end - beta_start
    for index in eachindex(destination, log_ratio)
        destination[index] = delta * log_ratio[index]
    end
    return destination
end

function _systematic_resample_indices(
    normalized_weights::AbstractVector,
    num_samples::Int,
    rng::AbstractRNG,
)
    num_samples > 0 || throw(ArgumentError("systematic resampling requires num_samples > 0"))
    isempty(normalized_weights) && throw(ArgumentError("systematic resampling requires non-empty weights"))

    indices = Vector{Int}(undef, num_samples)
    step = 1.0 / num_samples
    threshold = rand(rng) * step
    cumulative = normalized_weights[1]
    source_index = 1
    for sample_index in 1:num_samples
        position = threshold + (sample_index - 1) * step
        while position > cumulative && source_index < length(normalized_weights)
            source_index += 1
            cumulative += normalized_weights[source_index]
        end
        indices[sample_index] = source_index
    end
    return indices
end

function _resampled_particle_matrix(
    particles::AbstractMatrix,
    ancestors::AbstractVector{Int},
)
    samples = Matrix{Float64}(undef, size(particles, 1), length(ancestors))
    for (sample_index, ancestor_index) in enumerate(ancestors)
        copyto!(view(samples, :, sample_index), view(particles, :, ancestor_index))
    end
    return samples
end

function _resampled_particle_vector(
    values::AbstractVector,
    ancestors::AbstractVector{Int},
)
    samples = Vector{Float64}(undef, length(ancestors))
    for (sample_index, ancestor_index) in enumerate(ancestors)
        samples[sample_index] = Float64(values[ancestor_index])
    end
    return samples
end

function _gaussian_logdensity_from_particles!(
    destination::AbstractVector,
    particles::AbstractMatrix,
    location::AbstractVector,
    log_scale::AbstractVector,
    scratch_noise::AbstractMatrix,
)
    size(particles) == size(scratch_noise) ||
        throw(DimensionMismatch("expected particle and Gaussian scratch matrices to have matching shapes"))
    size(particles, 1) == length(location) == length(log_scale) ||
        throw(
            DimensionMismatch(
                "expected Gaussian particle matrices with $(length(location)) rows, got $(size(particles, 1))",
            ),
        )
    size(particles, 2) == length(destination) ||
        throw(DimensionMismatch("expected Gaussian log-density destination of length $(size(particles, 2)), got $(length(destination))"))

    for parameter_index in eachindex(location, log_scale)
        scale = exp(log_scale[parameter_index])
        for particle_index in axes(particles, 2)
            scratch_noise[parameter_index, particle_index] =
                (particles[parameter_index, particle_index] - location[parameter_index]) / scale
        end
    end
    return _gaussian_logdensity!(destination, scratch_noise, log_scale)
end

function _gaussian_gradient_from_particles!(
    destination::AbstractMatrix,
    particles::AbstractMatrix,
    location::AbstractVector,
    log_scale::AbstractVector,
)
    size(destination) == size(particles) ||
        throw(DimensionMismatch("expected Gaussian gradient destination and particle matrices to have matching shapes"))
    size(particles, 1) == length(location) == length(log_scale) ||
        throw(
            DimensionMismatch(
                "expected Gaussian particle matrices with $(length(location)) rows, got $(size(particles, 1))",
            ),
        )

    for parameter_index in eachindex(location, log_scale)
        inverse_variance = exp(-2.0 * log_scale[parameter_index])
        for particle_index in axes(particles, 2)
            destination[parameter_index, particle_index] =
                -(particles[parameter_index, particle_index] - location[parameter_index]) * inverse_variance
        end
    end
    return destination
end

function _tempered_logdensity!(
    destination::AbstractVector,
    beta::Float64,
    logjoint_values::AbstractVector,
    logproposal_values::AbstractVector,
)
    length(destination) == length(logjoint_values) == length(logproposal_values) ||
        throw(DimensionMismatch("expected tempered logdensity vectors of matching length"))
    for index in eachindex(destination, logjoint_values, logproposal_values)
        destination[index] =
            beta * logjoint_values[index] + (1.0 - beta) * logproposal_values[index]
    end
    return destination
end

function _tempered_gradient!(
    destination::AbstractMatrix,
    beta::Float64,
    logjoint_gradient::AbstractMatrix,
    logproposal_gradient::AbstractMatrix,
)
    size(destination) == size(logjoint_gradient) == size(logproposal_gradient) ||
        throw(DimensionMismatch("expected tempered gradient matrices of matching shape"))
    one_minus_beta = 1.0 - beta
    for column_index in axes(destination, 2), row_index in axes(destination, 1)
        destination[row_index, column_index] =
            beta * logjoint_gradient[row_index, column_index] +
            one_minus_beta * logproposal_gradient[row_index, column_index]
    end
    return destination
end

function _batched_tempered_target!(
    tempered_values::AbstractVector,
    tempered_gradient::AbstractMatrix,
    logjoint_values::AbstractVector,
    logjoint_gradient::AbstractMatrix,
    logproposal_values::AbstractVector,
    logproposal_gradient::AbstractMatrix,
    logproposal_noise::AbstractMatrix,
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    particles::AbstractMatrix,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
)
    _batched_logjoint_and_gradient_unconstrained!(logjoint_values, cache, particles)
    _gaussian_logdensity_from_particles!(
        logproposal_values,
        particles,
        proposal_location,
        proposal_log_scale,
        logproposal_noise,
    )
    _gaussian_gradient_from_particles!(
        logproposal_gradient,
        particles,
        proposal_location,
        proposal_log_scale,
    )
    _tempered_logdensity!(tempered_values, beta, logjoint_values, logproposal_values)
    _tempered_gradient!(tempered_gradient, beta, logjoint_gradient, logproposal_gradient)
    return tempered_values, tempered_gradient
end

