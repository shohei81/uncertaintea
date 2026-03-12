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

function ess(result::ImportanceSamplingResult)
    return result.effective_sample_size
end

function numsamples(result::SIRResult)
    return size(result.unconstrained_samples, 2)
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

function batched_importance_sampling(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_particles::Int,
    proposal_loc=nothing,
    proposal_log_scale=0.0,
    rng::AbstractRNG=Random.default_rng(),
)
    layout = parameterlayout(model)
    parameter_total = parametercount(layout)
    parameter_total > 0 || throw(ArgumentError("batched_importance_sampling requires at least one parameterized latent choice"))
    num_particles > 0 || throw(ArgumentError("batched_importance_sampling requires num_particles > 0"))

    location = _resolve_unconstrained_point(model, args, constraints, proposal_loc, rng, "proposal_loc")
    log_scale = _resolve_scale_vector("proposal_log_scale", proposal_log_scale, parameter_total)
    particles = Matrix{Float64}(undef, parameter_total, num_particles)
    noise = similar(particles)
    log_proposal = Vector{Float64}(undef, num_particles)
    _draw_gaussian_particles!(particles, noise, location, log_scale, rng)
    _gaussian_logdensity!(log_proposal, noise, log_scale)

    logjoint_values = batched_logjoint_unconstrained(model, particles, args, constraints)
    all(isfinite, logjoint_values) ||
        throw(ArgumentError("batched_importance_sampling encountered a non-finite unconstrained logjoint value"))

    logweights = Vector{Float64}(undef, num_particles)
    for particle_index in eachindex(logweights)
        logweights[particle_index] = logjoint_values[particle_index] - log_proposal[particle_index]
    end

    normalized_weights, log_weight_total = _normalized_logweights(logweights)
    effective_sample_size = 0.0
    for weight in normalized_weights
        effective_sample_size += weight * weight
    end
    effective_sample_size = inv(effective_sample_size)

    constrained_particles = Matrix{Float64}(undef, parametervaluecount(layout), num_particles)
    _batched_transform_to_constrained!(constrained_particles, model, particles)

    return ImportanceSamplingResult(
        model,
        args,
        constraints,
        location,
        log_scale,
        particles,
        constrained_particles,
        logjoint_values,
        logweights,
        normalized_weights,
        log_weight_total - log(num_particles),
        effective_sample_size,
        _batched_evaluation_backend(model),
    )
end

function batched_sir(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_particles::Int,
    num_samples::Int=num_particles,
    proposal_loc=nothing,
    proposal_log_scale=0.0,
    rng::AbstractRNG=Random.default_rng(),
)
    importance = batched_importance_sampling(
        model,
        args,
        constraints;
        num_particles=num_particles,
        proposal_loc=proposal_loc,
        proposal_log_scale=proposal_log_scale,
        rng=rng,
    )
    ancestors = _systematic_resample_indices(importance.normalized_weights, num_samples, rng)
    unconstrained_samples = _resampled_particle_matrix(importance.unconstrained_particles, ancestors)
    constrained_samples = _resampled_particle_matrix(importance.constrained_particles, ancestors)
    return SIRResult(importance, ancestors, unconstrained_samples, constrained_samples)
end

function batched_smc(args...; kwargs...)
    return batched_sir(args...; kwargs...)
end
