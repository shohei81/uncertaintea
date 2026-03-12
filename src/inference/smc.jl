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

function batched_smc(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_particles::Int,
    proposal_loc=nothing,
    proposal_log_scale=0.0,
    target_ess_ratio::Real=0.8,
    max_stages::Int=32,
    resample_final::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
)
    layout = parameterlayout(model)
    parameter_total = parametercount(layout)
    parameter_total > 0 || throw(ArgumentError("batched_smc requires at least one parameterized latent choice"))
    num_particles > 0 || throw(ArgumentError("batched_smc requires num_particles > 0"))
    0 < target_ess_ratio <= 1 || throw(ArgumentError("batched_smc requires 0 < target_ess_ratio <= 1"))
    max_stages > 0 || throw(ArgumentError("batched_smc requires max_stages > 0"))

    location = _resolve_unconstrained_point(model, args, constraints, proposal_loc, rng, "proposal_loc")
    log_scale = _resolve_scale_vector("proposal_log_scale", proposal_log_scale, parameter_total)
    particles = Matrix{Float64}(undef, parameter_total, num_particles)
    noise = similar(particles)
    logproposal_values = Vector{Float64}(undef, num_particles)
    logjoint_values = Vector{Float64}(undef, num_particles)
    logweights = Vector{Float64}(undef, num_particles)
    _draw_gaussian_particles!(particles, noise, location, log_scale, rng)
    _gaussian_logdensity!(logproposal_values, noise, log_scale)
    copyto!(logjoint_values, batched_logjoint_unconstrained(model, particles, args, constraints))
    all(isfinite, logjoint_values) ||
        throw(ArgumentError("batched_smc encountered a non-finite unconstrained logjoint value"))

    log_ratio = Vector{Float64}(undef, num_particles)
    for index in eachindex(log_ratio)
        log_ratio[index] = logjoint_values[index] - logproposal_values[index]
    end

    normalized_weights = fill(1.0 / num_particles, num_particles)
    effective_sample_size = Float64(num_particles)
    min_effective_sample_size = Float64(target_ess_ratio) * num_particles
    beta = 0.0
    log_evidence_estimate = 0.0
    stages = SMCStageSummary[]
    ancestor_history = Vector{Vector{Int}}()

    for stage_index in 1:max_stages
        beta_next = _adaptive_tempering_beta(log_ratio, beta, min_effective_sample_size, logweights)
        _incremental_tempering_weights!(logweights, log_ratio, beta, beta_next)
        normalized_weights, log_weight_total = _normalized_logweights(logweights)
        effective_sample_size = _effective_sample_size(normalized_weights)
        log_increment = log_weight_total - log(num_particles)
        log_evidence_estimate += log_increment
        resampled = beta_next < 1.0 || resample_final

        push!(
            stages,
            SMCStageSummary(
                stage_index,
                beta,
                beta_next,
                effective_sample_size,
                log_increment,
                resampled,
            ),
        )

        if resampled
            ancestors = _systematic_resample_indices(normalized_weights, num_particles, rng)
            push!(ancestor_history, ancestors)
            particles = _resampled_particle_matrix(particles, ancestors)
            logproposal_values = _resampled_particle_matrix(reshape(logproposal_values, 1, :), ancestors)[1, :]
            logjoint_values = _resampled_particle_matrix(reshape(logjoint_values, 1, :), ancestors)[1, :]
            log_ratio = _resampled_particle_matrix(reshape(log_ratio, 1, :), ancestors)[1, :]
            fill!(normalized_weights, 1.0 / num_particles)
            fill!(logweights, 0.0)
            effective_sample_size = Float64(num_particles)
        end

        beta = beta_next
        beta >= 1.0 - 1e-12 && break
    end

    beta >= 1.0 - 1e-12 || throw(ArgumentError("batched_smc reached max_stages=$max_stages before the tempering schedule reached 1.0"))

    if !isempty(stages) && last(stages).resampled
        logweights .= 0.0
        normalized_weights .= 1.0 / num_particles
        effective_sample_size = Float64(num_particles)
    end

    constrained_particles = Matrix{Float64}(undef, parametervaluecount(layout), num_particles)
    _batched_transform_to_constrained!(constrained_particles, model, particles)
    importance = ImportanceSamplingResult(
        model,
        args,
        constraints,
        location,
        log_scale,
        particles,
        constrained_particles,
        logjoint_values,
        copy(logweights),
        copy(normalized_weights),
        log_evidence_estimate,
        effective_sample_size,
        _batched_evaluation_backend(model),
    )
    return SMCResult(importance, stages, ancestor_history)
end
