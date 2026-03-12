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

function _batched_random_walk_move!(
    particles::AbstractMatrix,
    logjoint_values::AbstractVector,
    logproposal_values::AbstractVector,
    log_ratio::AbstractVector,
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    move_scale::AbstractVector,
    num_steps::Int,
    rng::AbstractRNG,
)
    num_steps > 0 || return 0.0
    size(particles, 1) == length(proposal_location) == length(proposal_log_scale) == length(move_scale) ||
        throw(DimensionMismatch("expected move-step vectors to match particle rows"))
    size(particles, 2) == length(logjoint_values) == length(logproposal_values) == length(log_ratio) ||
        throw(DimensionMismatch("expected move-step particle metadata to match particle count"))

    proposal_particles = similar(particles)
    proposal_noise = similar(particles)
    current_tempered = Vector{Float64}(undef, size(particles, 2))
    proposal_tempered = similar(current_tempered)
    proposal_logjoint = similar(logjoint_values)
    proposal_logproposal = similar(logproposal_values)
    accepted = 0
    total = size(particles, 2) * num_steps

    for _ in 1:num_steps
        for parameter_index in axes(particles, 1)
            scale = move_scale[parameter_index]
            for particle_index in axes(particles, 2)
                proposal_particles[parameter_index, particle_index] =
                    particles[parameter_index, particle_index] + scale * randn(rng)
            end
        end

        copyto!(proposal_logjoint, batched_logjoint_unconstrained(model, proposal_particles, args, constraints))
        _gaussian_logdensity_from_particles!(
            proposal_logproposal,
            proposal_particles,
            proposal_location,
            proposal_log_scale,
            proposal_noise,
        )
        _tempered_logdensity!(current_tempered, beta, logjoint_values, logproposal_values)
        _tempered_logdensity!(proposal_tempered, beta, proposal_logjoint, proposal_logproposal)

        for particle_index in eachindex(current_tempered, proposal_tempered)
            log_accept_ratio = proposal_tempered[particle_index] - current_tempered[particle_index]
            if isfinite(proposal_logjoint[particle_index]) && log(rand(rng)) < min(0.0, log_accept_ratio)
                copyto!(view(particles, :, particle_index), view(proposal_particles, :, particle_index))
                logjoint_values[particle_index] = proposal_logjoint[particle_index]
                logproposal_values[particle_index] = proposal_logproposal[particle_index]
                log_ratio[particle_index] = proposal_logjoint[particle_index] - proposal_logproposal[particle_index]
                accepted += 1
            end
        end
    end

    return accepted / total
end

function _batched_hmc_move!(
    particles::AbstractMatrix,
    logjoint_values::AbstractVector,
    logproposal_values::AbstractVector,
    log_ratio::AbstractVector,
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    num_leapfrog_steps::Int,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    num_leapfrog_steps > 0 || throw(ArgumentError("tempered HMC move requires num_leapfrog_steps > 0"))
    step_size > 0 || throw(ArgumentError("tempered HMC move requires step_size > 0"))
    size(particles, 1) == length(inverse_mass_matrix) ||
        throw(DimensionMismatch("expected inverse mass matrix of length $(size(particles, 1)), got $(length(inverse_mass_matrix))"))

    cache = BatchedLogjointGradientCache(model, particles, args, constraints)
    num_particles = size(particles, 2)
    parameter_total = size(particles, 1)
    momentum = Matrix{Float64}(undef, parameter_total, num_particles)
    proposal_particles = Matrix{Float64}(undef, parameter_total, num_particles)
    proposal_momentum = similar(momentum)
    current_logjoint_gradient = Matrix{Float64}(undef, parameter_total, num_particles)
    current_logproposal_gradient = similar(current_logjoint_gradient)
    current_tempered_gradient = similar(current_logjoint_gradient)
    proposal_logjoint_values = Vector{Float64}(undef, num_particles)
    proposal_logproposal_values = similar(proposal_logjoint_values)
    proposal_tempered_values = similar(proposal_logjoint_values)
    proposal_logjoint_gradient = similar(current_logjoint_gradient)
    proposal_logproposal_gradient = similar(current_logjoint_gradient)
    proposal_tempered_gradient = similar(current_logjoint_gradient)
    proposal_noise = similar(current_logjoint_gradient)
    current_tempered_values = Vector{Float64}(undef, num_particles)
    current_hamiltonian = Vector{Float64}(undef, num_particles)
    proposal_hamiltonian = similar(current_hamiltonian)
    valid = trues(num_particles)
    accepted = 0

    _batched_tempered_target!(
        current_tempered_values,
        current_tempered_gradient,
        logjoint_values,
        current_logjoint_gradient,
        logproposal_values,
        current_logproposal_gradient,
        proposal_noise,
        model,
        cache,
        particles,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
    )

    sqrt_inverse_mass_matrix = sqrt.(Float64.(inverse_mass_matrix))
    _sample_batched_momentum!(momentum, rng, sqrt_inverse_mass_matrix)
    copyto!(proposal_particles, particles)
    copyto!(proposal_momentum, momentum)
    fill!(valid, true)

    for particle_index in 1:num_particles
        for parameter_index in 1:parameter_total
            proposal_momentum[parameter_index, particle_index] +=
                (step_size / 2) * current_tempered_gradient[parameter_index, particle_index]
        end
    end

    for leapfrog_step in 1:num_leapfrog_steps
        for particle_index in 1:num_particles
            valid[particle_index] || continue
            for parameter_index in 1:parameter_total
                proposal_particles[parameter_index, particle_index] +=
                    step_size * inverse_mass_matrix[parameter_index] * proposal_momentum[parameter_index, particle_index]
            end
        end

        _batched_tempered_target!(
            proposal_tempered_values,
            proposal_tempered_gradient,
            proposal_logjoint_values,
            proposal_logjoint_gradient,
            proposal_logproposal_values,
            proposal_logproposal_gradient,
            proposal_noise,
            model,
            cache,
            proposal_particles,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
        )

        for particle_index in 1:num_particles
            valid[particle_index] || continue
            if !isfinite(proposal_tempered_values[particle_index]) ||
               !isfinite(proposal_logjoint_values[particle_index]) ||
               !all(isfinite, view(proposal_tempered_gradient, :, particle_index))
                valid[particle_index] = false
                continue
            end

            if leapfrog_step < num_leapfrog_steps
                for parameter_index in 1:parameter_total
                    proposal_momentum[parameter_index, particle_index] +=
                        step_size * proposal_tempered_gradient[parameter_index, particle_index]
                end
            end
        end
    end

    for particle_index in 1:num_particles
        valid[particle_index] || continue
        for parameter_index in 1:parameter_total
            proposal_momentum[parameter_index, particle_index] +=
                (step_size / 2) * proposal_tempered_gradient[parameter_index, particle_index]
            proposal_momentum[parameter_index, particle_index] *= -1
        end
    end

    _batched_hamiltonian!(current_hamiltonian, current_tempered_values, momentum, inverse_mass_matrix)
    _batched_hamiltonian!(proposal_hamiltonian, proposal_tempered_values, proposal_momentum, inverse_mass_matrix)

    for particle_index in 1:num_particles
        valid[particle_index] || continue
        log_accept_ratio = current_hamiltonian[particle_index] - proposal_hamiltonian[particle_index]
        if log(rand(rng)) < min(0.0, log_accept_ratio)
            copyto!(view(particles, :, particle_index), view(proposal_particles, :, particle_index))
            logjoint_values[particle_index] = proposal_logjoint_values[particle_index]
            logproposal_values[particle_index] = proposal_logproposal_values[particle_index]
            log_ratio[particle_index] = proposal_logjoint_values[particle_index] - proposal_logproposal_values[particle_index]
            accepted += 1
        end
    end

    return accepted / num_particles
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
    move_kernel::Symbol=:random_walk,
    move_steps::Int=0,
    move_scale=0.1,
    move_step_size::Real=0.05,
    move_num_leapfrog_steps::Int=4,
    move_inverse_mass_matrix=1.0,
    rng::AbstractRNG=Random.default_rng(),
)
    layout = parameterlayout(model)
    parameter_total = parametercount(layout)
    parameter_total > 0 || throw(ArgumentError("batched_smc requires at least one parameterized latent choice"))
    num_particles > 0 || throw(ArgumentError("batched_smc requires num_particles > 0"))
    0 < target_ess_ratio <= 1 || throw(ArgumentError("batched_smc requires 0 < target_ess_ratio <= 1"))
    max_stages > 0 || throw(ArgumentError("batched_smc requires max_stages > 0"))
    move_steps >= 0 || throw(ArgumentError("batched_smc requires move_steps >= 0"))
    move_step_size > 0 || throw(ArgumentError("batched_smc requires move_step_size > 0"))
    move_num_leapfrog_steps > 0 || throw(ArgumentError("batched_smc requires move_num_leapfrog_steps > 0"))
    move_kernel in (:random_walk, :hmc) ||
        throw(ArgumentError("batched_smc move_kernel must be :random_walk or :hmc"))

    location = _resolve_unconstrained_point(model, args, constraints, proposal_loc, rng, "proposal_loc")
    log_scale = _resolve_scale_vector("proposal_log_scale", proposal_log_scale, parameter_total)
    move_scale_vector = _resolve_scale_vector("move_scale", move_scale, parameter_total)
    move_inverse_mass = _resolve_scale_vector("move_inverse_mass_matrix", move_inverse_mass_matrix, parameter_total)
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
        stage_move_steps = 0
        move_acceptance_rate = 0.0

        if resampled
            ancestors = _systematic_resample_indices(normalized_weights, num_particles, rng)
            push!(ancestor_history, ancestors)
            particles = _resampled_particle_matrix(particles, ancestors)
            logproposal_values = _resampled_particle_vector(logproposal_values, ancestors)
            logjoint_values = _resampled_particle_vector(logjoint_values, ancestors)
            log_ratio = _resampled_particle_vector(log_ratio, ancestors)
            fill!(normalized_weights, 1.0 / num_particles)
            fill!(logweights, 0.0)
            effective_sample_size = Float64(num_particles)
            if move_steps > 0
                if move_kernel === :random_walk
                    move_acceptance_rate = _batched_random_walk_move!(
                        particles,
                        logjoint_values,
                        logproposal_values,
                        log_ratio,
                        model,
                        args,
                        constraints,
                        location,
                        log_scale,
                        beta_next,
                        move_scale_vector,
                        move_steps,
                        rng,
                    )
                else
                    acceptance_sum = 0.0
                    for _ in 1:move_steps
                        acceptance_sum += _batched_hmc_move!(
                            particles,
                            logjoint_values,
                            logproposal_values,
                            log_ratio,
                            model,
                            args,
                            constraints,
                            location,
                            log_scale,
                            beta_next,
                            move_step_size,
                            move_num_leapfrog_steps,
                            move_inverse_mass,
                            rng,
                        )
                    end
                    move_acceptance_rate = acceptance_sum / move_steps
                end
                stage_move_steps = move_steps
            end
        end

        push!(
            stages,
            SMCStageSummary(
                stage_index,
                beta,
                beta_next,
                effective_sample_size,
                log_increment,
                resampled,
                move_kernel,
                stage_move_steps,
                move_acceptance_rate,
            ),
        )

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
