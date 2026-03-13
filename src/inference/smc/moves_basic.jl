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
