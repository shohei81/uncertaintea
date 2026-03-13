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

function _single_gaussian_logdensity(
    position::AbstractVector,
    location::AbstractVector,
    log_scale::AbstractVector,
)
    length(position) == length(location) == length(log_scale) ||
        throw(DimensionMismatch("expected Gaussian single-particle vectors of matching length"))
    value = -0.5 * length(position) * log(2.0 * pi)
    for parameter_index in eachindex(position, location, log_scale)
        scale = exp(log_scale[parameter_index])
        noise = (position[parameter_index] - location[parameter_index]) / scale
        value -= log_scale[parameter_index] + 0.5 * noise * noise
    end
    return value
end

function _single_gaussian_gradient!(
    destination::AbstractVector,
    position::AbstractVector,
    location::AbstractVector,
    log_scale::AbstractVector,
)
    length(destination) == length(position) == length(location) == length(log_scale) ||
        throw(DimensionMismatch("expected Gaussian single-particle vectors of matching length"))
    for parameter_index in eachindex(destination, position, location, log_scale)
        destination[parameter_index] =
            -(position[parameter_index] - location[parameter_index]) * exp(-2.0 * log_scale[parameter_index])
    end
    return destination
end

function _sample_batched_nuts_directions!(
    destination::AbstractVector{Int},
    rng::AbstractRNG,
)
    for index in eachindex(destination)
        destination[index] = _sample_nuts_direction(rng)
    end
    return destination
end

function _tempered_target_value_and_gradient!(
    gradient_destination::AbstractVector,
    proposal_gradient::AbstractVector,
    model::TeaModel,
    model_gradient_cache::LogjointGradientCache,
    position::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
)
    model_logjoint = logjoint_unconstrained(model, position, args, constraints)
    model_gradient = _logjoint_gradient!(model_gradient_cache, position)
    proposal_logdensity = _single_gaussian_logdensity(position, proposal_location, proposal_log_scale)
    _single_gaussian_gradient!(proposal_gradient, position, proposal_location, proposal_log_scale)
    one_minus_beta = 1.0 - beta
    for parameter_index in eachindex(gradient_destination, model_gradient, proposal_gradient)
        gradient_destination[parameter_index] =
            beta * model_gradient[parameter_index] + one_minus_beta * proposal_gradient[parameter_index]
    end
    tempered_value = beta * model_logjoint + one_minus_beta * proposal_logdensity
    return tempered_value, model_logjoint, proposal_logdensity
end

function _tempered_nuts_leapfrog_step!(
    destination::NUTSState,
    state::NUTSState,
    gradient_buffer::AbstractVector,
    proposal_gradient::AbstractVector,
    model::TeaModel,
    model_gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
)
    q = destination.position
    p = destination.momentum
    gradient = destination.gradient
    copyto!(q, state.position)
    copyto!(p, state.momentum)
    p .+= (step_size / 2) .* state.gradient
    q .+= step_size .* (inverse_mass_matrix .* p)
    proposed_logjoint, _, _ = _tempered_target_value_and_gradient!(
        gradient_buffer,
        proposal_gradient,
        model,
        model_gradient_cache,
        q,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
    )
    isfinite(proposed_logjoint) || return false
    all(isfinite, gradient_buffer) || return false
    copyto!(gradient, gradient_buffer)
    p .+= (step_size / 2) .* gradient
    destination.logjoint = proposed_logjoint
    return true
end

function _build_tempered_nuts_subtree(
    subtree_workspace::NUTSSubtreeWorkspace,
    gradient_buffer::AbstractVector,
    proposal_gradient::AbstractVector,
    model::TeaModel,
    start_state::NUTSState,
    model_gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    direction::Int,
    depth::Int,
    initial_hamiltonian::Float64,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    current = subtree_workspace.current
    next = subtree_workspace.next
    left = subtree_workspace.left
    right = subtree_workspace.right
    proposal = subtree_workspace.proposal
    summary = subtree_workspace.summary
    _copyto_nuts_state!(current, start_state)
    _copyto_nuts_state!(left, start_state)
    _copyto_nuts_state!(right, start_state)
    _copyto_nuts_state!(proposal, start_state)
    _reset_nuts_subtree_summary!(summary)

    for _ in 1:(1 << depth)
        if !_tempered_nuts_leapfrog_step!(
            next,
            current,
            gradient_buffer,
            proposal_gradient,
            model,
            model_gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
            direction * step_size,
        )
            summary.divergent = true
            break
        end
        current, next = next, current
        summary.integration_steps += 1
        if direction < 0
            _copyto_nuts_state!(left, current)
        else
            _copyto_nuts_state!(right, current)
        end
        summary.proposed_energy = _hamiltonian(current.logjoint, current.momentum, inverse_mass_matrix)
        summary.delta_energy = summary.proposed_energy - initial_hamiltonian
        if !isfinite(summary.delta_energy) || summary.delta_energy > max_delta_energy
            summary.divergent = true
            break
        end
        summary.accept_prob = min(1.0, exp(min(0.0, -summary.delta_energy)))
        summary.accept_stat_sum += summary.accept_prob
        summary.accept_stat_count += 1
        summary.candidate_log_weight = -summary.proposed_energy
        summary.combined_log_weight = _logaddexp(summary.log_weight, summary.candidate_log_weight)
        if !isfinite(summary.log_weight) || log(rand(rng)) < summary.candidate_log_weight - summary.combined_log_weight
            _copyto_nuts_state!(proposal, current)
            summary.proposal_energy = summary.proposed_energy
            summary.proposal_energy_error = summary.delta_energy
        end
        summary.log_weight = summary.combined_log_weight
        summary.turning = _is_turning(left.position, right.position, left.momentum, right.momentum)
        summary.turning && break
    end

    return _nuts_subtree_summary(summary)
end

function _continue_tempered_nuts_proposal!(
    continuation::NUTSContinuationState,
    tree_workspace::NUTSSubtreeWorkspace,
    gradient_buffer::AbstractVector,
    proposal_gradient::AbstractVector,
    model::TeaModel,
    model_gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    initial_hamiltonian::Float64,
    rng::AbstractRNG,
)
    while _nuts_continuation_active(
        continuation.tree_depth,
        max_tree_depth,
        continuation.divergent,
        continuation.turning,
    )
        direction = _sample_nuts_direction(rng)
        subtree = _build_tempered_nuts_subtree(
            tree_workspace,
            gradient_buffer,
            proposal_gradient,
            model,
            _nuts_subtree_start_state(continuation, direction),
            model_gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
            step_size,
            direction,
            continuation.tree_depth,
            initial_hamiltonian,
            max_delta_energy,
            rng,
        )
        continuation.tree_depth += 1
        if subtree.integration_steps == 0
            continuation.divergent = subtree.divergent
            break
        end
        _copy_nuts_continuation_frontier_from_tree!(continuation, tree_workspace, direction)
        combined_log_weight = continuation.log_weight
        if isfinite(tree_workspace.summary.log_weight)
            combined_log_weight = _logaddexp(continuation.log_weight, tree_workspace.summary.log_weight)
            if log(rand(rng)) < tree_workspace.summary.log_weight - combined_log_weight
                _copy_nuts_continuation_proposal_from_tree!(continuation, tree_workspace)
            end
        end
        _merge_nuts_subtree_summary!(continuation, tree_workspace, combined_log_weight)
        _merge_nuts_continuation_turning!(continuation, tree_workspace.summary.turning)
    end
    return continuation
end

function _tempered_nuts_proposal(
    model::TeaModel,
    position::AbstractVector{Float64},
    current_tempered_logjoint::Float64,
    current_tempered_gradient::AbstractVector{Float64},
    model_gradient_cache::LogjointGradientCache,
    proposal_gradient::AbstractVector,
    gradient_buffer::AbstractVector,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    rng::AbstractRNG,
)
    tree_workspace = NUTSSubtreeWorkspace(length(position))
    continuation = NUTSContinuationState(length(position))
    initial_momentum = _sample_momentum(rng, inverse_mass_matrix)
    initial_state = _load_nuts_state!(tree_workspace.current, position, initial_momentum, current_tempered_logjoint, current_tempered_gradient)
    initial_hamiltonian = _hamiltonian(initial_state.logjoint, initial_state.momentum, inverse_mass_matrix)
    direction = _sample_nuts_direction(rng)
    first_step_valid = _tempered_nuts_leapfrog_step!(
        tree_workspace.next,
        initial_state,
        gradient_buffer,
        proposal_gradient,
        model,
        model_gradient_cache,
        inverse_mass_matrix,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
        direction * step_size,
    )
    _initialize_nuts_first_trajectory!(
        continuation,
        tree_workspace,
        first_step_valid,
        direction,
        initial_hamiltonian,
        inverse_mass_matrix,
        max_delta_energy,
        rng,
    )

    _continue_tempered_nuts_proposal!(
        continuation,
        tree_workspace,
        gradient_buffer,
        proposal_gradient,
        model,
        model_gradient_cache,
        inverse_mass_matrix,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
        step_size,
        max_tree_depth,
        max_delta_energy,
        initial_hamiltonian,
        rng,
    )

    accept_stat, proposed_energy, energy_error, moved = _nuts_proposal_summary(continuation, position)
    return continuation.proposal, accept_stat, continuation.tree_depth, continuation.integration_steps, proposed_energy, energy_error, continuation.divergent, moved
end

function _batched_nuts_move!(
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
    max_tree_depth::Int,
    max_delta_energy::Float64,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    num_particles = size(particles, 2)
    parameter_total = size(particles, 1)
    length(inverse_mass_matrix) == parameter_total ||
        throw(DimensionMismatch("expected inverse mass matrix of length $parameter_total, got $(length(inverse_mass_matrix))"))
    inverse_mass = Float64[inverse_mass_matrix[index] for index in eachindex(inverse_mass_matrix)]
    cache = BatchedLogjointGradientCache(model, particles, args, constraints)
    momentum = Matrix{Float64}(undef, parameter_total, num_particles)
    proposal_particles = Matrix{Float64}(undef, parameter_total, num_particles)
    proposal_momentum = similar(momentum)
    current_logjoint_gradient = Matrix{Float64}(undef, parameter_total, num_particles)
    current_logproposal_gradient = similar(current_logjoint_gradient)
    current_tempered_gradient = similar(current_logjoint_gradient)
    current_tempered_values = Vector{Float64}(undef, num_particles)
    proposal_logjoint_values = Vector{Float64}(undef, num_particles)
    proposal_logproposal_values = similar(proposal_logjoint_values)
    proposal_tempered_values = similar(proposal_logjoint_values)
    proposal_logjoint_gradient = similar(current_logjoint_gradient)
    proposal_logproposal_gradient = similar(current_logjoint_gradient)
    proposal_tempered_gradient = similar(current_logjoint_gradient)
    proposal_noise = Matrix{Float64}(undef, parameter_total, num_particles)
    current_hamiltonian = Vector{Float64}(undef, num_particles)
    directions = Vector{Int}(undef, num_particles)
    valid = trues(num_particles)
    proposal_gradient_buffer = Matrix{Float64}(undef, parameter_total, num_particles)
    column_gradient_buffer = Matrix{Float64}(undef, parameter_total, num_particles)
    column_gradient_caches = _batched_nuts_column_gradient_caches(
        model,
        particles,
        args,
        constraints,
        column_gradient_buffer,
    )
    total_accept_stat = 0.0

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
    _sample_batched_momentum!(momentum, rng, sqrt.(inverse_mass))
    _sample_batched_nuts_directions!(directions, rng)
    copyto!(proposal_particles, particles)
    copyto!(proposal_momentum, momentum)
    fill!(valid, true)

    for particle_index in axes(particles, 2), parameter_index in axes(particles, 1)
        signed_step_size = directions[particle_index] * step_size
        proposal_momentum[parameter_index, particle_index] +=
            (signed_step_size / 2) * current_tempered_gradient[parameter_index, particle_index]
        proposal_particles[parameter_index, particle_index] +=
            signed_step_size * inverse_mass[parameter_index] * proposal_momentum[parameter_index, particle_index]
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
        if !isfinite(proposal_tempered_values[particle_index]) ||
           !isfinite(proposal_logjoint_values[particle_index]) ||
           !all(isfinite, view(proposal_tempered_gradient, :, particle_index))
            valid[particle_index] = false
            continue
        end
        signed_step_size = directions[particle_index] * step_size
        for parameter_index in 1:parameter_total
            proposal_momentum[parameter_index, particle_index] +=
                (signed_step_size / 2) * proposal_tempered_gradient[parameter_index, particle_index]
        end
    end

    _batched_hamiltonian!(current_hamiltonian, current_tempered_values, momentum, inverse_mass)
    for particle_index in 1:num_particles
        position = collect(view(particles, :, particle_index))
        tree_workspace = NUTSSubtreeWorkspace(parameter_total)
        continuation = NUTSContinuationState(parameter_total)
        proposal_gradient = view(proposal_gradient_buffer, :, particle_index)
        model_gradient_cache = column_gradient_caches[particle_index]
        _load_nuts_state!(
            tree_workspace.current,
            position,
            view(momentum, :, particle_index),
            current_tempered_values[particle_index],
            view(current_tempered_gradient, :, particle_index),
        )
        _load_nuts_state!(
            tree_workspace.next,
            view(proposal_particles, :, particle_index),
            view(proposal_momentum, :, particle_index),
            proposal_tempered_values[particle_index],
            view(proposal_tempered_gradient, :, particle_index),
        )
        initial_hamiltonian = current_hamiltonian[particle_index]
        _initialize_nuts_first_trajectory!(
            continuation,
            tree_workspace,
            valid[particle_index],
            directions[particle_index],
            initial_hamiltonian,
            inverse_mass,
            max_delta_energy,
            rng,
        )
        if max_tree_depth > 1
            _continue_tempered_nuts_proposal!(
                continuation,
                tree_workspace,
                view(column_gradient_buffer, :, particle_index),
                proposal_gradient,
                model,
                model_gradient_cache,
                inverse_mass,
                args,
                constraints,
                proposal_location,
                proposal_log_scale,
                beta,
                step_size,
                max_tree_depth,
                max_delta_energy,
                initial_hamiltonian,
                rng,
            )
        end
        proposal = continuation.proposal
        accept_stat, _, _, moved = _nuts_proposal_summary(continuation, position)
        total_accept_stat += accept_stat
        if moved
            copyto!(view(particles, :, particle_index), proposal.position)
            if max_tree_depth == 1 && valid[particle_index]
                logjoint_values[particle_index] = proposal_logjoint_values[particle_index]
                logproposal_values[particle_index] = proposal_logproposal_values[particle_index]
            else
                logjoint_values[particle_index] = logjoint_unconstrained(model, proposal.position, args, constraints)
                logproposal_values[particle_index] = _single_gaussian_logdensity(
                    proposal.position,
                    proposal_location,
                    proposal_log_scale,
                )
            end
            log_ratio[particle_index] = logjoint_values[particle_index] - logproposal_values[particle_index]
        end
    end

    return total_accept_stat / num_particles
end
