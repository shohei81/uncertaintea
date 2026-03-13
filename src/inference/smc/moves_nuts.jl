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

function _batched_tempered_nuts_leapfrog_step_to!(
    destination_position::AbstractMatrix,
    destination_momentum::AbstractMatrix,
    destination_gradient::AbstractMatrix,
    destination_logjoint::AbstractVector,
    valid::AbstractVector{Bool},
    start_position::AbstractMatrix,
    start_momentum::AbstractMatrix,
    start_gradient::AbstractMatrix,
    logjoint_values::AbstractVector,
    logjoint_gradient::AbstractMatrix,
    logproposal_values::AbstractVector,
    logproposal_gradient::AbstractMatrix,
    proposal_noise::AbstractMatrix,
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    inverse_mass_matrix::AbstractVector,
    step_size::Float64,
    direction::AbstractVector{Int},
    active::AbstractVector{Bool},
)
    num_particles = size(start_position, 2)
    size(destination_position) == size(start_position) == size(start_momentum) == size(start_gradient) ||
        throw(DimensionMismatch("expected batched tempered NUTS buffers of matching size"))
    size(destination_momentum) == size(start_position) == size(destination_gradient) ||
        throw(DimensionMismatch("expected batched tempered NUTS destination buffers of matching size"))
    length(destination_logjoint) == num_particles == length(valid) == length(direction) == length(active) ||
        throw(DimensionMismatch("expected batched tempered NUTS vectors of length $num_particles"))

    copyto!(destination_position, start_position)
    copyto!(destination_momentum, start_momentum)
    fill!(valid, false)

    for particle_index in 1:num_particles
        active[particle_index] || continue
        valid[particle_index] = true
        signed_step_size = direction[particle_index] * step_size
        for parameter_index in axes(start_position, 1)
            destination_momentum[parameter_index, particle_index] +=
                (signed_step_size / 2) * start_gradient[parameter_index, particle_index]
            destination_position[parameter_index, particle_index] +=
                signed_step_size *
                inverse_mass_matrix[parameter_index] *
                destination_momentum[parameter_index, particle_index]
        end
    end

    _batched_tempered_target!(
        destination_logjoint,
        destination_gradient,
        logjoint_values,
        logjoint_gradient,
        logproposal_values,
        logproposal_gradient,
        proposal_noise,
        model,
        cache,
        destination_position,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
    )

    for particle_index in 1:num_particles
        valid[particle_index] || continue
        if !isfinite(destination_logjoint[particle_index]) ||
           !isfinite(logjoint_values[particle_index]) ||
           !all(isfinite, view(destination_gradient, :, particle_index))
            valid[particle_index] = false
            continue
        end
        signed_step_size = direction[particle_index] * step_size
        for parameter_index in axes(destination_position, 1)
            destination_momentum[parameter_index, particle_index] +=
                (signed_step_size / 2) * destination_gradient[parameter_index, particle_index]
        end
    end

    return destination_position, destination_momentum, destination_gradient, destination_logjoint, valid
end

function _copy_nuts_column_to_state!(
    destination::NUTSState,
    position::AbstractMatrix,
    momentum::AbstractMatrix,
    logjoint::AbstractVector,
    gradient::AbstractMatrix,
    index::Int,
)
    return _load_nuts_state!(
        destination,
        view(position, :, index),
        view(momentum, :, index),
        logjoint[index],
        view(gradient, :, index),
    )
end

function _continue_batched_tempered_nuts_depth_cohort!(
    workspace::TemperedNUTSCohortWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    cohort_depth::Int,
    rng::AbstractRNG,
)
    num_particles = length(continuations)
    parameter_total = size(workspace.current_position, 1)
    active = workspace.active
    subtree_active = workspace.subtree_active
    directions = workspace.directions
    current_position = workspace.current_position
    current_momentum = workspace.current_momentum
    current_gradient = workspace.current_gradient
    current_logjoint = workspace.current_logjoint
    next_position = workspace.next_position
    next_momentum = workspace.next_momentum
    next_gradient = workspace.next_gradient
    next_logjoint = workspace.next_logjoint
    left_position = workspace.left_position
    left_momentum = workspace.left_momentum
    left_gradient = workspace.left_gradient
    left_logjoint = workspace.left_logjoint
    right_position = workspace.right_position
    right_momentum = workspace.right_momentum
    right_gradient = workspace.right_gradient
    right_logjoint = workspace.right_logjoint
    proposal_position = workspace.proposal_position
    proposal_momentum = workspace.proposal_momentum
    proposal_gradient = workspace.proposal_gradient
    proposal_logjoint = workspace.proposal_logjoint
    logjoint_values = workspace.logjoint_values
    logjoint_gradient = workspace.logjoint_gradient
    logproposal_values = workspace.logproposal_values
    logproposal_gradient = workspace.logproposal_gradient
    proposal_noise = workspace.proposal_noise
    valid = workspace.valid
    subtree_log_weight = workspace.subtree_log_weight
    subtree_accept_stat_sum = workspace.subtree_accept_stat_sum
    subtree_accept_stat_count = workspace.subtree_accept_stat_count
    subtree_integration_steps = workspace.subtree_integration_steps
    subtree_proposal_energy = workspace.subtree_proposal_energy
    subtree_proposal_energy_error = workspace.subtree_proposal_energy_error
    subtree_turning = workspace.subtree_turning
    subtree_divergent = workspace.subtree_divergent

    fill!(active, false)
    for particle_index in 1:num_particles
        continuation = continuations[particle_index]
        active[particle_index] =
            continuation.tree_depth == cohort_depth &&
            _nuts_continuation_active(
                continuation.tree_depth,
                max_tree_depth,
                continuation.divergent,
                continuation.turning,
            )
    end
    any(active) || return continuations
    _sample_batched_nuts_directions!(directions, rng)
    copyto!(subtree_active, active)
    fill!(subtree_log_weight, -Inf)
    fill!(subtree_accept_stat_sum, 0.0)
    fill!(subtree_accept_stat_count, 0)
    fill!(subtree_integration_steps, 0)
    fill!(subtree_proposal_energy, Inf)
    fill!(subtree_proposal_energy_error, Inf)
    fill!(subtree_turning, false)
    fill!(subtree_divergent, false)

    for particle_index in 1:num_particles
        active[particle_index] || continue
        start_state = _nuts_subtree_start_state(continuations[particle_index], directions[particle_index])
        copyto!(view(current_position, :, particle_index), start_state.position)
        copyto!(view(current_momentum, :, particle_index), start_state.momentum)
        copyto!(view(current_gradient, :, particle_index), start_state.gradient)
        current_logjoint[particle_index] = start_state.logjoint
        copyto!(view(left_position, :, particle_index), start_state.position)
        copyto!(view(left_momentum, :, particle_index), start_state.momentum)
        copyto!(view(left_gradient, :, particle_index), start_state.gradient)
        left_logjoint[particle_index] = start_state.logjoint
        copyto!(view(right_position, :, particle_index), start_state.position)
        copyto!(view(right_momentum, :, particle_index), start_state.momentum)
        copyto!(view(right_gradient, :, particle_index), start_state.gradient)
        right_logjoint[particle_index] = start_state.logjoint
        copyto!(view(proposal_position, :, particle_index), start_state.position)
        copyto!(view(proposal_momentum, :, particle_index), start_state.momentum)
        copyto!(view(proposal_gradient, :, particle_index), start_state.gradient)
        proposal_logjoint[particle_index] = start_state.logjoint
    end

    for _ in 1:(1 << cohort_depth)
        any(subtree_active) || break
        _batched_tempered_nuts_leapfrog_step_to!(
            next_position,
            next_momentum,
            next_gradient,
            next_logjoint,
            valid,
            current_position,
            current_momentum,
            current_gradient,
            logjoint_values,
            logjoint_gradient,
            logproposal_values,
            logproposal_gradient,
            proposal_noise,
            model,
            cache,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
            inverse_mass_matrix,
            step_size,
            directions,
            subtree_active,
        )

        for particle_index in 1:num_particles
            subtree_active[particle_index] || continue
            if !valid[particle_index]
                subtree_divergent[particle_index] = true
                subtree_active[particle_index] = false
                continue
            end
            subtree_integration_steps[particle_index] += 1
            if directions[particle_index] < 0
                copyto!(view(left_position, :, particle_index), view(next_position, :, particle_index))
                copyto!(view(left_momentum, :, particle_index), view(next_momentum, :, particle_index))
                copyto!(view(left_gradient, :, particle_index), view(next_gradient, :, particle_index))
                left_logjoint[particle_index] = next_logjoint[particle_index]
            else
                copyto!(view(right_position, :, particle_index), view(next_position, :, particle_index))
                copyto!(view(right_momentum, :, particle_index), view(next_momentum, :, particle_index))
                copyto!(view(right_gradient, :, particle_index), view(next_gradient, :, particle_index))
                right_logjoint[particle_index] = next_logjoint[particle_index]
            end
            proposed_energy = _hamiltonian(
                next_logjoint[particle_index],
                view(next_momentum, :, particle_index),
                inverse_mass_matrix,
            )
            delta_energy = proposed_energy - initial_hamiltonian[particle_index]
            if !isfinite(delta_energy) || delta_energy > max_delta_energy
                subtree_divergent[particle_index] = true
                subtree_active[particle_index] = false
                continue
            end
            accept_prob = min(1.0, exp(min(0.0, -delta_energy)))
            subtree_accept_stat_sum[particle_index] += accept_prob
            subtree_accept_stat_count[particle_index] += 1
            candidate_log_weight = -proposed_energy
            combined_log_weight = _logaddexp(
                subtree_log_weight[particle_index],
                candidate_log_weight,
            )
            if !isfinite(subtree_log_weight[particle_index]) ||
               log(rand(rng)) < candidate_log_weight - combined_log_weight
                copyto!(view(proposal_position, :, particle_index), view(next_position, :, particle_index))
                copyto!(view(proposal_momentum, :, particle_index), view(next_momentum, :, particle_index))
                copyto!(view(proposal_gradient, :, particle_index), view(next_gradient, :, particle_index))
                proposal_logjoint[particle_index] = next_logjoint[particle_index]
                subtree_proposal_energy[particle_index] = proposed_energy
                subtree_proposal_energy_error[particle_index] = delta_energy
            end
            subtree_log_weight[particle_index] = combined_log_weight
            subtree_turning[particle_index] = _is_turning(
                view(left_position, :, particle_index),
                view(right_position, :, particle_index),
                view(left_momentum, :, particle_index),
                view(right_momentum, :, particle_index),
            )
            subtree_active[particle_index] = !subtree_turning[particle_index]
        end

        current_position, next_position = next_position, current_position
        current_momentum, next_momentum = next_momentum, current_momentum
        current_gradient, next_gradient = next_gradient, current_gradient
        current_logjoint, next_logjoint = next_logjoint, current_logjoint
    end

    for particle_index in 1:num_particles
        active[particle_index] || continue
        continuation = continuations[particle_index]
        continuation.tree_depth += 1
        if subtree_integration_steps[particle_index] == 0
            continuation.divergent = subtree_divergent[particle_index]
            continue
        end
        if directions[particle_index] < 0
            _copy_nuts_column_to_state!(
                continuation.left,
                left_position,
                left_momentum,
                left_logjoint,
                left_gradient,
                particle_index,
            )
        else
            _copy_nuts_column_to_state!(
                continuation.right,
                right_position,
                right_momentum,
                right_logjoint,
                right_gradient,
                particle_index,
            )
        end
        combined_log_weight = continuation.log_weight
        if isfinite(subtree_log_weight[particle_index])
            combined_log_weight = _logaddexp(
                continuation.log_weight,
                subtree_log_weight[particle_index],
            )
            if log(rand(rng)) < subtree_log_weight[particle_index] - combined_log_weight
                _copy_nuts_column_to_state!(
                    continuation.proposal,
                    proposal_position,
                    proposal_momentum,
                    proposal_logjoint,
                    proposal_gradient,
                    particle_index,
                )
                continuation.proposal_energy = subtree_proposal_energy[particle_index]
                continuation.proposal_energy_error = subtree_proposal_energy_error[particle_index]
            end
        end
        continuation.integration_steps += subtree_integration_steps[particle_index]
        continuation.accept_stat_sum += subtree_accept_stat_sum[particle_index]
        continuation.accept_stat_count += subtree_accept_stat_count[particle_index]
        if isfinite(subtree_log_weight[particle_index])
            continuation.log_weight = combined_log_weight
        end
        continuation.divergent = subtree_divergent[particle_index]
        _merge_nuts_continuation_turning!(continuation, subtree_turning[particle_index])
    end

    return continuations
end

function _continue_batched_tempered_nuts_cohorts!(
    workspace::TemperedNUTSMoveWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    while true
        active_depth, active_depth_count = _tempered_nuts_active_depth!(
            _tempered_nuts_depth_counts!(workspace, max_tree_depth),
            continuations,
            max_tree_depth,
        )
        active_depth_count > 0 || break
        _continue_batched_tempered_nuts_depth_cohort!(
            workspace.cohort,
            continuations,
            model,
            cache,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
            step_size,
            max_tree_depth,
            max_delta_energy,
            initial_hamiltonian,
            inverse_mass_matrix,
            active_depth,
            rng,
        )
    end
    return continuations
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
    workspace::TemperedNUTSMoveWorkspace,
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
    _validate_tempered_nuts_move_workspace!(workspace, particles, inverse_mass_matrix)
    num_particles = workspace.num_particles
    parameter_total = workspace.parameter_total
    cache = workspace.cache
    momentum = workspace.momentum
    proposal_particles = workspace.proposal_particles
    proposal_momentum = workspace.proposal_momentum
    current_logjoint_gradient = workspace.current_logjoint_gradient
    current_logproposal_gradient = workspace.current_logproposal_gradient
    current_tempered_gradient = workspace.current_tempered_gradient
    current_tempered_values = workspace.current_tempered_values
    proposal_logjoint_values = workspace.proposal_logjoint_values
    proposal_logproposal_values = workspace.proposal_logproposal_values
    proposal_tempered_values = workspace.proposal_tempered_values
    proposal_logjoint_gradient = workspace.proposal_logjoint_gradient
    proposal_logproposal_gradient = workspace.proposal_logproposal_gradient
    proposal_tempered_gradient = workspace.proposal_tempered_gradient
    proposal_noise = workspace.proposal_noise
    current_hamiltonian = workspace.current_hamiltonian
    directions = workspace.directions
    valid = workspace.valid
    inverse_mass = workspace.inverse_mass
    tree_workspaces = workspace.tree_workspaces
    continuations = workspace.continuations
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
    _sample_batched_momentum!(momentum, rng, workspace.sqrt_inverse_mass)
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
        tree_workspace = tree_workspaces[particle_index]
        continuation = continuations[particle_index]
        _load_nuts_state!(
            tree_workspace.current,
            view(particles, :, particle_index),
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
        _initialize_nuts_first_trajectory!(
            continuation,
            tree_workspace,
            valid[particle_index],
            directions[particle_index],
            current_hamiltonian[particle_index],
            inverse_mass,
            max_delta_energy,
            rng,
        )
    end

    if max_tree_depth > 1
        _continue_batched_tempered_nuts_cohorts!(
            workspace,
            continuations,
            model,
            cache,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
            step_size,
            max_tree_depth,
            max_delta_energy,
            current_hamiltonian,
            inverse_mass,
            rng,
        )
    end

    for particle_index in 1:num_particles
        current_position = view(particles, :, particle_index)
        continuation = continuations[particle_index]
        proposal = continuation.proposal
        accept_stat, _, _, moved = _nuts_proposal_summary(continuation, current_position)
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
    workspace = TemperedNUTSMoveWorkspace(model, particles, args, constraints)
    return _batched_nuts_move!(
        workspace,
        particles,
        logjoint_values,
        logproposal_values,
        log_ratio,
        model,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
        step_size,
        max_tree_depth,
        max_delta_energy,
        inverse_mass_matrix,
        rng,
    )
end
