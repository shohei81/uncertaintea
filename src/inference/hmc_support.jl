function _validate_hmc_arguments(
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
)
    num_params > 0 || throw(ArgumentError("HMC requires at least one parameterized latent choice"))
    num_samples > 0 || throw(ArgumentError("HMC requires num_samples > 0"))
    num_warmup >= 0 || throw(ArgumentError("HMC requires num_warmup >= 0"))
    step_size > 0 || throw(ArgumentError("HMC requires step_size > 0"))
    num_leapfrog_steps > 0 || throw(ArgumentError("HMC requires num_leapfrog_steps > 0"))
    0 < target_accept < 1 || throw(ArgumentError("HMC requires 0 < target_accept < 1"))
    divergence_threshold > 0 || throw(ArgumentError("HMC requires divergence_threshold > 0"))
    mass_matrix_regularization > 0 || throw(ArgumentError("HMC requires mass_matrix_regularization > 0"))
    mass_matrix_min_samples > 0 || throw(ArgumentError("HMC requires mass_matrix_min_samples > 0"))
    return nothing
end

function _validate_nuts_arguments(
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    max_tree_depth::Int,
    target_accept::Real,
    max_delta_energy::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
)
    num_params > 0 || throw(ArgumentError("NUTS requires at least one parameterized latent choice"))
    num_samples > 0 || throw(ArgumentError("NUTS requires num_samples > 0"))
    num_warmup >= 0 || throw(ArgumentError("NUTS requires num_warmup >= 0"))
    step_size > 0 || throw(ArgumentError("NUTS requires step_size > 0"))
    max_tree_depth > 0 || throw(ArgumentError("NUTS requires max_tree_depth > 0"))
    0 < target_accept < 1 || throw(ArgumentError("NUTS requires 0 < target_accept < 1"))
    max_delta_energy > 0 || throw(ArgumentError("NUTS requires max_delta_energy > 0"))
    mass_matrix_regularization > 0 || throw(ArgumentError("NUTS requires mass_matrix_regularization > 0"))
    mass_matrix_min_samples > 0 || throw(ArgumentError("NUTS requires mass_matrix_min_samples > 0"))
    return nothing
end

function _validate_hmc_chains_arguments(num_chains::Int, sampler_name::AbstractString="HMC")
    num_chains > 0 || throw(ArgumentError("$sampler_name requires num_chains > 0"))
    return nothing
end

function _validate_batched_hmc_arguments(
    num_chains::Int,
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    args,
    constraints,
)
    _validate_hmc_chains_arguments(num_chains)
    _validate_hmc_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )
    _validate_batched_args(args, num_chains)
    _validate_batched_constraints(constraints, num_chains)
    return nothing
end

function _validate_batched_nuts_arguments(
    num_chains::Int,
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    max_tree_depth::Int,
    target_accept::Real,
    max_delta_energy::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    args,
    constraints,
)
    _validate_hmc_chains_arguments(num_chains, "NUTS")
    _validate_nuts_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        max_tree_depth,
        target_accept,
        max_delta_energy,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )
    _validate_batched_args(args, num_chains)
    _validate_batched_constraints(constraints, num_chains)
    return nothing
end

function _initial_hmc_position(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    initial_params,
    rng::AbstractRNG,
)
    if isnothing(initial_params)
        trace, _ = generate(model, args, constraints; rng=rng)
        return transform_to_unconstrained(trace)
    end

    layout = parameterlayout(model)
    expected = parametercount(layout)
    constrained_expected = parametervaluecount(layout)
    if length(initial_params) == expected
        return Float64[value for value in initial_params]
    elseif length(initial_params) == constrained_expected
        return transform_to_unconstrained(model, Float64[value for value in initial_params])
    end
    throw(DimensionMismatch("expected $expected unconstrained or $constrained_expected constrained initial parameters, got $(length(initial_params))"))
end

function _sample_momentum(rng::AbstractRNG, inverse_mass_matrix::AbstractVector)
    return randn(rng, length(inverse_mass_matrix)) ./ sqrt.(inverse_mass_matrix)
end

function _initial_batched_hmc_positions(
    model::TeaModel,
    args,
    constraints,
    initial_params,
    rng::AbstractRNG,
    num_params::Int,
    constrained_num_params::Int,
    num_chains::Int,
)
    batch_args = _validate_batched_args(args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    positions = Matrix{Float64}(undef, num_params, num_chains)
    seeds = rand(rng, UInt, num_chains)

    for chain_index in 1:num_chains
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, constrained_num_params, num_chains)
        chain_rng = MersenneTwister(seeds[chain_index])
        positions[:, chain_index] = _initial_hmc_position(
            model,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            chain_initial_params,
            chain_rng,
        )
    end

    return positions
end

function _sample_batched_momentum(
    rng::AbstractRNG,
    inverse_mass_matrix::AbstractVector,
    num_chains::Int,
)
    momentum = Matrix{Float64}(undef, length(inverse_mass_matrix), num_chains)
    _sample_batched_momentum!(momentum, rng, sqrt.(Float64.(inverse_mass_matrix)))
    return momentum
end

function _sample_batched_momentum!(
    destination::AbstractMatrix,
    rng::AbstractRNG,
    sqrt_inverse_mass_matrix::AbstractVector,
)
    size(destination, 1) == length(sqrt_inverse_mass_matrix) ||
        throw(DimensionMismatch("expected momentum matrix with $(length(sqrt_inverse_mass_matrix)) rows, got $(size(destination, 1))"))

    for chain_index in axes(destination, 2)
        for parameter_index in eachindex(sqrt_inverse_mass_matrix)
            destination[parameter_index, chain_index] =
                randn(rng) / sqrt_inverse_mass_matrix[parameter_index]
        end
    end
    return destination
end

function _update_sqrt_inverse_mass_matrix!(
    destination::AbstractVector,
    inverse_mass_matrix::AbstractVector,
)
    length(destination) == length(inverse_mass_matrix) ||
        throw(DimensionMismatch("expected inverse mass matrix of length $(length(destination)), got $(length(inverse_mass_matrix))"))
    for index in eachindex(destination, inverse_mass_matrix)
        destination[index] = sqrt(Float64(inverse_mass_matrix[index]))
    end
    return destination
end

function _batched_kinetic_energy(
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    energy = Vector{Float64}(undef, size(momentum, 2))
    return _batched_kinetic_energy!(energy, momentum, inverse_mass_matrix)
end

function _batched_kinetic_energy!(
    destination::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    size(momentum, 1) == length(inverse_mass_matrix) ||
        throw(DimensionMismatch("expected momentum matrix with $(length(inverse_mass_matrix)) rows, got $(size(momentum, 1))"))
    size(momentum, 2) == length(destination) ||
        throw(DimensionMismatch("expected kinetic-energy destination of length $(size(momentum, 2)), got $(length(destination))"))

    for chain_index in axes(momentum, 2)
        kinetic_energy = 0.0
        for parameter_index in eachindex(inverse_mass_matrix)
            momentum_value = momentum[parameter_index, chain_index]
            kinetic_energy += momentum_value^2 * inverse_mass_matrix[parameter_index]
        end
        destination[chain_index] = kinetic_energy / 2
    end
    return destination
end

function _batched_hamiltonian(
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    hamiltonian = Vector{Float64}(undef, length(logjoint_values))
    return _batched_hamiltonian!(hamiltonian, logjoint_values, momentum, inverse_mass_matrix)
end

function _batched_hamiltonian!(
    destination::AbstractVector,
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    length(logjoint_values) == length(destination) ||
        throw(DimensionMismatch("expected hamiltonian inputs of length $(length(destination)), got $(length(logjoint_values))"))

    _batched_kinetic_energy!(destination, momentum, inverse_mass_matrix)
    for chain_index in eachindex(destination)
        destination[chain_index] -= Float64(logjoint_values[chain_index])
    end
    return destination
end

function _batched_acceptance_probability(log_accept_ratio::AbstractVector)
    probabilities = Vector{Float64}(undef, length(log_accept_ratio))
    return _batched_acceptance_probability!(probabilities, log_accept_ratio)
end

function _batched_acceptance_probability!(
    destination::AbstractVector,
    log_accept_ratio::AbstractVector,
)
    length(destination) == length(log_accept_ratio) ||
        throw(DimensionMismatch("expected acceptance-probability destination of length $(length(log_accept_ratio)), got $(length(destination))"))

    for index in eachindex(log_accept_ratio)
        destination[index] = _acceptance_probability(log_accept_ratio[index])
    end
    return destination
end

function _mean_acceptance_stats!(
    destination::AbstractVector,
    accept_sum::AbstractVector,
    accept_count::AbstractVector{Int},
)
    length(destination) == length(accept_sum) == length(accept_count) ||
        throw(DimensionMismatch("expected acceptance-stat inputs of matching length, got $(length(destination)), $(length(accept_sum)), and $(length(accept_count))"))
    for index in eachindex(destination)
        destination[index] = accept_count[index] == 0 ? 0.0 : accept_sum[index] / accept_count[index]
    end
    return destination
end

function _mean_acceptance_stat(
    accept_sum::Real,
    accept_count::Integer,
)
    return accept_count == 0 ? 0.0 : Float64(accept_sum) / accept_count
end

function _energy_errors!(
    destination::AbstractVector,
    proposed_energy::AbstractVector,
    current_energy::AbstractVector,
)
    length(destination) == length(proposed_energy) == length(current_energy) ||
        throw(DimensionMismatch("expected energy-error inputs of matching length, got $(length(destination)), $(length(proposed_energy)), and $(length(current_energy))"))
    for index in eachindex(destination)
        destination[index] = proposed_energy[index] - current_energy[index]
    end
    return destination
end

function _position_moved(
    proposal_position::AbstractVector,
    current_position::AbstractVector,
)
    length(proposal_position) == length(current_position) ||
        throw(DimensionMismatch("expected moved-position inputs of matching length, got $(length(proposal_position)) and $(length(current_position))"))
    for index in eachindex(proposal_position, current_position)
        proposal_position[index] == current_position[index] || return true
    end
    return false
end

function _nuts_proposal_summary(
    continuation::NUTSContinuationState,
    current_position::AbstractVector,
)
    proposed_energy = continuation.proposal_energy
    energy_error = continuation.proposal_energy_error
    accept_stat = _mean_acceptance_stat(
        continuation.accept_stat_sum,
        continuation.accept_stat_count,
    )
    moved = _position_moved(continuation.proposal.position, current_position)
    return accept_stat, proposed_energy, energy_error, moved
end

function _batched_positions_moved!(
    destination::AbstractVector{Bool},
    proposal_position::AbstractMatrix,
    current_position::AbstractMatrix,
)
    size(proposal_position) == size(current_position) ||
        throw(DimensionMismatch("expected moved-position inputs of matching size, got $(size(proposal_position)) and $(size(current_position))"))
    length(destination) == size(proposal_position, 2) ||
        throw(DimensionMismatch("expected moved-position destination of length $(size(proposal_position, 2)), got $(length(destination))"))
    for chain_index in eachindex(destination)
        moved = false
        for parameter_index in axes(proposal_position, 1)
            if proposal_position[parameter_index, chain_index] != current_position[parameter_index, chain_index]
                moved = true
                break
            end
        end
        destination[chain_index] = moved
    end
    return destination
end

function _finalize_batched_nuts_proposals!(
    workspace::BatchedNUTSWorkspace,
    position::AbstractMatrix,
)
    copyto!(workspace.proposed_logjoint, workspace.continuation_proposal_logjoint)
    copyto!(workspace.proposed_energy, workspace.continuation_proposed_energy)
    copyto!(workspace.energy_error, workspace.continuation_delta_energy)
    _mean_acceptance_stats!(
        workspace.accept_prob,
        workspace.continuation_accept_stat_sum,
        workspace.continuation_accept_stat_count,
    )
    _batched_positions_moved!(workspace.control.accepted_step, workspace.proposal_position, position)
    return workspace
end

function _mean_acceptance_probability(accept_prob::AbstractVector)
    isempty(accept_prob) && return 0.0
    return sum(accept_prob) / length(accept_prob)
end

function _mean_batched_adaptation_probability(
    accept_prob::AbstractVector,
    divergent::AbstractVector,
)
    length(accept_prob) == length(divergent) ||
        throw(DimensionMismatch("expected acceptance and divergence vectors of matching length, got $(length(accept_prob)) and $(length(divergent))"))
    isempty(accept_prob) && return 0.0

    total = 0.0
    for index in eachindex(accept_prob, divergent)
        total += divergent[index] ? 0.0 : accept_prob[index]
    end
    return total / length(accept_prob)
end

function _sample_nuts_direction(rng::AbstractRNG)
    return rand(rng, Bool) ? 1 : -1
end

function _sample_batched_nuts_directions!(
    destination::AbstractVector{Int},
    rng::AbstractRNG,
    active::AbstractVector{Bool},
)
    length(destination) == length(active) ||
        throw(DimensionMismatch("expected NUTS direction destination of length $(length(active)), got $(length(destination))"))
    for index in eachindex(destination, active)
        active[index] || continue
        destination[index] = _sample_nuts_direction(rng)
    end
    return destination
end

function _nuts_continuation_active(
    tree_depth::Integer,
    max_tree_depth::Integer,
    divergent::Bool,
    turning::Bool,
)
    return tree_depth < max_tree_depth && !divergent && !turning
end

function _nuts_subtree_start_state(
    continuation::NUTSContinuationState,
    direction::Int,
)
    return direction < 0 ? continuation.left : continuation.right
end

function _batched_leapfrog!(
    workspace::BatchedHMCWorkspace,
    model::TeaModel,
    position::Matrix{Float64},
    current_gradient::Matrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    num_leapfrog_steps::Int,
)
    q = workspace.proposal_position
    p = workspace.proposal_momentum
    proposed_gradient = workspace.proposal_gradient
    valid = workspace.valid
    num_chains = size(q, 2)
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(current_gradient) == size(position) ||
        throw(DimensionMismatch("expected current gradient workspace of size $(size(position)), got $(size(current_gradient))"))
    size(proposed_gradient) == size(position) ||
        throw(DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(proposed_gradient))"))

    copyto!(q, position)
    copyto!(p, workspace.momentum)
    fill!(valid, true)
    gradient = current_gradient

    for chain_index in 1:num_chains
        if !all(isfinite, view(gradient, :, chain_index))
            valid[chain_index] = false
        else
            p[:, chain_index] .+= (step_size / 2) .* gradient[:, chain_index]
        end
    end

    for leapfrog_step in 1:num_leapfrog_steps
        for chain_index in 1:num_chains
            valid[chain_index] || continue
            q[:, chain_index] .+= step_size .* (inverse_mass_matrix .* p[:, chain_index])
        end

        if leapfrog_step < num_leapfrog_steps
            gradient = batched_logjoint_gradient_unconstrained!(workspace.gradient_cache, q)
            for chain_index in 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(gradient, :, chain_index))
                    valid[chain_index] = false
                else
                    p[:, chain_index] .+= step_size .* gradient[:, chain_index]
                end
            end
        else
            proposed_logjoint, gradient = _batched_logjoint_and_gradient_unconstrained!(
                workspace.proposed_logjoint,
                workspace.gradient_cache,
                q,
            )
            copyto!(proposed_gradient, gradient)
            for chain_index in 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(proposed_gradient, :, chain_index)) || !isfinite(proposed_logjoint[chain_index])
                    valid[chain_index] = false
                end
            end
        end
    end

    for chain_index in 1:num_chains
        valid[chain_index] || continue
        p[:, chain_index] .+= (step_size / 2) .* proposed_gradient[:, chain_index]
        p[:, chain_index] .*= -1
    end

    return q, p, workspace.proposed_logjoint, proposed_gradient, valid
end

function _leapfrog(
    model::TeaModel,
    position::Vector{Float64},
    momentum::Vector{Float64},
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    num_leapfrog_steps::Int,
)
    q = copy(position)
    p = copy(momentum)

    gradient = _logjoint_gradient!(gradient_cache, q)
    all(isfinite, gradient) || return nothing
    p .+= (step_size / 2) .* gradient

    for leapfrog_step in 1:num_leapfrog_steps
        q .+= step_size .* (inverse_mass_matrix .* p)
        gradient = _logjoint_gradient!(gradient_cache, q)
        all(isfinite, gradient) || return nothing

        if leapfrog_step < num_leapfrog_steps
            p .+= step_size .* gradient
        end
    end

    p .+= (step_size / 2) .* gradient
    p .*= -1

    proposed_logjoint = logjoint_unconstrained(model, q, args, constraints)
    isfinite(proposed_logjoint) || return nothing
    return q, p, proposed_logjoint
end

function _kinetic_energy(momentum::AbstractVector, inverse_mass_matrix::AbstractVector)
    return sum((momentum .^ 2) .* inverse_mass_matrix) / 2
end

function _hamiltonian(logjoint_value::Float64, momentum::AbstractVector, inverse_mass_matrix::AbstractVector)
    return -logjoint_value + _kinetic_energy(momentum, inverse_mass_matrix)
end
