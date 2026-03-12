function _dual_averaging_state(step_size::Float64, target_accept::Float64)
    log_step_size = log(step_size)
    return DualAveragingState(
        target_accept,
        0.05,
        10.0,
        0.75,
        log(10 * step_size),
        log_step_size,
        log_step_size,
        0.0,
        0,
    )
end

function _update_step_size!(state::DualAveragingState, accept_prob::Float64)
    state.iteration += 1
    eta = 1 / (state.iteration + state.t0)
    state.hbar = (1 - eta) * state.hbar + eta * (state.target_accept - accept_prob)
    state.log_step_size = state.mu - sqrt(state.iteration) / state.gamma * state.hbar
    eta_bar = state.iteration^(-state.kappa)
    state.log_step_size_avg = eta_bar * state.log_step_size + (1 - eta_bar) * state.log_step_size_avg
    return exp(state.log_step_size)
end

function _final_step_size(state::DualAveragingState)
    return exp(state.log_step_size_avg)
end

function _running_variance_state(num_params::Int, window_length::Int=_RUNNING_VARIANCE_CLIP_START + 16)
    window_length > 0 || throw(ArgumentError("running variance state requires window_length > 0"))
    return RunningVarianceState(
        zeros(num_params),
        zeros(num_params),
        zeros(num_params),
        window_length,
        0,
        0.0,
        0.0,
    )
end

function _warmup_window_length(schedule::WarmupSchedule, window_index::Int)
    1 <= window_index <= length(schedule.slow_window_ends) ||
        throw(BoundsError(schedule.slow_window_ends, window_index))
    window_start = window_index == 1 ? schedule.initial_buffer + 1 : schedule.slow_window_ends[window_index - 1] + 1
    return schedule.slow_window_ends[window_index] - window_start + 1
end

function _warmup_window_start(schedule::WarmupSchedule, window_index::Int)
    1 <= window_index <= length(schedule.slow_window_ends) ||
        throw(BoundsError(schedule.slow_window_ends, window_index))
    return window_index == 1 ? schedule.initial_buffer + 1 : schedule.slow_window_ends[window_index - 1] + 1
end

function _running_variance_window_progress(state::RunningVarianceState)
    if state.window_length <= _RUNNING_VARIANCE_CLIP_START
        return 1.0
    end
    return min(
        max(state.count - _RUNNING_VARIANCE_CLIP_START, 0) /
        (state.window_length - _RUNNING_VARIANCE_CLIP_START),
        1.0,
    )
end

function _running_variance_clip_scale(state::RunningVarianceState)
    state.count <= _RUNNING_VARIANCE_CLIP_START && return _RUNNING_VARIANCE_CLIP_SCALE_EARLY
    progress = _running_variance_window_progress(state)
    return _RUNNING_VARIANCE_CLIP_SCALE_EARLY +
        (_RUNNING_VARIANCE_CLIP_SCALE_LATE - _RUNNING_VARIANCE_CLIP_SCALE_EARLY) * progress
end

function _running_variance_sample!(
    state::RunningVarianceState,
    sample::AbstractVector,
)
    clipped_sample = state.clipped_sample
    if state.count < _RUNNING_VARIANCE_CLIP_START
        copyto!(clipped_sample, sample)
        return clipped_sample
    end

    clip_scale = _running_variance_clip_scale(state)
    @inbounds for index in eachindex(clipped_sample, sample, state.mean, state.m2)
        variance = state.m2[index] / max(state.count - 1, 1)
        bound = clip_scale * sqrt(max(variance, _RUNNING_VARIANCE_FLOOR))
        delta = sample[index] - state.mean[index]
        clipped_sample[index] = state.mean[index] + clamp(delta, -bound, bound)
    end
    return clipped_sample
end

function _warmup_schedule(num_warmup::Int)
    num_warmup < 0 && throw(ArgumentError("warmup schedule requires num_warmup >= 0"))
    num_warmup == 0 && return WarmupSchedule(0, Int[], 0)

    if num_warmup < 20
        initial_buffer = min(5, max(num_warmup - 1, 0))
        if initial_buffer == num_warmup
            return WarmupSchedule(num_warmup, Int[], 0)
        end
        return WarmupSchedule(initial_buffer, [num_warmup], 0)
    end

    initial_buffer = min(max(5, fld(num_warmup, 10)), num_warmup)
    terminal_buffer = min(max(5, fld(num_warmup, 10)), max(num_warmup - initial_buffer, 0))
    slow_budget = num_warmup - initial_buffer - terminal_buffer
    if slow_budget <= 0
        return WarmupSchedule(num_warmup, Int[], 0)
    end

    window_ends = Int[]
    next_window_size = min(25, slow_budget)
    window_start = initial_buffer + 1
    remaining = slow_budget
    while remaining > 0
        window_size = remaining <= fld(3 * next_window_size, 2) ? remaining : next_window_size
        window_end = window_start + window_size - 1
        push!(window_ends, window_end)
        remaining -= window_size
        window_start = window_end + 1
        next_window_size *= 2
    end

    return WarmupSchedule(initial_buffer, window_ends, terminal_buffer)
end

function _running_variance_effective_count(state::RunningVarianceState)
    state.weight_square_sum <= 0 && return 0.0
    return state.weight_sum^2 / state.weight_square_sum
end

function _mass_adaptation_window_summary(
    schedule::WarmupSchedule,
    window_index::Int,
    state::RunningVarianceState,
    inverse_mass_matrix::AbstractVector,
    updated::Bool,
)
    mass_matrix = 1 ./ inverse_mass_matrix
    pooled_samples = state.count
    return HMCMassAdaptationWindowSummary(
        window_index,
        _warmup_window_start(schedule, window_index),
        schedule.slow_window_ends[window_index],
        state.window_length,
        pooled_samples,
        state.weight_sum,
        _running_variance_effective_count(state),
        pooled_samples == 0 ? 0.0 : state.weight_sum / pooled_samples,
        _RUNNING_VARIANCE_CLIP_SCALE_EARLY,
        _running_variance_clip_scale(state),
        updated,
        sum(mass_matrix) / length(mass_matrix),
        minimum(mass_matrix),
        maximum(mass_matrix),
    )
end

function _update_running_variance!(
    state::RunningVarianceState,
    sample::AbstractVector,
    weight::Real,
)
    weight_value = Float64(weight)
    0 <= weight_value <= 1 || throw(ArgumentError("running variance weight must lie in [0, 1], got $weight"))
    iszero(weight_value) && return nothing

    update_sample = _running_variance_sample!(state, sample)
    state.count += 1
    new_weight_sum = state.weight_sum + weight_value
    @inbounds for index in eachindex(update_sample, state.mean, state.m2)
        delta = update_sample[index] - state.mean[index]
        state.mean[index] += (weight_value / new_weight_sum) * delta
        delta2 = update_sample[index] - state.mean[index]
        state.m2[index] += weight_value * delta * delta2
    end
    state.weight_sum = new_weight_sum
    state.weight_square_sum += weight_value^2
    return nothing
end

function _update_running_variance!(state::RunningVarianceState, sample::AbstractVector)
    _update_running_variance!(state, sample, 1.0)
    return nothing
end

function _update_running_variance!(
    state::RunningVarianceState,
    samples::AbstractMatrix,
    weights::AbstractVector,
)
    size(samples, 2) == length(weights) ||
        throw(DimensionMismatch("expected $(size(samples, 2)) running variance weights, got $(length(weights))"))
    for column_index in axes(samples, 2)
        _update_running_variance!(state, view(samples, :, column_index), weights[column_index])
    end
    return nothing
end

function _inverse_mass_matrix(state::RunningVarianceState, regularization::Float64)
    effective_count = _running_variance_effective_count(state)
    if effective_count < 2
        return ones(length(state.mean))
    end

    variance_denom = state.weight_sum - state.weight_square_sum / state.weight_sum
    variance_denom <= 0 && return ones(length(state.mean))
    variance = state.m2 ./ variance_denom
    shrinkage = effective_count / (effective_count + 5.0)
    regularized_variance = shrinkage .* variance .+ (1 - shrinkage)
    return 1 ./ max.(regularized_variance, regularization)
end

function _mass_adaptation_weight(
    state::RunningVarianceState,
    accepted_step::Bool,
    accept_prob::Real,
    divergent_step::Bool,
)
    divergent_step && return 0.0
    accepted_step && return 1.0
    if !isfinite(accept_prob)
        return 0.0
    end
    progress = _running_variance_window_progress(state)
    rejection_weight = clamp(Float64(accept_prob), 0.0, 1.0)
    return _RUNNING_VARIANCE_REJECTION_WEIGHT_EARLY +
        (rejection_weight - _RUNNING_VARIANCE_REJECTION_WEIGHT_EARLY) * progress
end

function _mass_adaptation_weights!(
    state::RunningVarianceState,
    destination::AbstractVector,
    accepted_step::AbstractVector,
    accept_prob::AbstractVector,
    divergent_step::AbstractVector,
)
    length(destination) == length(accepted_step) == length(accept_prob) == length(divergent_step) ||
        throw(DimensionMismatch("mass adaptation weights require matching vector lengths"))
    @inbounds for index in eachindex(destination, accepted_step, accept_prob, divergent_step)
        destination[index] = _mass_adaptation_weight(
            state,
            accepted_step[index],
            accept_prob[index],
            divergent_step[index],
        )
    end
    return destination
end

function _acceptance_probability(log_accept_ratio::Float64)
    if !isfinite(log_accept_ratio)
        return 0.0
    elseif log_accept_ratio >= 0
        return 1.0
    end
    return exp(log_accept_ratio)
end

function _proposal_log_accept_ratio(
    current_logjoint::Float64,
    current_momentum::AbstractVector,
    proposal,
    inverse_mass_matrix::AbstractVector,
)
    isnothing(proposal) && return -Inf
    _, proposed_momentum, proposed_logjoint = proposal
    current_hamiltonian = _hamiltonian(current_logjoint, current_momentum, inverse_mass_matrix)
    proposed_hamiltonian = _hamiltonian(proposed_logjoint, proposed_momentum, inverse_mass_matrix)
    return current_hamiltonian - proposed_hamiltonian
end

function _proposal_diagnostics(
    current_logjoint::Float64,
    current_momentum::AbstractVector,
    proposal,
    inverse_mass_matrix::AbstractVector,
    divergence_threshold::Float64,
)
    current_hamiltonian = _hamiltonian(current_logjoint, current_momentum, inverse_mass_matrix)
    if isnothing(proposal)
        return current_hamiltonian, Inf, true
    end

    _, proposed_momentum, proposed_logjoint = proposal
    proposed_hamiltonian = _hamiltonian(proposed_logjoint, proposed_momentum, inverse_mass_matrix)
    energy_error = proposed_hamiltonian - current_hamiltonian
    divergent = !isfinite(energy_error) || abs(energy_error) > divergence_threshold
    return proposed_hamiltonian, energy_error, divergent
end

function _find_reasonable_batched_step_size(
    workspace::BatchedHMCWorkspace,
    model::TeaModel,
    position::Matrix{Float64},
    current_logjoint::AbstractVector,
    current_gradient::Matrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    reasonable_step_size = step_size
    min_step_size = 1e-8
    max_step_size = 1e3
    target_accept = 0.5

    for _ in 0:20
        _update_sqrt_inverse_mass_matrix!(workspace.sqrt_inverse_mass_matrix, inverse_mass_matrix)
        _sample_batched_momentum!(workspace.momentum, rng, workspace.sqrt_inverse_mass_matrix)
        _, proposal_momentum, proposed_logjoint, _, valid = _batched_leapfrog!(
            workspace,
            model,
            position,
            current_gradient,
            inverse_mass_matrix,
            args,
            constraints,
            reasonable_step_size,
            1,
        )

        current_hamiltonian = _batched_hamiltonian!(
            workspace.current_hamiltonian,
            current_logjoint,
            workspace.momentum,
            inverse_mass_matrix,
        )
        proposed_hamiltonian = workspace.proposed_hamiltonian
        copyto!(proposed_hamiltonian, current_hamiltonian)
        log_accept_ratio = workspace.log_accept_ratio
        fill!(log_accept_ratio, -Inf)
        divergent_step = workspace.divergent_step
        fill!(divergent_step, true)
        for chain_index in eachindex(current_logjoint)
            if valid[chain_index]
                proposed_hamiltonian[chain_index] = _hamiltonian(
                    proposed_logjoint[chain_index],
                    view(proposal_momentum, :, chain_index),
                    inverse_mass_matrix,
                )
                log_accept_ratio[chain_index] =
                    current_hamiltonian[chain_index] - proposed_hamiltonian[chain_index]
                energy_error = proposed_hamiltonian[chain_index] - current_hamiltonian[chain_index]
                divergent_step[chain_index] =
                    !isfinite(energy_error) || abs(energy_error) > divergence_threshold
            end
        end

        accept_prob = _batched_acceptance_probability!(workspace.accept_prob, log_accept_ratio)
        mean_accept_prob = _mean_batched_adaptation_probability(accept_prob, divergent_step)
        direction = mean_accept_prob > target_accept ? 1.0 : -1.0
        next_step_size = reasonable_step_size * (2.0 ^ direction)
        if next_step_size < min_step_size || next_step_size > max_step_size
            break
        end

        if (direction > 0 && mean_accept_prob <= target_accept) ||
           (direction < 0 && mean_accept_prob >= target_accept)
            break
        end
        reasonable_step_size = next_step_size
    end

    return clamp(reasonable_step_size, min_step_size, max_step_size)
end

function _find_reasonable_step_size(
    model::TeaModel,
    position::Vector{Float64},
    current_logjoint::Float64,
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    rng::AbstractRNG,
)
    reasonable_step_size = step_size
    min_step_size = 1e-8
    max_step_size = 1e3
    log_target = log(0.5)
    momentum = _sample_momentum(rng, inverse_mass_matrix)
    proposal = _leapfrog(
        model,
        position,
        momentum,
        gradient_cache,
        inverse_mass_matrix,
        args,
        constraints,
        reasonable_step_size,
        1,
    )
    log_accept_ratio = _proposal_log_accept_ratio(current_logjoint, momentum, proposal, inverse_mass_matrix)
    direction = log_accept_ratio > log_target ? 1.0 : -1.0

    for _ in 1:20
        next_step_size = reasonable_step_size * (2.0 ^ direction)
        if next_step_size < min_step_size || next_step_size > max_step_size
            break
        end

        reasonable_step_size = next_step_size
        momentum = _sample_momentum(rng, inverse_mass_matrix)
        proposal = _leapfrog(
            model,
            position,
            momentum,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            reasonable_step_size,
            1,
        )
        log_accept_ratio = _proposal_log_accept_ratio(current_logjoint, momentum, proposal, inverse_mass_matrix)
        if (direction > 0 && log_accept_ratio <= log_target) ||
           (direction < 0 && log_accept_ratio >= log_target)
            break
        end
    end

    return clamp(reasonable_step_size, min_step_size, max_step_size)
end

function _chain_initial_params(initial_params, chain_index::Int, num_params::Int, constrained_num_params::Int, num_chains::Int)
    if isnothing(initial_params)
        return nothing
    elseif initial_params isa AbstractMatrix
        (size(initial_params) == (num_params, num_chains) ||
         size(initial_params) == (constrained_num_params, num_chains)) ||
            throw(DimensionMismatch("expected initial_params matrix of size ($num_params, $num_chains) or ($constrained_num_params, $num_chains), got $(size(initial_params))"))
        return collect(Float64, view(initial_params, :, chain_index))
    elseif initial_params isa AbstractVector && !isempty(initial_params) && first(initial_params) isa AbstractVector
        length(initial_params) == num_chains ||
            throw(DimensionMismatch("expected $num_chains initial parameter vectors, got $(length(initial_params))"))
        chain_params = initial_params[chain_index]
        (length(chain_params) == num_params || length(chain_params) == constrained_num_params) ||
            throw(DimensionMismatch("expected $num_params unconstrained or $constrained_num_params constrained initial parameters for chain $chain_index, got $(length(chain_params))"))
        return Float64[value for value in chain_params]
    elseif initial_params isa AbstractVector
        (length(initial_params) == num_params || length(initial_params) == constrained_num_params) ||
            throw(DimensionMismatch("expected $num_params unconstrained or $constrained_num_params constrained initial parameters, got $(length(initial_params))"))
        return Float64[value for value in initial_params]
    end

    throw(ArgumentError("unsupported initial_params container for multi-chain HMC"))
end
