struct HMCChain
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    unconstrained_samples::Matrix{Float64}
    constrained_samples::Matrix{Float64}
    logjoint_values::Vector{Float64}
    energies::Vector{Float64}
    energy_errors::Vector{Float64}
    accepted::BitVector
    divergent::BitVector
    step_size::Float64
    mass_matrix::Vector{Float64}
    num_leapfrog_steps::Int
    target_accept::Float64
end

struct HMCChains{A,C}
    model::TeaModel
    args::A
    constraints::C
    chains::Vector{HMCChain}
end

struct HMCParameterSummary
    index::Int
    binding::Symbol
    address::Any
    mean::Float64
    sd::Float64
    quantiles::Vector{Float64}
    rhat::Float64
    ess::Float64
end

struct HMCSummary
    model::TeaModel
    space::Symbol
    quantile_probs::Vector{Float64}
    parameters::Vector{HMCParameterSummary}
end

mutable struct DualAveragingState
    target_accept::Float64
    gamma::Float64
    t0::Float64
    kappa::Float64
    mu::Float64
    log_step_size::Float64
    log_step_size_avg::Float64
    hbar::Float64
    iteration::Int
end

mutable struct RunningVarianceState
    mean::Vector{Float64}
    m2::Vector{Float64}
    count::Int
end

struct WarmupSchedule
    initial_buffer::Int
    slow_window_ends::Vector{Int}
    terminal_buffer::Int
end

Base.length(chain::HMCChain) = size(chain.unconstrained_samples, 2)
Base.length(chains::HMCChains) = length(chains.chains)
Base.length(summary::HMCSummary) = length(summary.parameters)
Base.getindex(chains::HMCChains, index::Int) = chains.chains[index]
Base.getindex(summary::HMCSummary, index::Int) = summary.parameters[index]
Base.firstindex(chains::HMCChains) = firstindex(chains.chains)
Base.firstindex(summary::HMCSummary) = firstindex(summary.parameters)
Base.lastindex(chains::HMCChains) = lastindex(chains.chains)
Base.lastindex(summary::HMCSummary) = lastindex(summary.parameters)
Base.iterate(chains::HMCChains, state...) = iterate(chains.chains, state...)
Base.iterate(summary::HMCSummary, state...) = iterate(summary.parameters, state...)

function Base.show(io::IO, chain::HMCChain)
    print(
        io,
        "HMCChain(",
        chain.model.name,
        ", samples=",
        length(chain),
        ", acceptance_rate=",
        round(acceptancerate(chain); digits=3),
        ", divergences=",
        count(identity, chain.divergent),
        ", step_size=",
        round(chain.step_size; digits=4),
        ")",
    )
end

function Base.show(io::IO, chains::HMCChains)
    print(
        io,
        "HMCChains(",
        chains.model.name,
        ", chains=",
        length(chains),
        ", samples=",
        numsamples(chains),
        ", acceptance_rate=",
        round(acceptancerate(chains); digits=3),
        ", divergences=",
        sum(count(identity, chain.divergent) for chain in chains.chains),
        ")",
    )
end

function Base.show(io::IO, summary::HMCSummary)
    print(
        io,
        "HMCSummary(",
        summary.model.name,
        ", space=",
        summary.space,
        ", parameters=",
        length(summary),
        ", quantiles=",
        summary.quantile_probs,
        ")",
    )
end

function acceptancerate(chain::HMCChain)
    isempty(chain.accepted) && return 0.0
    return count(identity, chain.accepted) / length(chain.accepted)
end

function acceptancerate(chains::HMCChains)
    total_samples = sum(length(chain.accepted) for chain in chains.chains)
    total_samples == 0 && return 0.0
    return sum(count(identity, chain.accepted) for chain in chains.chains) / total_samples
end

function divergencerate(chain::HMCChain)
    isempty(chain.divergent) && return 0.0
    return count(identity, chain.divergent) / length(chain.divergent)
end

function divergencerate(chains::HMCChains)
    total_samples = sum(length(chain.divergent) for chain in chains.chains)
    total_samples == 0 && return 0.0
    return sum(count(identity, chain.divergent) for chain in chains.chains) / total_samples
end

function _diagnostic_space_samples(chain::HMCChain, space::Symbol)
    if space === :constrained
        return chain.constrained_samples
    elseif space === :unconstrained
        return chain.unconstrained_samples
    end

    throw(ArgumentError("diagnostic space must be :constrained or :unconstrained"))
end

function nchains(chains::HMCChains)
    return length(chains)
end

function numsamples(chains::HMCChains)
    isempty(chains.chains) && return 0
    return length(first(chains.chains))
end

function _summary_address(address::AddressSpec)
    parts = Any[part isa AddressLiteralPart ? part.value : part.value for part in address.parts]
    return length(parts) == 1 ? first(parts) : Tuple(parts)
end

function _validate_hmc_diagnostics(chains::HMCChains, space::Symbol)
    length(chains) >= 2 || throw(ArgumentError("multi-chain diagnostics require at least 2 chains"))
    num_samples = numsamples(chains)
    num_samples >= 4 || throw(ArgumentError("multi-chain diagnostics require at least 4 samples per chain"))

    first_samples = _diagnostic_space_samples(first(chains.chains), space)
    num_params = size(first_samples, 1)
    for chain in chains.chains
        chain_samples = _diagnostic_space_samples(chain, space)
        size(chain_samples, 1) == num_params ||
            throw(DimensionMismatch("all chains must have the same parameter dimension"))
        size(chain_samples, 2) == num_samples ||
            throw(DimensionMismatch("all chains must have the same number of samples"))
    end

    return num_params, num_samples
end

function _validate_summary_quantiles(quantile_probs)
    isempty(quantile_probs) && throw(ArgumentError("summary quantiles must be non-empty"))
    probabilities = Float64[Float64(prob) for prob in quantile_probs]
    for prob in probabilities
        0.0 <= prob <= 1.0 || throw(ArgumentError("summary quantiles must lie in [0, 1]"))
    end
    return probabilities
end

function _sample_mean(values::AbstractVector)
    return sum(values) / length(values)
end

function _sample_variance(values::AbstractVector, mean_value::Real=_sample_mean(values))
    length(values) > 1 || return 0.0
    return sum((value - mean_value)^2 for value in values) / (length(values) - 1)
end

function _sample_sd(values::AbstractVector, mean_value::Real=_sample_mean(values))
    return sqrt(max(_sample_variance(values, mean_value), 0.0))
end

function _pooled_parameter_draws(chains::HMCChains, parameter_index::Int, space::Symbol)
    _, num_samples = _validate_hmc_diagnostics(chains, space)
    pooled = Vector{Float64}(undef, length(chains) * num_samples)
    offset = 1
    for chain in chains.chains
        samples = _diagnostic_space_samples(chain, space)
        pooled[offset:(offset + num_samples - 1)] = samples[parameter_index, :]
        offset += num_samples
    end
    return pooled
end

function _quantile(sorted_values::AbstractVector, probability::Float64)
    num_values = length(sorted_values)
    num_values == 0 && throw(ArgumentError("quantile requires at least one value"))
    num_values == 1 && return Float64(sorted_values[1])

    position = 1 + (num_values - 1) * probability
    lower = floor(Int, position)
    upper = ceil(Int, position)
    lower == upper && return Float64(sorted_values[lower])
    weight = position - lower
    return (1 - weight) * sorted_values[lower] + weight * sorted_values[upper]
end

function _quantiles(values::AbstractVector, probabilities::AbstractVector{Float64})
    sorted_values = sort(collect(values))
    return Float64[_quantile(sorted_values, probability) for probability in probabilities]
end

function _split_chain_parameter_draws(chains::HMCChains, parameter_index::Int, space::Symbol)
    _, num_samples = _validate_hmc_diagnostics(chains, space)
    split_samples = fld(num_samples, 2)
    even_samples = 2 * split_samples
    split_draws = Matrix{Float64}(undef, 2 * length(chains), split_samples)

    for (chain_index, chain) in enumerate(chains.chains)
        samples = _diagnostic_space_samples(chain, space)
        split_draws[2 * chain_index - 1, :] = samples[parameter_index, 1:split_samples]
        split_draws[2 * chain_index, :] = samples[parameter_index, split_samples + 1:even_samples]
    end

    return split_draws
end

function _chain_draw_statistics(draws::AbstractMatrix)
    num_chains, num_samples = size(draws)
    chain_means = Vector{Float64}(undef, num_chains)
    chain_variances = Vector{Float64}(undef, num_chains)
    for chain_index in 1:num_chains
        chain_draws = view(draws, chain_index, :)
        chain_means[chain_index] = _sample_mean(chain_draws)
        chain_variances[chain_index] = _sample_variance(chain_draws, chain_means[chain_index])
    end

    within_variance = _sample_mean(chain_variances)
    between_variance = num_samples > 1 ? num_samples * _sample_variance(chain_means) : 0.0
    var_plus = ((num_samples - 1) / num_samples) * within_variance + between_variance / num_samples
    return chain_means, chain_variances, within_variance, between_variance, var_plus
end

function _split_rhat(draws::AbstractMatrix)
    _, _, within_variance, _, var_plus = _chain_draw_statistics(draws)
    if within_variance == 0
        return var_plus == 0 ? 1.0 : Inf
    end

    return sqrt(max(var_plus / within_variance, 1.0))
end

function _autocovariance(draws::AbstractVector, lag::Int, mean_value::Real)
    num_samples = length(draws)
    total = 0.0
    for index in 1:(num_samples - lag)
        total += (draws[index] - mean_value) * (draws[index + lag] - mean_value)
    end
    return total / num_samples
end

function _split_ess(draws::AbstractMatrix)
    num_chains, num_samples = size(draws)
    chain_means, _, within_variance, _, var_plus = _chain_draw_statistics(draws)
    total_draws = num_chains * num_samples

    if within_variance == 0 && var_plus == 0
        return Float64(total_draws)
    elseif var_plus <= 0
        return 0.0
    end

    pair_sums = Float64[]
    autocovariance_means = Vector{Float64}(undef, num_chains)
    for pair_start in 0:2:(num_samples - 1)
        pair_sum = 0.0
        for lag in pair_start:min(pair_start + 1, num_samples - 1)
            for chain_index in 1:num_chains
                autocovariance_means[chain_index] = _autocovariance(view(draws, chain_index, :), lag, chain_means[chain_index])
            end
            mean_autocovariance = _sample_mean(autocovariance_means)
            rho_hat = lag == 0 ? 1.0 : 1 - (within_variance - mean_autocovariance) / var_plus
            pair_sum += min(rho_hat, 1.0)
        end

        pair_sum > 0 || break
        push!(pair_sums, pair_sum)
    end

    for index in 2:length(pair_sums)
        pair_sums[index] = min(pair_sums[index], pair_sums[index - 1])
    end

    tau_hat = -1 + 2 * sum(pair_sums)
    tau_hat = max(tau_hat, 1.0)
    return min(Float64(total_draws), Float64(total_draws) / tau_hat)
end

function rhat(chains::HMCChains; space::Symbol=:constrained)
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    values = Vector{Float64}(undef, num_params)
    for parameter_index in 1:num_params
        values[parameter_index] = _split_rhat(_split_chain_parameter_draws(chains, parameter_index, space))
    end
    return values
end

function ess(chains::HMCChains; space::Symbol=:constrained)
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    values = Vector{Float64}(undef, num_params)
    for parameter_index in 1:num_params
        values[parameter_index] = _split_ess(_split_chain_parameter_draws(chains, parameter_index, space))
    end
    return values
end

function summarize(chains::HMCChains; space::Symbol=:constrained, quantiles=(0.05, 0.5, 0.95))
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    quantile_probs = _validate_summary_quantiles(quantiles)
    rhats = rhat(chains; space=space)
    ess_values = ess(chains; space=space)
    layout = parameterlayout(chains.model)
    parametercount(layout) == num_params ||
        throw(DimensionMismatch("summary expected $num_params parameters in layout, got $(parametercount(layout))"))

    parameters = Vector{HMCParameterSummary}(undef, num_params)
    for slot in layout.slots
        draws = _pooled_parameter_draws(chains, slot.index, space)
        mean_value = _sample_mean(draws)
        parameters[slot.index] = HMCParameterSummary(
            slot.index,
            slot.binding,
            _summary_address(slot.address),
            mean_value,
            _sample_sd(draws, mean_value),
            _quantiles(draws, quantile_probs),
            rhats[slot.index],
            ess_values[slot.index],
        )
    end

    return HMCSummary(chains.model, space, quantile_probs, parameters)
end

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

function _validate_hmc_chains_arguments(num_chains::Int)
    num_chains > 0 || throw(ArgumentError("HMC requires num_chains > 0"))
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
        1e-3,
        1,
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

    expected = parametercount(parameterlayout(model))
    length(initial_params) == expected || throw(DimensionMismatch("expected $expected initial parameters, got $(length(initial_params))"))
    return Float64[value for value in initial_params]
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
    num_chains::Int,
)
    batch_args = _validate_batched_args(args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    positions = Matrix{Float64}(undef, num_params, num_chains)
    seeds = rand(rng, UInt, num_chains)

    for chain_index in 1:num_chains
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, num_chains)
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
    return randn(rng, length(inverse_mass_matrix), num_chains) ./ sqrt.(inverse_mass_matrix)
end

function _batched_kinetic_energy(
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    num_chains = size(momentum, 2)
    energy = Vector{Float64}(undef, num_chains)
    for chain_index in 1:num_chains
        energy[chain_index] = _kinetic_energy(view(momentum, :, chain_index), inverse_mass_matrix)
    end
    return energy
end

function _batched_hamiltonian(
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    return (-Float64.(logjoint_values)) .+ _batched_kinetic_energy(momentum, inverse_mass_matrix)
end

function _batched_acceptance_probability(log_accept_ratio::AbstractVector)
    probabilities = Vector{Float64}(undef, length(log_accept_ratio))
    for index in eachindex(log_accept_ratio)
        probabilities[index] = _acceptance_probability(log_accept_ratio[index])
    end
    return probabilities
end

function _batched_leapfrog(
    model::TeaModel,
    position::Matrix{Float64},
    momentum::Matrix{Float64},
    gradient_cache::BatchedLogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    num_leapfrog_steps::Int,
)
    q = copy(position)
    p = copy(momentum)
    num_chains = size(q, 2)
    valid = trues(num_chains)
    gradient = batched_logjoint_gradient_unconstrained!(gradient_cache, q)

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

        gradient = batched_logjoint_gradient_unconstrained!(gradient_cache, q)
        for chain_index in 1:num_chains
            valid[chain_index] || continue
            if !all(isfinite, view(gradient, :, chain_index))
                valid[chain_index] = false
            elseif leapfrog_step < num_leapfrog_steps
                p[:, chain_index] .+= step_size .* gradient[:, chain_index]
            end
        end
    end

    for chain_index in 1:num_chains
        valid[chain_index] || continue
        p[:, chain_index] .+= (step_size / 2) .* gradient[:, chain_index]
        p[:, chain_index] .*= -1
    end

    proposed_logjoint = batched_logjoint_unconstrained(model, q, args, constraints)
    for chain_index in 1:num_chains
        if valid[chain_index] && !isfinite(proposed_logjoint[chain_index])
            valid[chain_index] = false
        end
    end

    return q, p, proposed_logjoint, valid
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

function _running_variance_state(num_params::Int)
    return RunningVarianceState(zeros(num_params), zeros(num_params), 0)
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

function _update_running_variance!(state::RunningVarianceState, sample::AbstractVector)
    state.count += 1
    delta = sample .- state.mean
    state.mean .+= delta ./ state.count
    delta2 = sample .- state.mean
    state.m2 .+= delta .* delta2
    return nothing
end

function _inverse_mass_matrix(state::RunningVarianceState, regularization::Float64)
    if state.count < 2
        return ones(length(state.mean))
    end

    variance = state.m2 ./ (state.count - 1)
    shrinkage = state.count / (state.count + 5.0)
    regularized_variance = shrinkage .* variance .+ (1 - shrinkage)
    return 1 ./ max.(regularized_variance, regularization)
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

function _chain_initial_params(initial_params, chain_index::Int, num_params::Int, num_chains::Int)
    if isnothing(initial_params)
        return nothing
    elseif initial_params isa AbstractMatrix
        size(initial_params) == (num_params, num_chains) ||
            throw(DimensionMismatch("expected initial_params matrix of size ($num_params, $num_chains), got $(size(initial_params))"))
        return collect(Float64, view(initial_params, :, chain_index))
    elseif initial_params isa AbstractVector && !isempty(initial_params) && first(initial_params) isa AbstractVector
        length(initial_params) == num_chains ||
            throw(DimensionMismatch("expected $num_chains initial parameter vectors, got $(length(initial_params))"))
        chain_params = initial_params[chain_index]
        length(chain_params) == num_params ||
            throw(DimensionMismatch("expected $num_params initial parameters for chain $chain_index, got $(length(chain_params))"))
        return Float64[value for value in chain_params]
    elseif initial_params isa AbstractVector
        length(initial_params) == num_params ||
            throw(DimensionMismatch("expected $num_params initial parameters, got $(length(initial_params))"))
        return Float64[value for value in initial_params]
    end

    throw(ArgumentError("unsupported initial_params container for multi-chain HMC"))
end

function hmc(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    num_params = parametercount(parameterlayout(model))
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

    position = _initial_hmc_position(model, args, constraints, initial_params, rng)
    current_logjoint = logjoint_unconstrained(model, position, args, constraints)
    isfinite(current_logjoint) ||
        throw(ArgumentError("initial HMC parameters produced a non-finite unconstrained logjoint"))
    gradient_cache = _logjoint_gradient_cache(model, position, args, constraints)

    unconstrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    constrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    logjoint_values = Vector{Float64}(undef, num_samples)
    energies = Vector{Float64}(undef, num_samples)
    energy_errors = Vector{Float64}(undef, num_samples)
    accepted = falses(num_samples)
    divergent = falses(num_samples)
    total_iterations = num_warmup + num_samples
    hmc_step_size = Float64(step_size)
    hmc_target_accept = Float64(target_accept)
    hmc_divergence_threshold = Float64(divergence_threshold)
    inverse_mass_matrix = ones(num_params)
    warmup_schedule = _warmup_schedule(num_warmup)
    mass_window_index = 1
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        hmc_step_size = _find_reasonable_step_size(
            model,
            position,
            current_logjoint,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            hmc_step_size,
            rng,
        )
    end
    dual_state = _dual_averaging_state(hmc_step_size, hmc_target_accept)
    variance_state = _running_variance_state(num_params)

    sample_index = 0
    for iteration in 1:total_iterations
        momentum = _sample_momentum(rng, inverse_mass_matrix)
        proposal = _leapfrog(
            model,
            position,
            momentum,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            hmc_step_size,
            num_leapfrog_steps,
        )

        accepted_step = false
        log_accept_ratio = _proposal_log_accept_ratio(current_logjoint, momentum, proposal, inverse_mass_matrix)
        accept_prob = _acceptance_probability(log_accept_ratio)
        proposal_energy, energy_error, divergent_step = _proposal_diagnostics(
            current_logjoint,
            momentum,
            proposal,
            inverse_mass_matrix,
            hmc_divergence_threshold,
        )
        sample_energy = isnothing(proposal) ? _hamiltonian(current_logjoint, momentum, inverse_mass_matrix) : proposal_energy
        if !isnothing(proposal)
            proposed_position, _, proposed_logjoint = proposal

            if log(rand(rng)) < min(0.0, log_accept_ratio)
                position = proposed_position
                current_logjoint = proposed_logjoint
                accepted_step = true
            end
        end

        if !accepted_step
            sample_energy = _hamiltonian(current_logjoint, momentum, inverse_mass_matrix)
        end

        if iteration <= num_warmup
            if adapt_step_size
                hmc_step_size = _update_step_size!(dual_state, accept_prob)
            end

            if adapt_mass_matrix &&
               mass_window_index <= length(warmup_schedule.slow_window_ends) &&
               iteration > warmup_schedule.initial_buffer
                _update_running_variance!(variance_state, position)
                if iteration == warmup_schedule.slow_window_ends[mass_window_index]
                    if variance_state.count >= mass_matrix_min_samples
                        inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                    end
                    variance_state = _running_variance_state(num_params)
                    mass_window_index += 1
                    if adapt_step_size && iteration < num_warmup
                        hmc_step_size = _find_reasonable_step_size(
                            model,
                            position,
                            current_logjoint,
                            gradient_cache,
                            inverse_mass_matrix,
                            args,
                            constraints,
                            hmc_step_size,
                            rng,
                        )
                        dual_state = _dual_averaging_state(hmc_step_size, hmc_target_accept)
                    end
                end
            end

            if iteration == num_warmup
                if adapt_step_size
                    hmc_step_size = _final_step_size(dual_state)
                end
                if adapt_mass_matrix && variance_state.count >= mass_matrix_min_samples
                    inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                end
            end
        else
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] = transform_to_constrained(model, position)
            logjoint_values[sample_index] = current_logjoint
            energies[sample_index] = sample_energy
            energy_errors[sample_index] = energy_error
            accepted[sample_index] = accepted_step
            divergent[sample_index] = divergent_step
        end
    end

    return HMCChain(
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        energies,
        energy_errors,
        accepted,
        divergent,
        hmc_step_size,
        1 ./ inverse_mass_matrix,
        num_leapfrog_steps,
        hmc_target_accept,
    )
end

function hmc_chains(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    _validate_hmc_chains_arguments(num_chains)
    num_params = parametercount(parameterlayout(model))
    seeds = rand(rng, UInt, num_chains)
    chains = Vector{HMCChain}(undef, num_chains)

    for chain_index in 1:num_chains
        chain_rng = MersenneTwister(seeds[chain_index])
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, num_chains)
        chains[chain_index] = hmc(
            model,
            args,
            constraints;
            num_samples=num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            initial_params=chain_initial_params,
            target_accept=target_accept,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            find_reasonable_step_size=find_reasonable_step_size,
            divergence_threshold=divergence_threshold,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            rng=chain_rng,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

function batched_hmc(
    model::TeaModel,
    args=(),
    constraints=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    divergence_threshold::Real=1000.0,
    rng::AbstractRNG=Random.default_rng(),
)
    num_params = parametercount(parameterlayout(model))
    _validate_batched_hmc_arguments(
        num_chains,
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        args,
        constraints,
    )

    batch_args = _validate_batched_args(args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    position = _initial_batched_hmc_positions(
        model,
        batch_args,
        batch_constraints,
        initial_params,
        rng,
        num_params,
        num_chains,
    )
    current_logjoint = batched_logjoint_unconstrained(model, position, batch_args, batch_constraints)
    all(isfinite, current_logjoint) ||
        throw(ArgumentError("initial batched HMC parameters produced a non-finite unconstrained logjoint"))
    gradient_cache = BatchedLogjointGradientCache(model, position, batch_args, batch_constraints)

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    logjoint_values = Matrix{Float64}(undef, num_samples, num_chains)
    energies = Matrix{Float64}(undef, num_samples, num_chains)
    energy_errors = Matrix{Float64}(undef, num_samples, num_chains)
    accepted = falses(num_samples, num_chains)
    divergent = falses(num_samples, num_chains)
    total_iterations = num_warmup + num_samples
    hmc_step_size = Float64(step_size)
    hmc_target_accept = Float64(target_accept)
    hmc_divergence_threshold = Float64(divergence_threshold)
    inverse_mass_matrix = ones(num_params)

    sample_index = 0
    for iteration in 1:total_iterations
        momentum = _sample_batched_momentum(rng, inverse_mass_matrix, num_chains)
        proposal_position, proposal_momentum, proposed_logjoint, valid = _batched_leapfrog(
            model,
            position,
            momentum,
            gradient_cache,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            hmc_step_size,
            num_leapfrog_steps,
        )

        current_hamiltonian = _batched_hamiltonian(current_logjoint, momentum, inverse_mass_matrix)
        proposed_hamiltonian = copy(current_hamiltonian)
        log_accept_ratio = fill(-Inf, num_chains)
        energy_error = fill(Inf, num_chains)
        divergent_step = trues(num_chains)

        for chain_index in 1:num_chains
            if valid[chain_index]
                proposed_hamiltonian[chain_index] = _hamiltonian(
                    proposed_logjoint[chain_index],
                    view(proposal_momentum, :, chain_index),
                    inverse_mass_matrix,
                )
                log_accept_ratio[chain_index] =
                    current_hamiltonian[chain_index] - proposed_hamiltonian[chain_index]
                energy_error[chain_index] = proposed_hamiltonian[chain_index] - current_hamiltonian[chain_index]
                divergent_step[chain_index] =
                    !isfinite(energy_error[chain_index]) ||
                    abs(energy_error[chain_index]) > hmc_divergence_threshold
            end
        end

        accept_prob = _batched_acceptance_probability(log_accept_ratio)
        accepted_step = falses(num_chains)
        for chain_index in 1:num_chains
            if valid[chain_index] && log(rand(rng)) < min(0.0, log_accept_ratio[chain_index])
                position[:, chain_index] = proposal_position[:, chain_index]
                current_logjoint[chain_index] = proposed_logjoint[chain_index]
                accepted_step[chain_index] = true
            end
        end

        if iteration > num_warmup
            sample_index += 1
            for chain_index in 1:num_chains
                unconstrained_samples[:, sample_index, chain_index] = position[:, chain_index]
                constrained_samples[:, sample_index, chain_index] =
                    transform_to_constrained(model, view(position, :, chain_index))
                logjoint_values[sample_index, chain_index] = current_logjoint[chain_index]
                energies[sample_index, chain_index] =
                    accepted_step[chain_index] ? proposed_hamiltonian[chain_index] : current_hamiltonian[chain_index]
                energy_errors[sample_index, chain_index] = energy_error[chain_index]
                accepted[sample_index, chain_index] = accepted_step[chain_index]
                divergent[sample_index, chain_index] = divergent_step[chain_index]
            end
        end
    end

    mass_matrix = 1 ./ inverse_mass_matrix
    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index in 1:num_chains
        chains[chain_index] = HMCChain(
            model,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            unconstrained_samples[:, :, chain_index],
            constrained_samples[:, :, chain_index],
            vec(logjoint_values[:, chain_index]),
            vec(energies[:, chain_index]),
            vec(energy_errors[:, chain_index]),
            vec(accepted[:, chain_index]),
            vec(divergent[:, chain_index]),
            hmc_step_size,
            copy(mass_matrix),
            num_leapfrog_steps,
            hmc_target_accept,
        )
    end

    return HMCChains(model, args, constraints, chains)
end
