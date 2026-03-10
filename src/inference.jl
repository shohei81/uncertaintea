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

Base.length(chain::HMCChain) = size(chain.unconstrained_samples, 2)

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

function acceptancerate(chain::HMCChain)
    isempty(chain.accepted) && return 0.0
    return count(identity, chain.accepted) / length(chain.accepted)
end

function divergencerate(chain::HMCChain)
    isempty(chain.divergent) && return 0.0
    return count(identity, chain.divergent) / length(chain.divergent)
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

            if adapt_mass_matrix
                _update_running_variance!(variance_state, position)
                if variance_state.count >= mass_matrix_min_samples
                    inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
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
