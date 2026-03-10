struct HMCChain
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    unconstrained_samples::Matrix{Float64}
    constrained_samples::Matrix{Float64}
    logjoint_values::Vector{Float64}
    accepted::BitVector
    step_size::Float64
    num_leapfrog_steps::Int
    gradient_eps::Float64
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
        ")",
    )
end

function acceptancerate(chain::HMCChain)
    isempty(chain.accepted) && return 0.0
    return count(identity, chain.accepted) / length(chain.accepted)
end

function _validate_hmc_arguments(num_params::Int, num_samples::Int, num_warmup::Int, step_size::Real, num_leapfrog_steps::Int, gradient_eps::Real)
    num_params > 0 || throw(ArgumentError("HMC requires at least one parameterized latent choice"))
    num_samples > 0 || throw(ArgumentError("HMC requires num_samples > 0"))
    num_warmup >= 0 || throw(ArgumentError("HMC requires num_warmup >= 0"))
    step_size > 0 || throw(ArgumentError("HMC requires step_size > 0"))
    num_leapfrog_steps > 0 || throw(ArgumentError("HMC requires num_leapfrog_steps > 0"))
    gradient_eps > 0 || throw(ArgumentError("HMC requires gradient_eps > 0"))
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

function _finite_difference_gradient(
    model::TeaModel,
    position::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
    gradient_eps::Float64,
)
    gradient = Vector{Float64}(undef, length(position))
    for idx in eachindex(position)
        delta = gradient_eps * max(1.0, abs(position[idx]))
        forward = copy(position)
        backward = copy(position)
        forward[idx] += delta
        backward[idx] -= delta
        f_plus = logjoint_unconstrained(model, forward, args, constraints)
        f_minus = logjoint_unconstrained(model, backward, args, constraints)
        gradient[idx] = (f_plus - f_minus) / (2 * delta)
    end
    return gradient
end

function _leapfrog(
    model::TeaModel,
    position::Vector{Float64},
    momentum::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    num_leapfrog_steps::Int,
    gradient_eps::Float64,
)
    q = copy(position)
    p = copy(momentum)

    gradient = _finite_difference_gradient(model, q, args, constraints, gradient_eps)
    all(isfinite, gradient) || return nothing
    p .+= (step_size / 2) .* gradient

    for leapfrog_step in 1:num_leapfrog_steps
        q .+= step_size .* p
        gradient = _finite_difference_gradient(model, q, args, constraints, gradient_eps)
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

function _hamiltonian(logjoint_value::Float64, momentum::AbstractVector)
    return -logjoint_value + sum(abs2, momentum) / 2
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
    gradient_eps::Real=1e-4,
    rng::AbstractRNG=Random.default_rng(),
)
    num_params = parametercount(parameterlayout(model))
    _validate_hmc_arguments(num_params, num_samples, num_warmup, step_size, num_leapfrog_steps, gradient_eps)

    position = _initial_hmc_position(model, args, constraints, initial_params, rng)
    current_logjoint = logjoint_unconstrained(model, position, args, constraints)
    isfinite(current_logjoint) ||
        throw(ArgumentError("initial HMC parameters produced a non-finite unconstrained logjoint"))

    unconstrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    constrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    logjoint_values = Vector{Float64}(undef, num_samples)
    accepted = falses(num_samples)
    total_iterations = num_warmup + num_samples
    hmc_step_size = Float64(step_size)
    hmc_gradient_eps = Float64(gradient_eps)

    sample_index = 0
    for iteration in 1:total_iterations
        momentum = randn(rng, num_params)
        proposal = _leapfrog(
            model,
            position,
            momentum,
            args,
            constraints,
            hmc_step_size,
            num_leapfrog_steps,
            hmc_gradient_eps,
        )

        accepted_step = false
        if !isnothing(proposal)
            proposed_position, proposed_momentum, proposed_logjoint = proposal
            current_hamiltonian = _hamiltonian(current_logjoint, momentum)
            proposed_hamiltonian = _hamiltonian(proposed_logjoint, proposed_momentum)
            log_accept_ratio = current_hamiltonian - proposed_hamiltonian

            if log(rand(rng)) < min(0.0, log_accept_ratio)
                position = proposed_position
                current_logjoint = proposed_logjoint
                accepted_step = true
            end
        end

        if iteration > num_warmup
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] = transform_to_constrained(model, position)
            logjoint_values[sample_index] = current_logjoint
            accepted[sample_index] = accepted_step
        end
    end

    return HMCChain(
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        accepted,
        hmc_step_size,
        num_leapfrog_steps,
        hmc_gradient_eps,
    )
end
