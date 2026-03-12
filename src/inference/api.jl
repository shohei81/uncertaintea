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
    acceptance_stats = Vector{Float64}(undef, num_samples)
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
    variance_state = _running_variance_state(
        num_params,
        isempty(warmup_schedule.slow_window_ends) ? (_RUNNING_VARIANCE_CLIP_START + 16) :
            _warmup_window_length(warmup_schedule, mass_window_index),
    )
    mass_adaptation_windows = HMCMassAdaptationWindowSummary[]

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
                _update_running_variance!(
                    variance_state,
                    position,
                    _mass_adaptation_weight(variance_state, accepted_step, accept_prob, divergent_step),
                )
                if iteration == warmup_schedule.slow_window_ends[mass_window_index]
                    mass_updated = false
                    if _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                        inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                        mass_updated = true
                    end
                    push!(
                        mass_adaptation_windows,
                        _mass_adaptation_window_summary(
                            warmup_schedule,
                            mass_window_index,
                            variance_state,
                            inverse_mass_matrix,
                            mass_updated,
                        ),
                    )
                    mass_window_index += 1
                    if mass_window_index <= length(warmup_schedule.slow_window_ends)
                        variance_state = _running_variance_state(
                            num_params,
                            _warmup_window_length(warmup_schedule, mass_window_index),
                        )
                    else
                        variance_state = _running_variance_state(num_params)
                    end
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
                if adapt_mass_matrix && _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                    inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                end
            end
        else
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] = transform_to_constrained(model, position)
            logjoint_values[sample_index] = current_logjoint
            acceptance_stats[sample_index] = accept_prob
            energies[sample_index] = sample_energy
            energy_errors[sample_index] = energy_error
            accepted[sample_index] = accepted_step
            divergent[sample_index] = divergent_step
        end
    end

    return HMCChain(
        :hmc,
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        acceptance_stats,
        energies,
        energy_errors,
        accepted,
        divergent,
        hmc_step_size,
        1 ./ inverse_mass_matrix,
        num_leapfrog_steps,
        0,
        zeros(Int, num_samples),
        fill(num_leapfrog_steps, num_samples),
        hmc_target_accept,
        mass_adaptation_windows,
    )
end

function nuts(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    max_tree_depth::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    max_delta_energy::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    num_params = parametercount(parameterlayout(model))
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

    position = _initial_hmc_position(model, args, constraints, initial_params, rng)
    current_logjoint = logjoint_unconstrained(model, position, args, constraints)
    isfinite(current_logjoint) ||
        throw(ArgumentError("initial NUTS parameters produced a non-finite unconstrained logjoint"))
    gradient_cache = _logjoint_gradient_cache(model, position, args, constraints)
    current_gradient = copy(_logjoint_gradient!(gradient_cache, position))
    all(isfinite, current_gradient) ||
        throw(ArgumentError("initial NUTS parameters produced a non-finite unconstrained gradient"))

    unconstrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    constrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    logjoint_values = Vector{Float64}(undef, num_samples)
    acceptance_stats = Vector{Float64}(undef, num_samples)
    energies = Vector{Float64}(undef, num_samples)
    energy_errors = Vector{Float64}(undef, num_samples)
    accepted = falses(num_samples)
    divergent = falses(num_samples)
    tree_depths = zeros(Int, num_samples)
    integration_steps_per_sample = zeros(Int, num_samples)
    total_iterations = num_warmup + num_samples
    nuts_step_size = Float64(step_size)
    nuts_target_accept = Float64(target_accept)
    nuts_max_delta_energy = Float64(max_delta_energy)
    inverse_mass_matrix = ones(num_params)
    warmup_schedule = _warmup_schedule(num_warmup)
    mass_window_index = 1
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        nuts_step_size = _find_reasonable_step_size(
            model,
            position,
            current_logjoint,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            nuts_step_size,
            rng,
        )
    end
    dual_state = _dual_averaging_state(nuts_step_size, nuts_target_accept)
    variance_state = _running_variance_state(
        num_params,
        isempty(warmup_schedule.slow_window_ends) ? (_RUNNING_VARIANCE_CLIP_START + 16) :
            _warmup_window_length(warmup_schedule, mass_window_index),
    )
    mass_adaptation_windows = HMCMassAdaptationWindowSummary[]

    sample_index = 0
    for iteration in 1:total_iterations
        proposal, accept_stat, tree_depth, integration_steps_used, proposal_energy, energy_error, divergent_step, moved_step =
            _nuts_proposal(
                model,
                position,
                current_logjoint,
                current_gradient,
                gradient_cache,
                inverse_mass_matrix,
                args,
                constraints,
                nuts_step_size,
                max_tree_depth,
                nuts_max_delta_energy,
                rng,
            )

        if moved_step
            position = proposal.position
            current_logjoint = proposal.logjoint
            current_gradient = proposal.gradient
        end

        if iteration <= num_warmup
            if adapt_step_size
                nuts_step_size = _update_step_size!(dual_state, accept_stat)
            end

            if adapt_mass_matrix &&
               mass_window_index <= length(warmup_schedule.slow_window_ends) &&
               iteration > warmup_schedule.initial_buffer
                _update_running_variance!(
                    variance_state,
                    position,
                    _mass_adaptation_weight(variance_state, false, accept_stat, divergent_step),
                )
                if iteration == warmup_schedule.slow_window_ends[mass_window_index]
                    mass_updated = false
                    if _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                        inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                        mass_updated = true
                    end
                    push!(
                        mass_adaptation_windows,
                        _mass_adaptation_window_summary(
                            warmup_schedule,
                            mass_window_index,
                            variance_state,
                            inverse_mass_matrix,
                            mass_updated,
                        ),
                    )
                    mass_window_index += 1
                    if mass_window_index <= length(warmup_schedule.slow_window_ends)
                        variance_state = _running_variance_state(
                            num_params,
                            _warmup_window_length(warmup_schedule, mass_window_index),
                        )
                    else
                        variance_state = _running_variance_state(num_params)
                    end
                    if adapt_step_size && iteration < num_warmup
                        nuts_step_size = _find_reasonable_step_size(
                            model,
                            position,
                            current_logjoint,
                            gradient_cache,
                            inverse_mass_matrix,
                            args,
                            constraints,
                            nuts_step_size,
                            rng,
                        )
                        dual_state = _dual_averaging_state(nuts_step_size, nuts_target_accept)
                    end
                end
            end

            if iteration == num_warmup
                if adapt_step_size
                    nuts_step_size = _final_step_size(dual_state)
                end
                if adapt_mass_matrix && _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                    inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                end
            end
        else
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] = transform_to_constrained(model, position)
            logjoint_values[sample_index] = current_logjoint
            acceptance_stats[sample_index] = accept_stat
            energies[sample_index] = proposal_energy
            energy_errors[sample_index] = energy_error
            accepted[sample_index] = moved_step
            divergent[sample_index] = divergent_step
            tree_depths[sample_index] = tree_depth
            integration_steps_per_sample[sample_index] = integration_steps_used
        end
    end

    return HMCChain(
        :nuts,
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        acceptance_stats,
        energies,
        energy_errors,
        accepted,
        divergent,
        nuts_step_size,
        1 ./ inverse_mass_matrix,
        0,
        max_tree_depth,
        tree_depths,
        integration_steps_per_sample,
        nuts_target_accept,
        mass_adaptation_windows,
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

function nuts_chains(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    max_tree_depth::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    max_delta_energy::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    _validate_hmc_chains_arguments(num_chains, "NUTS")
    num_params = parametercount(parameterlayout(model))
    seeds = rand(rng, UInt, num_chains)
    chains = Vector{HMCChain}(undef, num_chains)

    for chain_index in 1:num_chains
        chain_rng = MersenneTwister(seeds[chain_index])
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, num_chains)
        chains[chain_index] = nuts(
            model,
            args,
            constraints;
            num_samples=num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            initial_params=chain_initial_params,
            target_accept=target_accept,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            find_reasonable_step_size=find_reasonable_step_size,
            max_delta_energy=max_delta_energy,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            rng=chain_rng,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

function batched_nuts(
    model::TeaModel,
    args=(),
    constraints=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    max_tree_depth::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    max_delta_energy::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    num_params = parametercount(parameterlayout(model))
    _validate_batched_nuts_arguments(
        num_chains,
        num_params,
        num_samples,
        num_warmup,
        step_size,
        max_tree_depth,
        target_accept,
        max_delta_energy,
        mass_matrix_regularization,
        mass_matrix_min_samples,
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
    workspace = BatchedNUTSWorkspace(model, position, batch_args, batch_constraints)
    current_logjoint = Vector{Float64}(undef, num_chains)
    current_gradient = workspace.current_gradient
    _, gradient = _batched_logjoint_and_gradient_unconstrained!(
        current_logjoint,
        workspace.gradient_cache,
        position,
    )
    copyto!(current_gradient, gradient)
    all(isfinite, current_logjoint) ||
        throw(ArgumentError("initial batched NUTS parameters produced a non-finite unconstrained logjoint"))
    all(isfinite, current_gradient) ||
        throw(ArgumentError("initial batched NUTS parameters produced a non-finite unconstrained gradient"))

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    logjoint_values = Matrix{Float64}(undef, num_samples, num_chains)
    acceptance_stats = Matrix{Float64}(undef, num_samples, num_chains)
    energies = Matrix{Float64}(undef, num_samples, num_chains)
    energy_errors = Matrix{Float64}(undef, num_samples, num_chains)
    accepted = falses(num_samples, num_chains)
    divergent = falses(num_samples, num_chains)
    tree_depths = Matrix{Int}(undef, num_samples, num_chains)
    integration_steps_values = Matrix{Int}(undef, num_samples, num_chains)
    total_iterations = num_warmup + num_samples
    nuts_step_size = Float64(step_size)
    nuts_target_accept = Float64(target_accept)
    nuts_max_delta_energy = Float64(max_delta_energy)
    inverse_mass_matrix = ones(num_params)
    warmup_schedule = _warmup_schedule(num_warmup)
    mass_window_index = 1
    step_size_workspace = nothing
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        step_size_workspace = BatchedHMCWorkspace(model, position, batch_args, batch_constraints, inverse_mass_matrix)
        nuts_step_size = _find_reasonable_batched_step_size(
            step_size_workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            nuts_step_size,
            nuts_max_delta_energy,
            rng,
        )
    end
    dual_state = _dual_averaging_state(nuts_step_size, nuts_target_accept)
    variance_state = _running_variance_state(
        num_params,
        isempty(warmup_schedule.slow_window_ends) ? (_RUNNING_VARIANCE_CLIP_START + 16) :
            _warmup_window_length(warmup_schedule, mass_window_index),
    )
    mass_adaptation_windows = HMCMassAdaptationWindowSummary[]

    sample_index = 0
    for iteration in 1:total_iterations
        _batched_nuts_proposals!(
            workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            nuts_step_size,
            max_tree_depth,
            nuts_max_delta_energy,
            rng,
        )

        for chain_index in 1:num_chains
            if workspace.control.accepted_step[chain_index]
                copyto!(view(position, :, chain_index), view(workspace.proposal_position, :, chain_index))
                copyto!(view(current_gradient, :, chain_index), view(workspace.proposal_gradient, :, chain_index))
                current_logjoint[chain_index] = workspace.proposed_logjoint[chain_index]
            end
        end

        if iteration <= num_warmup
            if adapt_step_size
                nuts_step_size = _update_step_size!(
                    dual_state,
                    _mean_batched_adaptation_probability(workspace.accept_prob, workspace.control.divergent_step),
                )
            end

            if adapt_mass_matrix &&
               mass_window_index <= length(warmup_schedule.slow_window_ends) &&
               iteration > warmup_schedule.initial_buffer
                @inbounds for chain_index in 1:num_chains
                    workspace.mass_adaptation_weights[chain_index] = _mass_adaptation_weight(
                        variance_state,
                        false,
                        workspace.accept_prob[chain_index],
                        workspace.control.divergent_step[chain_index],
                    )
                end
                _update_running_variance!(
                    variance_state,
                    position,
                    workspace.mass_adaptation_weights,
                )
                if iteration == warmup_schedule.slow_window_ends[mass_window_index]
                    mass_updated = false
                    if _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                        inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                        mass_updated = true
                    end
                    push!(
                        mass_adaptation_windows,
                        _mass_adaptation_window_summary(
                            warmup_schedule,
                            mass_window_index,
                            variance_state,
                            inverse_mass_matrix,
                            mass_updated,
                        ),
                    )
                    mass_window_index += 1
                    if mass_window_index <= length(warmup_schedule.slow_window_ends)
                        variance_state = _running_variance_state(
                            num_params,
                            _warmup_window_length(warmup_schedule, mass_window_index),
                        )
                    else
                        variance_state = _running_variance_state(num_params)
                    end
                    if adapt_step_size && iteration < num_warmup
                        if isnothing(step_size_workspace)
                            step_size_workspace = BatchedHMCWorkspace(model, position, batch_args, batch_constraints, inverse_mass_matrix)
                        end
                        nuts_step_size = _find_reasonable_batched_step_size(
                            step_size_workspace,
                            model,
                            position,
                            current_logjoint,
                            current_gradient,
                            inverse_mass_matrix,
                            batch_args,
                            batch_constraints,
                            nuts_step_size,
                            nuts_max_delta_energy,
                            rng,
                        )
                        dual_state = _dual_averaging_state(nuts_step_size, nuts_target_accept)
                    end
                end
            end

            if iteration == num_warmup
                if adapt_step_size
                    nuts_step_size = _final_step_size(dual_state)
                end
                if adapt_mass_matrix && _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                    inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                end
            end
        else
            sample_index += 1
            for chain_index in 1:num_chains
                copyto!(view(unconstrained_samples, :, sample_index, chain_index), view(position, :, chain_index))
                _transform_to_constrained!(
                    view(workspace.constrained_position, :, chain_index),
                    model,
                    view(position, :, chain_index),
                )
                copyto!(
                    view(constrained_samples, :, sample_index, chain_index),
                    view(workspace.constrained_position, :, chain_index),
                )
                logjoint_values[sample_index, chain_index] = current_logjoint[chain_index]
                acceptance_stats[sample_index, chain_index] = workspace.accept_prob[chain_index]
                energies[sample_index, chain_index] = workspace.proposed_energy[chain_index]
                energy_errors[sample_index, chain_index] = workspace.energy_error[chain_index]
                accepted[sample_index, chain_index] = workspace.control.accepted_step[chain_index]
                divergent[sample_index, chain_index] = workspace.control.divergent_step[chain_index]
                tree_depths[sample_index, chain_index] = workspace.control.tree_depths[chain_index]
                integration_steps_values[sample_index, chain_index] = workspace.control.integration_steps[chain_index]
            end
        end
    end

    mass_matrix = 1 ./ inverse_mass_matrix
    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index in 1:num_chains
        chains[chain_index] = HMCChain(
            :nuts,
            model,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            unconstrained_samples[:, :, chain_index],
            constrained_samples[:, :, chain_index],
            vec(logjoint_values[:, chain_index]),
            vec(acceptance_stats[:, chain_index]),
            vec(energies[:, chain_index]),
            vec(energy_errors[:, chain_index]),
            vec(accepted[:, chain_index]),
            vec(divergent[:, chain_index]),
            nuts_step_size,
            copy(mass_matrix),
            0,
            max_tree_depth,
            vec(tree_depths[:, chain_index]),
            vec(integration_steps_values[:, chain_index]),
            nuts_target_accept,
            copy(mass_adaptation_windows),
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
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
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
        mass_matrix_regularization,
        mass_matrix_min_samples,
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
    inverse_mass_matrix = ones(num_params)
    workspace = BatchedHMCWorkspace(model, position, batch_args, batch_constraints, inverse_mass_matrix)
    current_logjoint = Vector{Float64}(undef, num_chains)
    current_gradient = workspace.current_gradient
    _, gradient = _batched_logjoint_and_gradient_unconstrained!(
        current_logjoint,
        workspace.gradient_cache,
        position,
    )
    copyto!(current_gradient, gradient)
    all(isfinite, current_logjoint) ||
        throw(ArgumentError("initial batched HMC parameters produced a non-finite unconstrained logjoint"))
    all(isfinite, current_gradient) ||
        throw(ArgumentError("initial batched HMC parameters produced a non-finite unconstrained gradient"))

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    logjoint_values = Matrix{Float64}(undef, num_samples, num_chains)
    acceptance_stats = Matrix{Float64}(undef, num_samples, num_chains)
    energies = Matrix{Float64}(undef, num_samples, num_chains)
    energy_errors = Matrix{Float64}(undef, num_samples, num_chains)
    accepted = falses(num_samples, num_chains)
    divergent = falses(num_samples, num_chains)
    total_iterations = num_warmup + num_samples
    hmc_step_size = Float64(step_size)
    hmc_target_accept = Float64(target_accept)
    hmc_divergence_threshold = Float64(divergence_threshold)
    warmup_schedule = _warmup_schedule(num_warmup)
    mass_window_index = 1
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        hmc_step_size = _find_reasonable_batched_step_size(
            workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            hmc_step_size,
            hmc_divergence_threshold,
            rng,
        )
    end
    dual_state = _dual_averaging_state(hmc_step_size, hmc_target_accept)
    variance_state = _running_variance_state(
        num_params,
        isempty(warmup_schedule.slow_window_ends) ? (_RUNNING_VARIANCE_CLIP_START + 16) :
            _warmup_window_length(warmup_schedule, mass_window_index),
    )
    mass_adaptation_windows = HMCMassAdaptationWindowSummary[]

    sample_index = 0
    for iteration in 1:total_iterations
        _update_sqrt_inverse_mass_matrix!(workspace.sqrt_inverse_mass_matrix, inverse_mass_matrix)
        _sample_batched_momentum!(workspace.momentum, rng, workspace.sqrt_inverse_mass_matrix)
        proposal_position, proposal_momentum, proposed_logjoint, proposal_gradient, valid = _batched_leapfrog!(
            workspace,
            model,
            position,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            hmc_step_size,
            num_leapfrog_steps,
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
        energy_error = workspace.energy_error
        fill!(energy_error, Inf)
        divergent_step = workspace.divergent_step
        fill!(divergent_step, true)

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

        accept_prob = _batched_acceptance_probability!(workspace.accept_prob, log_accept_ratio)
        accepted_step = workspace.accepted_step
        fill!(accepted_step, false)
        for chain_index in 1:num_chains
            if valid[chain_index] && log(rand(rng)) < min(0.0, log_accept_ratio[chain_index])
                copyto!(view(position, :, chain_index), view(proposal_position, :, chain_index))
                copyto!(view(current_gradient, :, chain_index), view(proposal_gradient, :, chain_index))
                current_logjoint[chain_index] = proposed_logjoint[chain_index]
                accepted_step[chain_index] = true
            end
        end

        if iteration <= num_warmup
            if adapt_step_size
                hmc_step_size = _update_step_size!(
                    dual_state,
                    _mean_batched_adaptation_probability(accept_prob, divergent_step),
                )
            end

            if adapt_mass_matrix &&
               mass_window_index <= length(warmup_schedule.slow_window_ends) &&
               iteration > warmup_schedule.initial_buffer
                _mass_adaptation_weights!(
                    variance_state,
                    workspace.mass_adaptation_weights,
                    accepted_step,
                    accept_prob,
                    divergent_step,
                )
                _update_running_variance!(
                    variance_state,
                    position,
                    workspace.mass_adaptation_weights,
                )
                if iteration == warmup_schedule.slow_window_ends[mass_window_index]
                    mass_updated = false
                    if _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                        inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                        mass_updated = true
                    end
                    push!(
                        mass_adaptation_windows,
                        _mass_adaptation_window_summary(
                            warmup_schedule,
                            mass_window_index,
                            variance_state,
                            inverse_mass_matrix,
                            mass_updated,
                        ),
                    )
                    mass_window_index += 1
                    if mass_window_index <= length(warmup_schedule.slow_window_ends)
                        variance_state = _running_variance_state(
                            num_params,
                            _warmup_window_length(warmup_schedule, mass_window_index),
                        )
                    else
                        variance_state = _running_variance_state(num_params)
                    end
                    if adapt_step_size && iteration < num_warmup
                        hmc_step_size = _find_reasonable_batched_step_size(
                            workspace,
                            model,
                            position,
                            current_logjoint,
                            current_gradient,
                            inverse_mass_matrix,
                            batch_args,
                            batch_constraints,
                            hmc_step_size,
                            hmc_divergence_threshold,
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
                if adapt_mass_matrix && _running_variance_effective_count(variance_state) >= mass_matrix_min_samples
                    inverse_mass_matrix = _inverse_mass_matrix(variance_state, Float64(mass_matrix_regularization))
                end
            end
        end

        if iteration > num_warmup
            sample_index += 1
            for chain_index in 1:num_chains
                copyto!(view(unconstrained_samples, :, sample_index, chain_index), view(position, :, chain_index))
                _transform_to_constrained!(
                    view(workspace.constrained_position, :, chain_index),
                    model,
                    view(position, :, chain_index),
                )
                copyto!(
                    view(constrained_samples, :, sample_index, chain_index),
                    view(workspace.constrained_position, :, chain_index),
                )
                logjoint_values[sample_index, chain_index] = current_logjoint[chain_index]
                acceptance_stats[sample_index, chain_index] = accept_prob[chain_index]
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
            :hmc,
            model,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            unconstrained_samples[:, :, chain_index],
            constrained_samples[:, :, chain_index],
            vec(logjoint_values[:, chain_index]),
            vec(acceptance_stats[:, chain_index]),
            vec(energies[:, chain_index]),
            vec(energy_errors[:, chain_index]),
            vec(accepted[:, chain_index]),
            vec(divergent[:, chain_index]),
            hmc_step_size,
            copy(mass_matrix),
            num_leapfrog_steps,
            0,
            zeros(Int, num_samples),
            fill(num_leapfrog_steps, num_samples),
            hmc_target_accept,
            copy(mass_adaptation_windows),
        )
    end

    return HMCChains(model, args, constraints, chains)
end
