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

