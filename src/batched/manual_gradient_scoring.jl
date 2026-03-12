function _score_backend_steps_and_gradient!(
    ::Tuple{},
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    return totals, gradients
end

function _score_backend_steps_and_gradient!(
    steps::Tuple,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    _score_backend_step_and_gradient!(first(steps), totals, gradients, cache, env, params, constraints)
    return _score_backend_steps_and_gradient!(Base.tail(steps), totals, gradients, cache, env, params, constraints)
end

function _accumulate_normal_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
    value_gradients::AbstractMatrix{Float64},
    mu_values::AbstractVector{Float64},
    mu_gradients::AbstractMatrix{Float64},
    sigma_values::AbstractVector{Float64},
    sigma_gradients::AbstractMatrix{Float64},
)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
        z = (value - mu) / sigma
        inv_sigma = 1 / sigma
        dvalue = -z * inv_sigma
        dmu = z * inv_sigma
        dsigma = (z * z - 1) * inv_sigma
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_lognormal_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
    value_gradients::AbstractMatrix{Float64},
    mu_values::AbstractVector{Float64},
    mu_gradients::AbstractMatrix{Float64},
    sigma_values::AbstractVector{Float64},
    sigma_gradients::AbstractMatrix{Float64},
)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_lognormal_logpdf(mu, sigma, value)
        if !(value > 0)
            continue
        end
        log_value = log(value)
        z = (log_value - mu) / sigma
        inv_sigma = 1 / sigma
        dvalue = (-(z * inv_sigma) - 1) / value
        dmu = z * inv_sigma
        dsigma = (z * z - 1) * inv_sigma
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_bernoulli_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    probability_values::AbstractVector{Float64},
    probability_gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
)
    for batch_index in eachindex(totals)
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_bernoulli_logpdf(probability, value)
        derivative = value != 0 ? 1 / probability : -1 / (1 - probability)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_exponential_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
    value_gradients::AbstractMatrix{Float64},
    rate_values::AbstractVector{Float64},
    rate_gradients::AbstractMatrix{Float64},
)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        rate = rate_values[batch_index]
        totals[batch_index] += _backend_exponential_logpdf(rate, value)
        if !(value >= 0)
            continue
        end
        dvalue = -rate
        drate = 1 / rate - value
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                drate * rate_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_gamma_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
    value_gradients::AbstractMatrix{Float64},
    shape_values::AbstractVector{Float64},
    shape_gradients::AbstractMatrix{Float64},
    rate_values::AbstractVector{Float64},
    rate_gradients::AbstractMatrix{Float64},
)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        shape = shape_values[batch_index]
        rate = rate_values[batch_index]
        totals[batch_index] += _backend_gamma_logpdf(shape, rate, value)
        if !(value > 0)
            continue
        end
        dvalue = (shape - 1) / value - rate
        dshape = log(rate) - digamma(shape) + log(value)
        drate = shape / rate - value
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dshape * shape_gradients[parameter_index, batch_index] +
                drate * rate_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_poisson_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    lambda_values::AbstractVector{Float64},
    lambda_gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
)
    for batch_index in eachindex(totals)
        lambda = lambda_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_poisson_logpdf(lambda, value)
        count = _poisson_count(value)
        isnothing(count) && continue
        derivative = count / lambda - 1
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * lambda_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_studentt_gradient!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    value_values::AbstractVector{Float64},
    value_gradients::AbstractMatrix{Float64},
    nu_values::AbstractVector{Float64},
    nu_gradients::AbstractMatrix{Float64},
    mu_values::AbstractVector{Float64},
    mu_gradients::AbstractMatrix{Float64},
    sigma_values::AbstractVector{Float64},
    sigma_gradients::AbstractMatrix{Float64},
)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        nu = nu_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_studentt_logpdf(nu, mu, sigma, value)
        z = (value - mu) / sigma
        denominator = nu + z * z
        dvalue = -((nu + 1) * z) / (sigma * denominator)
        dmu = -dvalue
        dsigma = (z * z - nu) / (sigma * denominator)
        dnu = 0.5 * (
            digamma((nu + 1) / 2) -
            digamma(nu / 2) -
            1 / nu -
            log1p((z * z) / nu) +
            ((nu + 1) * z * z) / (nu * denominator)
        )
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dnu * nu_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendNormalChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 5)
    _accumulate_normal_gradient!(totals, gradients, value_values, value_gradients, mu_values, mu_gradients, sigma_values, sigma_gradients)
    isnothing(step.binding_slot) || _set_numeric_binding!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLognormalChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 5)
    _accumulate_lognormal_gradient!(totals, gradients, value_values, value_gradients, mu_values, mu_gradients, sigma_values, sigma_gradients)
    isnothing(step.binding_slot) || _set_numeric_binding!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendExponentialChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    rate_values = _batched_numeric_scratch!(env, 1)
    rate_gradients = _batched_backend_gradient_scratch!(cache, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(rate_values, rate_gradients, cache, env, step.rate, 3)
    _accumulate_exponential_gradient!(totals, gradients, value_values, value_gradients, rate_values, rate_gradients)
    isnothing(step.binding_slot) || _set_numeric_binding!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendGammaChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    rate_values = _batched_numeric_scratch!(env, 2)
    rate_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, step.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(rate_values, rate_gradients, cache, env, step.rate, 5)
    _accumulate_gamma_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        shape_values,
        shape_gradients,
        rate_values,
        rate_gradients,
    )
    isnothing(step.binding_slot) || _set_numeric_binding!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBernoulliChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    isnothing(step.parameter_slot) || throw(BatchedBackendFallback("batched backend gradient does not support Bernoulli latent parameters"))
    value_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 2)
    _accumulate_bernoulli_gradient!(totals, gradients, probability_values, probability_gradients, value_values)

    if !isnothing(step.binding_slot)
        if env.numeric_slots[step.binding_slot]
            copyto!(view(env.numeric_values, step.binding_slot, :), value_values)
        elseif env.index_slots[step.binding_slot]
            for batch_index in 1:env.batch_size
                env.index_values[step.binding_slot, batch_index] = Int(round(value_values[batch_index]))
            end
        else
            values = env.generic_values[step.binding_slot]
            for batch_index in 1:env.batch_size
                values[batch_index] = value_values[batch_index]
            end
        end
        env.assigned[step.binding_slot] = true
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendStudentTChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    nu_values = _batched_numeric_scratch!(env, 1)
    nu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    mu_values = _batched_numeric_scratch!(env, 2)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 3)
    sigma_values = _batched_numeric_scratch!(env, 3)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(nu_values, nu_gradients, cache, env, step.nu, 5)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 6)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 7)
    _accumulate_studentt_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        nu_values,
        nu_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
    isnothing(step.binding_slot) || _set_numeric_binding!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendPoissonChoicePlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    isnothing(step.parameter_slot) || throw(BatchedBackendFallback("batched backend gradient does not support Poisson latent parameters"))
    value_values = env.observed_values
    lambda_values = _batched_numeric_scratch!(env, 1)
    lambda_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(lambda_values, lambda_gradients, cache, env, step.lambda, 2)
    _accumulate_poisson_gradient!(totals, gradients, lambda_values, lambda_gradients, value_values)

    if !isnothing(step.binding_slot)
        if env.numeric_slots[step.binding_slot]
            copyto!(view(env.numeric_values, step.binding_slot, :), value_values)
            fill!(view(cache.slot_gradients, :, step.binding_slot, :), 0.0)
        elseif env.index_slots[step.binding_slot]
            for batch_index in 1:env.batch_size
                env.index_values[step.binding_slot, batch_index] = Int(round(value_values[batch_index]))
            end
        else
            values = env.generic_values[step.binding_slot]
            for batch_index in 1:env.batch_size
                values[batch_index] = Int(round(value_values[batch_index]))
            end
        end
        env.assigned[step.binding_slot] = true
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendDeterministicPlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    if env.numeric_slots[step.binding_slot]
        values = view(env.numeric_values, step.binding_slot, :)
        slot_gradients = view(cache.slot_gradients, :, step.binding_slot, :)
        _eval_backend_numeric_expr_and_gradient!(values, slot_gradients, cache, env, step.expr)
        env.assigned[step.binding_slot] = true
        return totals, gradients
    end

    _score_backend_step!(step, totals, env, params, constraints)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLoopPlanStep,
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    params::AbstractMatrix{Float64},
    constraints,
)
    reference_iterable = _batched_index_iterable_reference(env, step.iterable)
    had_previous = env.assigned[step.iterator_slot]
    previous_value = had_previous ? copy(env.index_values[step.iterator_slot, :]) : Int[]

    for item in reference_iterable
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        _score_backend_steps_and_gradient!(step.body, totals, gradients, cache, env, params, constraints)
    end

    _batched_environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return totals, gradients
end

function _batched_backend_logjoint_and_gradient_unconstrained!(
    totals::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    model::TeaModel,
    cache::BatchedBackendGradientCache,
    params::AbstractMatrix,
)
    size(params, 1) == size(gradients, 1) ||
        throw(DimensionMismatch("expected $(size(gradients, 1)) parameters, got $(size(params, 1))"))
    size(params, 2) == size(gradients, 2) ||
        throw(DimensionMismatch("expected $(size(gradients, 2)) batch elements, got $(size(params, 2))"))

    workspace = cache.workspace
    env = _prepare_batched_environment!(workspace, cache.args, size(params, 2), Float64)
    fill!(totals, 0.0)
    fill!(gradients, 0.0)
    fill!(cache.slot_gradients, 0.0)

    layout = parameterlayout(model)
    constrained = _batched_constrained_buffer!(workspace, size(params, 1), size(params, 2), Float64)
    logabsdet = _batched_logabsdet_buffer!(workspace, size(params, 2), Float64)
    for slot in layout.slots
        slot_index = slot.index
        if slot.transform isa IdentityTransform
            for batch_index in 1:size(params, 2)
                constrained[slot_index, batch_index] = Float64(params[slot_index, batch_index])
            end
        elseif slot.transform isa LogTransform
            for batch_index in 1:size(params, 2)
                unconstrained_value = Float64(params[slot_index, batch_index])
                constrained_value = exp(unconstrained_value)
                constrained[slot_index, batch_index] = constrained_value
                logabsdet[batch_index] += unconstrained_value
            end
        else
            throw(BatchedBackendFallback("batched backend gradient does not support transform $(typeof(slot.transform))"))
        end
    end

    _score_backend_steps_and_gradient!(workspace.backend_plan.steps, totals, gradients, cache, env, constrained, cache.constraints)

    for slot in layout.slots
        if slot.transform isa IdentityTransform
            continue
        elseif slot.transform isa LogTransform
            slot_index = slot.index
            for batch_index in 1:size(params, 2)
                gradients[slot_index, batch_index] =
                    gradients[slot_index, batch_index] * constrained[slot_index, batch_index] + 1.0
            end
        end
    end

    for batch_index in eachindex(totals)
        totals[batch_index] += logabsdet[batch_index]
    end
    return totals, gradients
end
