function _backend_dirichlet_logpdf(alpha, x)
    length(alpha) == length(x) ||
        throw(ArgumentError("dirichlet requires matching concentration and value lengths"))
    length(alpha) >= 2 || throw(ArgumentError("dirichlet requires at least two dimensions"))

    first_alpha = float(first(alpha))
    total_alpha = zero(first_alpha)
    accumulator = zero(first_alpha)
    for alpha_value in alpha
        alpha_float = float(alpha_value)
        alpha_float > zero(alpha_float) || throw(ArgumentError("dirichlet requires alpha > 0 in every dimension"))
        total_alpha += alpha_float
        accumulator -= loggamma(alpha_float)
    end
    accumulator += loggamma(total_alpha)

    total = zero(total_alpha)
    for (value, alpha_value) in zip(x, alpha)
        value_float = float(value)
        value_float > zero(value_float) || return oftype(accumulator, -Inf)
        total += value_float
        accumulator += (float(alpha_value) - 1) * log(value_float)
    end
    abs(total - one(total)) <= sqrt(eps(float(total))) * length(x) * 16 || return oftype(accumulator, -Inf)
    return accumulator
end

function _score_backend_step!(
    step::BackendDirichletChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    alpha = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.alpha)...)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_dirichlet_logpdf(alpha, value)
end

function _score_backend_step!(
    step::BackendDirichletChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = [_batched_numeric_scratch!(env, index) for index in 1:step.value_length]
    alpha_values = [_batched_numeric_scratch!(env, step.value_length + index) for index in 1:step.value_length]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    for component_index in 1:step.value_length
        _eval_backend_numeric_expr!(
            alpha_values[component_index],
            env,
            step.alpha[component_index],
            2 * step.value_length + 1,
        )
    end

    for batch_index in 1:env.batch_size
        total_alpha = 0.0
        accumulator = 0.0
        for component_index in 1:step.value_length
            alpha = alpha_values[component_index][batch_index]
            alpha > 0 || throw(ArgumentError("dirichlet requires alpha > 0 in every dimension"))
            total_alpha += alpha
            accumulator -= loggamma(alpha)
        end
        accumulator += loggamma(total_alpha)

        total = 0.0
        valid = true
        for component_index in 1:step.value_length
            value = choice_values[component_index][batch_index]
            value > 0 || begin
                valid = false
                break
            end
            total += value
            accumulator += (alpha_values[component_index][batch_index] - 1) * log(value)
        end

        totals[batch_index] += valid && abs(total - 1) <= sqrt(eps(total)) * step.value_length * 16 ? accumulator : -Inf
    end

    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end
