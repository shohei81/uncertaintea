const BACKEND_GRADIENT_SUPPORTED_PRIMITIVES =
    Set([:+, :-, :*, :/, :^, :%, :exp, :log, :log1p, :sqrt, :abs, :min, :max, :clamp])

_backend_gradient_supported_expr(::BackendLiteralExpr) = true
_backend_gradient_supported_expr(::BackendSlotExpr) = true
_backend_gradient_supported_expr(::BackendTupleExpr) = false

function _backend_gradient_supported_constant_expr(expr::BackendLiteralExpr)
    return expr.value isa Real && !(expr.value isa Bool)
end

_backend_gradient_supported_constant_expr(expr::AbstractBackendExpr) = false

function _backend_gradient_supported_expr(expr::BackendPrimitiveExpr)
    if expr.op === :^ || expr.op === :%
        length(expr.arguments) == 2 || return false
        return _backend_gradient_supported_expr(expr.arguments[1]) &&
               _backend_gradient_supported_constant_expr(expr.arguments[2])
    elseif expr.op === :clamp
        length(expr.arguments) == 3 || return false
    end
    expr.op in BACKEND_GRADIENT_SUPPORTED_PRIMITIVES || return false
    return all(_backend_gradient_supported_expr, expr.arguments)
end

function _backend_gradient_supported_expr(expr::BackendBlockExpr)
    return all(_backend_gradient_supported_expr, expr.arguments)
end

function _backend_gradient_supported_step(step::BackendNormalChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendLognormalChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendExponentialChoicePlanStep)
    return _backend_gradient_supported_expr(step.rate)
end

function _backend_gradient_supported_step(step::BackendBernoulliChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendPoissonChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.lambda)
end

function _backend_gradient_supported_step(step::BackendDeterministicPlanStep, numeric_slots::BitVector)
    return numeric_slots[step.binding_slot] ? _backend_gradient_supported_expr(step.expr) : true
end

function _backend_gradient_supported_step(step::BackendLoopPlanStep, numeric_slots::BitVector)
    return all(inner -> _backend_gradient_supported_step(inner, numeric_slots), step.body)
end

_backend_gradient_supported_step(step::BackendNormalChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendLognormalChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendExponentialChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBernoulliChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendPoissonChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)

function _backend_gradient_supported(plan::BackendExecutionPlan)
    return all(step -> _backend_gradient_supported_step(step, plan.numeric_slots), plan.steps)
end

function _batched_backend_gradient_scratch!(cache::BatchedBackendGradientCache, depth::Int)
    depth > 0 || throw(ArgumentError("batched backend gradient scratch depth must be positive"))
    parameter_count = size(cache.slot_gradients, 1)
    batch_size = size(cache.slot_gradients, 3)
    while length(cache.gradient_scratch) < depth
        push!(cache.gradient_scratch, Matrix{Float64}(undef, parameter_count, batch_size))
    end
    buffer = cache.gradient_scratch[depth]
    if size(buffer) != (parameter_count, batch_size)
        buffer = Matrix{Float64}(undef, parameter_count, batch_size)
        cache.gradient_scratch[depth] = buffer
    end
    return buffer
end

function _zero_gradient!(destination::AbstractMatrix{Float64})
    fill!(destination, 0.0)
    return destination
end

function _copy_slot_gradient!(
    destination::AbstractMatrix{Float64},
    slot_gradients::Array{Float64,3},
    slot::Int,
)
    for batch_index in axes(destination, 2), parameter_index in axes(destination, 1)
        destination[parameter_index, batch_index] = slot_gradients[parameter_index, slot, batch_index]
    end
    return destination
end

function _store_slot_gradient!(
    slot_gradients::Array{Float64,3},
    slot::Int,
    source::AbstractMatrix{Float64},
)
    for batch_index in axes(source, 2), parameter_index in axes(source, 1)
        slot_gradients[parameter_index, slot, batch_index] = source[parameter_index, batch_index]
    end
    return slot_gradients
end

function _fill_choice_gradient!(
    destination::AbstractMatrix{Float64},
    parameter_slot::Union{Nothing,Int},
)
    fill!(destination, 0.0)
    isnothing(parameter_slot) && return destination
    for batch_index in axes(destination, 2)
        destination[parameter_slot, batch_index] = 1.0
    end
    return destination
end

function _apply_backend_numeric_gradient_unary!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    env::BatchedPlanEnvironment{Float64},
    op::Symbol,
)
    if op === :-
        for batch_index in eachindex(values)
            values[batch_index] = -values[batch_index]
        end
        for batch_index in axes(gradients, 2), parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] = -gradients[parameter_index, batch_index]
        end
    elseif op === :exp
        for batch_index in eachindex(values)
            value = exp(values[batch_index])
            values[batch_index] = value
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] *= value
            end
        end
    elseif op === :log
        for batch_index in eachindex(values)
            value = values[batch_index]
            values[batch_index] = log(value)
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] /= value
            end
        end
    elseif op === :log1p
        for batch_index in eachindex(values)
            value = values[batch_index]
            values[batch_index] = log1p(value)
            denominator = 1 + value
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] /= denominator
            end
        end
    elseif op === :sqrt
        for batch_index in eachindex(values)
            root = sqrt(values[batch_index])
            values[batch_index] = root
            factor = 2 * root
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] /= factor
            end
        end
    elseif op === :abs
        for batch_index in eachindex(values)
            original = values[batch_index]
            values[batch_index] = abs(original)
            factor = original > 0 ? 1.0 : (original < 0 ? -1.0 : 0.0)
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] *= factor
            end
        end
    else
        _backend_numeric_error(env, "batched backend gradient does not support unary primitive `$(op)`")
    end
    return values, gradients
end

function _apply_backend_numeric_gradient_binary!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    env::BatchedPlanEnvironment{Float64},
    op::Symbol,
    rhs_values::AbstractVector{Float64},
    rhs_gradients::AbstractMatrix{Float64},
)
    if op === :+
        for batch_index in eachindex(values, rhs_values)
            values[batch_index] += rhs_values[batch_index]
        end
        for batch_index in axes(gradients, 2), parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += rhs_gradients[parameter_index, batch_index]
        end
    elseif op === :-
        for batch_index in eachindex(values, rhs_values)
            values[batch_index] -= rhs_values[batch_index]
        end
        for batch_index in axes(gradients, 2), parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] -= rhs_gradients[parameter_index, batch_index]
        end
    elseif op === :*
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] =
                    gradients[parameter_index, batch_index] * rhs_value +
                    lhs_value * rhs_gradients[parameter_index, batch_index]
            end
            values[batch_index] = lhs_value * rhs_value
        end
    elseif op === :/
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            denominator = rhs_value * rhs_value
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] =
                    (gradients[parameter_index, batch_index] * rhs_value -
                     lhs_value * rhs_gradients[parameter_index, batch_index]) / denominator
            end
            values[batch_index] = lhs_value / rhs_value
        end
    elseif op === :^
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            exponent = rhs_values[batch_index]
            power = lhs_value ^ exponent
            factor = exponent * (lhs_value ^ (exponent - 1))
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] *= factor
            end
            values[batch_index] = power
        end
    elseif op === :%
        for batch_index in eachindex(values, rhs_values)
            values[batch_index] = values[batch_index] % rhs_values[batch_index]
        end
    elseif op === :min
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            if lhs_value < rhs_value
                values[batch_index] = lhs_value
            elseif lhs_value > rhs_value
                values[batch_index] = rhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] = rhs_gradients[parameter_index, batch_index]
                end
            else
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] =
                        0.5 * (gradients[parameter_index, batch_index] + rhs_gradients[parameter_index, batch_index])
                end
            end
        end
    elseif op === :max
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            if lhs_value > rhs_value
                values[batch_index] = lhs_value
            elseif lhs_value < rhs_value
                values[batch_index] = rhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] = rhs_gradients[parameter_index, batch_index]
                end
            else
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] =
                        0.5 * (gradients[parameter_index, batch_index] + rhs_gradients[parameter_index, batch_index])
                end
            end
        end
    else
        _backend_numeric_error(env, "batched backend gradient does not support binary primitive `$(op)`")
    end
    return values, gradients
end

function _apply_backend_numeric_gradient_ternary!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    env::BatchedPlanEnvironment{Float64},
    op::Symbol,
    middle_values::AbstractVector{Float64},
    middle_gradients::AbstractMatrix{Float64},
    rhs_values::AbstractVector{Float64},
    rhs_gradients::AbstractMatrix{Float64},
)
    if op === :clamp
        for batch_index in eachindex(values, middle_values, rhs_values)
            lhs_value = values[batch_index]
            middle_value = middle_values[batch_index]
            rhs_value = rhs_values[batch_index]
            if lhs_value < middle_value
                values[batch_index] = middle_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] = middle_gradients[parameter_index, batch_index]
                end
            elseif lhs_value > rhs_value
                values[batch_index] = rhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] = rhs_gradients[parameter_index, batch_index]
                end
            elseif lhs_value == middle_value && lhs_value == rhs_value
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] = (
                        gradients[parameter_index, batch_index] +
                        middle_gradients[parameter_index, batch_index] +
                        rhs_gradients[parameter_index, batch_index]
                    ) / 3
                end
            elseif lhs_value == middle_value
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] =
                        0.5 * (gradients[parameter_index, batch_index] + middle_gradients[parameter_index, batch_index])
                end
            elseif lhs_value == rhs_value
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] =
                        0.5 * (gradients[parameter_index, batch_index] + rhs_gradients[parameter_index, batch_index])
                end
            else
                values[batch_index] = lhs_value
            end
        end
    else
        _backend_numeric_error(env, "batched backend gradient does not support ternary primitive `$(op)`")
    end
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendLiteralExpr,
    depth::Int=1,
)
    fill!(values, _require_numeric_value(env, expr.value, "batched backend numeric expression"))
    fill!(gradients, 0.0)
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendSlotExpr,
    depth::Int=1,
)
    env.assigned[expr.slot] || throw(BatchedBackendFallback("environment slot $(expr.slot) is not assigned"))
    if env.numeric_slots[expr.slot]
        copyto!(values, view(env.numeric_values, expr.slot, :))
        _copy_slot_gradient!(gradients, cache.slot_gradients, expr.slot)
        return values, gradients
    elseif env.index_slots[expr.slot]
        for batch_index in eachindex(values)
            values[batch_index] = Float64(env.index_values[expr.slot, batch_index])
        end
        fill!(gradients, 0.0)
        return values, gradients
    end
    _backend_numeric_error(env, "batched backend gradient slot $(expr.slot) is not numeric")
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendPrimitiveExpr,
    depth::Int=1,
)
    expr.op in BACKEND_GRADIENT_SUPPORTED_PRIMITIVES ||
        _backend_numeric_error(env, "batched backend gradient does not support primitive `$(expr.op)`")
    isempty(expr.arguments) && _backend_numeric_error(env, "batched backend gradient primitive requires arguments")

    _eval_backend_numeric_expr_and_gradient!(values, gradients, cache, env, first(expr.arguments), depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_numeric_gradient_unary!(values, gradients, env, expr.op)
    elseif expr.op === :clamp
        length(expr.arguments) == 3 ||
            _backend_numeric_error(env, "batched backend gradient `clamp` expects exactly 3 arguments")
        middle_values = _batched_numeric_scratch!(env, depth)
        middle_gradients = _batched_backend_gradient_scratch!(cache, depth)
        rhs_values = _batched_numeric_scratch!(env, depth + 1)
        rhs_gradients = _batched_backend_gradient_scratch!(cache, depth + 1)
        _eval_backend_numeric_expr_and_gradient!(middle_values, middle_gradients, cache, env, expr.arguments[2], depth + 2)
        _eval_backend_numeric_expr_and_gradient!(rhs_values, rhs_gradients, cache, env, expr.arguments[3], depth + 2)
        return _apply_backend_numeric_gradient_ternary!(
            values,
            gradients,
            env,
            expr.op,
            middle_values,
            middle_gradients,
            rhs_values,
            rhs_gradients,
        )
    end

    temp_values = _batched_numeric_scratch!(env, depth)
    temp_gradients = _batched_backend_gradient_scratch!(cache, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_numeric_expr_and_gradient!(temp_values, temp_gradients, cache, env, argument, depth + 1)
        _apply_backend_numeric_gradient_binary!(values, gradients, env, expr.op, temp_values, temp_gradients)
    end
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendTupleExpr,
    depth::Int=1,
)
    _backend_numeric_error(env, "batched backend gradient expression cannot be a tuple")
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{Float64},
    expr::BackendBlockExpr,
    depth::Int=1,
)
    for argument in expr.arguments
        _eval_backend_numeric_expr_and_gradient!(values, gradients, cache, env, argument, depth)
    end
    return values, gradients
end

function _set_numeric_binding!(
    env::BatchedPlanEnvironment{Float64},
    slot_gradients::Array{Float64,3},
    slot::Int,
    values::AbstractVector{Float64},
    gradients::AbstractMatrix{Float64},
)
    copyto!(view(env.numeric_values, slot, :), values)
    _store_slot_gradient!(slot_gradients, slot, gradients)
    env.assigned[slot] = true
    return env
end

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
