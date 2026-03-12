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

function _backend_gradient_supported_step(step::BackendLaplaceChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.scale)
end

function _backend_gradient_supported_step(step::BackendLognormalChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendExponentialChoicePlanStep)
    return _backend_gradient_supported_expr(step.rate)
end

function _backend_gradient_supported_step(step::BackendGammaChoicePlanStep)
    return _backend_gradient_supported_expr(step.shape) && _backend_gradient_supported_expr(step.rate)
end

function _backend_gradient_constant_index_expr(expr::BackendLiteralExpr, numeric_slots::BitVector)
    return true
end

function _backend_gradient_constant_index_expr(expr::BackendSlotExpr, numeric_slots::BitVector)
    return !numeric_slots[expr.slot]
end

function _backend_gradient_constant_index_expr(expr::BackendPrimitiveExpr, numeric_slots::BitVector)
    return all(arg -> _backend_gradient_constant_index_expr(arg, numeric_slots), expr.arguments)
end

function _backend_gradient_constant_index_expr(expr::BackendBlockExpr, numeric_slots::BitVector)
    return all(arg -> _backend_gradient_constant_index_expr(arg, numeric_slots), expr.arguments)
end

_backend_gradient_constant_index_expr(expr::AbstractBackendExpr, numeric_slots::BitVector) = false

function _backend_gradient_supported_step(step::BackendInverseGammaChoicePlanStep)
    return _backend_gradient_supported_expr(step.shape) && _backend_gradient_supported_expr(step.scale)
end

function _backend_gradient_supported_step(step::BackendWeibullChoicePlanStep)
    return _backend_gradient_supported_expr(step.shape) && _backend_gradient_supported_expr(step.scale)
end

function _backend_gradient_supported_step(step::BackendBetaChoicePlanStep)
    return _backend_gradient_supported_expr(step.alpha) && _backend_gradient_supported_expr(step.beta)
end

function _backend_gradient_supported_step(step::BackendBernoulliChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendGeometricChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendBinomialChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendNegativeBinomialChoicePlanStep)
    return isnothing(step.parameter_slot) &&
           _backend_gradient_supported_expr(step.successes) &&
           _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendCategoricalChoicePlanStep)
    return isnothing(step.parameter_slot) && all(_backend_gradient_supported_expr, step.probabilities)
end

function _backend_gradient_supported_step(step::BackendPoissonChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.lambda)
end

function _backend_gradient_supported_step(step::BackendStudentTChoicePlanStep)
    return _backend_gradient_supported_expr(step.nu) &&
           _backend_gradient_supported_expr(step.mu) &&
           _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendMvNormalChoicePlanStep)
    return false
end

function _backend_gradient_supported_step(step::BackendDeterministicPlanStep, numeric_slots::BitVector)
    return numeric_slots[step.binding_slot] ? _backend_gradient_supported_expr(step.expr) : true
end

function _backend_gradient_supported_step(step::BackendLoopPlanStep, numeric_slots::BitVector)
    return all(inner -> _backend_gradient_supported_step(inner, numeric_slots), step.body)
end

_backend_gradient_supported_step(step::BackendNormalChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendLaplaceChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendLognormalChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendExponentialChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendGammaChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendInverseGammaChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendWeibullChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBetaChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBernoulliChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendGeometricChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBinomialChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step) &&
    _backend_gradient_constant_index_expr(step.trials, numeric_slots)
_backend_gradient_supported_step(step::BackendNegativeBinomialChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendCategoricalChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendPoissonChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendStudentTChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendMvNormalChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)

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
