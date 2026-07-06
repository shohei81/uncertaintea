function _accumulate_laplace_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_laplace_logpdf(mu, scale, value)
        delta = value - mu
        sign_delta = delta > 0 ? one(T) : (delta < 0 ? -one(T) : zero(T))
        dvalue = -sign_delta / scale
        dmu = sign_delta / scale
        dscale = -1 / scale + abs(delta) / (scale * scale)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_geometric_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_geometric_logpdf(probability, value)
        count = _poisson_count(value)
        isnothing(count) && continue
        derivative = 1 / probability - count / (1 - probability)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_negativebinomial_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    successes_values::AbstractVector{T},
    successes_gradients::AbstractMatrix{T},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        successes = successes_values[batch_index]
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_negativebinomial_logpdf(successes, probability, value)
        count = _poisson_count(value)
        isnothing(count) && continue
        dsuccesses = digamma(count + successes) - digamma(successes) + log(probability)
        dprobability = successes / probability - count / (1 - probability)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dsuccesses * successes_gradients[parameter_index, batch_index] +
                dprobability * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_mvnormal_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{<:AbstractVector},
    value_gradients::AbstractVector{<:AbstractMatrix},
    mu_values::AbstractVector{<:AbstractVector},
    mu_gradients::AbstractVector{<:AbstractMatrix},
    sigma_values::AbstractVector{<:AbstractVector},
    sigma_gradients::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        for component_index in eachindex(value_values)
            value = value_values[component_index][batch_index]
            mu = mu_values[component_index][batch_index]
            sigma = sigma_values[component_index][batch_index]
            totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
            z = (value - mu) / sigma
            inv_sigma = 1 / sigma
            dvalue = -z * inv_sigma
            dmu = z * inv_sigma
            dsigma = (z * z - 1) * inv_sigma
            component_value_gradients = value_gradients[component_index]
            component_mu_gradients = mu_gradients[component_index]
            component_sigma_gradients = sigma_gradients[component_index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] +=
                    dvalue * component_value_gradients[parameter_index, batch_index] +
                    dmu * component_mu_gradients[parameter_index, batch_index] +
                    dsigma * component_sigma_gradients[parameter_index, batch_index]
            end
        end
    end
    return totals, gradients
end

function _accumulate_dirichlet_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    parameter_index::Union{Nothing,Int},
    value_values::AbstractVector{<:AbstractVector},
    alpha_values::AbstractVector{<:AbstractVector},
    alpha_gradients::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    value_length = length(value_values)
    choice_gradients = Vector{T}(undef, value_length)
    alpha_derivatives = Vector{T}(undef, value_length)
    for batch_index in eachindex(totals)
        total_alpha = zero(T)
        accumulator = zero(T)
        for component_index in 1:value_length
            alpha = alpha_values[component_index][batch_index]
            alpha > 0 || throw(ArgumentError("dirichlet requires alpha > 0 in every dimension"))
            total_alpha += alpha
            accumulator -= loggamma(alpha)
        end
        accumulator += loggamma(total_alpha)

        total = zero(T)
        valid = true
        weighted_choice_gradient = zero(T)
        for component_index in 1:value_length
            value = value_values[component_index][batch_index]
            value > 0 || begin
                valid = false
                break
            end
            total += value
            alpha = alpha_values[component_index][batch_index]
            accumulator += (alpha - 1) * log(value)
            choice_gradient = (alpha - 1) / value
            choice_gradients[component_index] = choice_gradient
            weighted_choice_gradient += choice_gradient * value
            alpha_derivatives[component_index] = digamma(total_alpha) - digamma(alpha) + log(value)
        end

        if !valid || abs(total - 1) > sqrt(eps(total)) * value_length * 16
            totals[batch_index] += -T(Inf)
            continue
        end

        totals[batch_index] += accumulator
        for component_index in 1:value_length
            alpha_derivative = alpha_derivatives[component_index]
            component_alpha_gradients = alpha_gradients[component_index]
            for parameter_row in axes(gradients, 1)
                gradients[parameter_row, batch_index] +=
                    alpha_derivative * component_alpha_gradients[parameter_row, batch_index]
            end
        end
        isnothing(parameter_index) && continue
        for component_index in 1:(value_length - 1)
            unconstrained_row = parameter_index + component_index - 1
            constrained_value = value_values[component_index][batch_index]
            gradients[unconstrained_row, batch_index] +=
                constrained_value * (choice_gradients[component_index] - weighted_choice_gradient)
        end
    end
    return totals, gradients
end

function _assign_backend_choice_vector_value!(
    env::BatchedPlanEnvironment{T},
    slot_gradients::Array{T,3},
    slot::Int,
    values::AbstractVector{<:AbstractVector},
    gradients::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    env.generic_slots[slot] || throw(BatchedBackendFallback("mvnormal backend binding slot $slot must be generic"))
    storage = env.generic_values[slot]
    for batch_index in 1:env.batch_size
        storage[batch_index] = [component_values[batch_index] for component_values in values]
    end
    env.assigned[slot] = true
    return env
end

function _score_backend_step_and_gradient!(
    step::BackendLaplaceChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 5)
    _accumulate_laplace_gradient!(totals, gradients, value_values, value_gradients, mu_values, mu_gradients, scale_values, scale_gradients)
    isnothing(step.binding_slot) || _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendMvNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    choice_values = [_batched_numeric_scratch!(env, index) for index in 1:step.value_length]
    choice_gradients = [_batched_backend_gradient_scratch!(cache, index) for index in 1:step.value_length]
    mu_values = [_batched_numeric_scratch!(env, step.value_length + index) for index in 1:step.value_length]
    mu_gradients = [_batched_backend_gradient_scratch!(cache, step.value_length + index) for index in 1:step.value_length]
    sigma_values = [_batched_numeric_scratch!(env, 2 * step.value_length + index) for index in 1:step.value_length]
    sigma_gradients = [_batched_backend_gradient_scratch!(cache, 2 * step.value_length + index) for index in 1:step.value_length]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    for component_index in 1:step.value_length
        _fill_choice_vector_gradient!(choice_gradients[component_index], step.value_index, component_index)
        _eval_backend_numeric_expr_and_gradient!(
            mu_values[component_index],
            mu_gradients[component_index],
            cache,
            env,
            step.mu[component_index],
            3 * step.value_length + 1,
        )
        _eval_backend_numeric_expr_and_gradient!(
            sigma_values[component_index],
            sigma_gradients[component_index],
            cache,
            env,
            step.sigma[component_index],
            3 * step.value_length + 2,
        )
    end

    _accumulate_mvnormal_gradient!(
        totals,
        gradients,
        choice_values,
        choice_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )

    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, choice_gradients)
    return totals, gradients
end

# Element-aware value+gradient evaluation of a broadcast argument for a fixed data
# element, reusing the scalar numeric-gradient primitive kernels. Vector generic slots
# contribute their element and carry zero gradient (data); numeric latents carry their
# accumulated slot gradient.
function _eval_backend_broadcast_numeric_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendLiteralExpr,
    element_index::Int,
    depth::Int,
) where {T<:AbstractFloat}
    fill!(values, T(_require_numeric_value(env, expr.value, "batched broadcast argument")))
    fill!(gradients, zero(T))
    return values, gradients
end

function _eval_backend_broadcast_numeric_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendSlotExpr,
    element_index::Int,
    depth::Int,
) where {T<:AbstractFloat}
    env.assigned[expr.slot] || throw(BatchedBackendFallback("environment slot $(expr.slot) is not assigned"))
    if env.numeric_slots[expr.slot]
        copyto!(values, view(env.numeric_values, expr.slot, :))
        _copy_slot_gradient!(gradients, cache.slot_gradients, expr.slot)
        return values, gradients
    elseif env.index_slots[expr.slot]
        for batch_index in eachindex(values)
            values[batch_index] = T(env.index_values[expr.slot, batch_index])
        end
        fill!(gradients, zero(T))
        return values, gradients
    end
    storage = env.generic_values[expr.slot]
    for batch_index in eachindex(values)
        value = storage[batch_index]
        scalar = value isa AbstractVector ? value[element_index] : value
        values[batch_index] = T(_require_numeric_value(env, scalar, "batched broadcast slot"))
    end
    fill!(gradients, zero(T))
    return values, gradients
end

function _eval_backend_broadcast_numeric_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendPrimitiveExpr,
    element_index::Int,
    depth::Int,
) where {T<:AbstractFloat}
    expr.op in BACKEND_GRADIENT_SUPPORTED_PRIMITIVES ||
        _backend_numeric_error(env, "batched broadcast gradient does not support primitive `$(expr.op)`")
    isempty(expr.arguments) && _backend_numeric_error(env, "batched broadcast gradient primitive requires arguments")

    _eval_backend_broadcast_numeric_and_gradient!(values, gradients, cache, env, first(expr.arguments), element_index, depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_numeric_gradient_unary!(values, gradients, env, expr.op)
    elseif expr.op === :clamp
        length(expr.arguments) == 3 ||
            _backend_numeric_error(env, "batched broadcast gradient `clamp` expects exactly 3 arguments")
        middle_values = _batched_numeric_scratch!(env, depth)
        middle_gradients = _batched_backend_gradient_scratch!(cache, depth)
        rhs_values = _batched_numeric_scratch!(env, depth + 1)
        rhs_gradients = _batched_backend_gradient_scratch!(cache, depth + 1)
        _eval_backend_broadcast_numeric_and_gradient!(middle_values, middle_gradients, cache, env, expr.arguments[2], element_index, depth + 2)
        _eval_backend_broadcast_numeric_and_gradient!(rhs_values, rhs_gradients, cache, env, expr.arguments[3], element_index, depth + 2)
        return _apply_backend_numeric_gradient_ternary!(
            values, gradients, env, expr.op, middle_values, middle_gradients, rhs_values, rhs_gradients,
        )
    end

    temp_values = _batched_numeric_scratch!(env, depth)
    temp_gradients = _batched_backend_gradient_scratch!(cache, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_broadcast_numeric_and_gradient!(temp_values, temp_gradients, cache, env, argument, element_index, depth + 1)
        _apply_backend_numeric_gradient_binary!(values, gradients, env, expr.op, temp_values, temp_gradients)
    end
    return values, gradients
end

function _eval_backend_broadcast_numeric_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendBlockExpr,
    element_index::Int,
    depth::Int,
) where {T<:AbstractFloat}
    for argument in expr.arguments
        _eval_backend_broadcast_numeric_and_gradient!(values, gradients, cache, env, argument, element_index, depth)
    end
    return values, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBroadcastNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.binding_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support broadcast normal bindings"))
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    observed = _batched_broadcast_observed_values(env, address_parts, constraints)
    n = _broadcast_uniform_length(observed)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 2)
    for element_index in 1:n
        _eval_backend_broadcast_numeric_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, element_index, 3)
        _eval_backend_broadcast_numeric_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, element_index, 5)
        for batch_index in eachindex(totals)
            value = observed[batch_index][element_index]
            mu = mu_values[batch_index]
            sigma = sigma_values[batch_index]
            totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
            z = (value - mu) / sigma
            inv_sigma = 1 / sigma
            dmu = z * inv_sigma
            dsigma = (z * z - 1) * inv_sigma
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] +=
                    dmu * mu_gradients[parameter_index, batch_index] +
                    dsigma * sigma_gradients[parameter_index, batch_index]
            end
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendDirichletChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    choice_values = [_batched_numeric_scratch!(env, index) for index in 1:step.value_length]
    alpha_values = [_batched_numeric_scratch!(env, step.value_length + index) for index in 1:step.value_length]
    alpha_gradients = [
        _batched_backend_gradient_scratch!(cache, step.value_length + index) for index in 1:step.value_length
    ]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    for component_index in 1:step.value_length
        _eval_backend_numeric_expr_and_gradient!(
            alpha_values[component_index],
            alpha_gradients[component_index],
            cache,
            env,
            step.alpha[component_index],
            2 * step.value_length + 1,
        )
    end

    _accumulate_dirichlet_gradient!(
        totals,
        gradients,
        step.parameter_index,
        choice_values,
        alpha_values,
        alpha_gradients,
    )

    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, alpha_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendGeometricChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) || throw(BatchedBackendFallback("batched backend gradient does not support Geometric latent parameters"))
    value_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 2)
    _accumulate_geometric_gradient!(totals, gradients, probability_values, probability_gradients, value_values)

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 3)),
        )
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendNegativeBinomialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support NegativeBinomial latent parameters"))
    value_values = env.observed_values
    successes_values = _batched_numeric_scratch!(env, 1)
    successes_gradients = _batched_backend_gradient_scratch!(cache, 1)
    probability_values = _batched_numeric_scratch!(env, 2)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(successes_values, successes_gradients, cache, env, step.successes, 3)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 4)
    _accumulate_negativebinomial_gradient!(
        totals,
        gradients,
        successes_values,
        successes_gradients,
        probability_values,
        probability_gradients,
        value_values,
    )

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 5)),
        )
    end
    return totals, gradients
end

# --- truncated normal gradient ------------------------------------------------

function _accumulate_truncatednormal_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    sigma_values::AbstractVector{T},
    sigma_gradients::AbstractMatrix{T},
    lower_values::AbstractVector{T},
    lower_gradients::AbstractMatrix{T},
    upper_values::AbstractVector{T},
    upper_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        lower = lower_values[batch_index]
        upper = upper_values[batch_index]
        totals[batch_index] += _backend_truncatednormal_logpdf(mu, sigma, lower, upper, value)
        (value < lower || value > upper) && continue
        inv_sigma = 1 / sigma
        zx = (value - mu) * inv_sigma
        za = (lower - mu) * inv_sigma
        zb = (upper - mu) * inv_sigma
        normalizer = exp(_log_normal_cdf_diff(za, zb))
        phi_a = _std_normal_pdf(za)
        phi_b = _std_normal_pdf(zb)
        za_phi_a = isinf(za) ? zero(T) : za * phi_a
        zb_phi_b = isinf(zb) ? zero(T) : zb * phi_b
        inv_sz = inv_sigma / normalizer
        dvalue = -zx * inv_sigma
        dmu = zx * inv_sigma - (phi_a - phi_b) * inv_sz
        dsigma = (zx * zx - one(T)) * inv_sigma - (za_phi_a - zb_phi_b) * inv_sz
        dlower = phi_a * inv_sz
        dupper = -phi_b * inv_sz
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index] +
                dlower * lower_gradients[parameter_index, batch_index] +
                dupper * upper_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendTruncatedNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    lower_values = _batched_numeric_scratch!(env, 3)
    lower_gradients = _batched_backend_gradient_scratch!(cache, 4)
    upper_values = _batched_numeric_scratch!(env, 4)
    upper_gradients = _batched_backend_gradient_scratch!(cache, 5)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 6)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 7)
    _eval_backend_numeric_expr_and_gradient!(lower_values, lower_gradients, cache, env, step.lower, 8)
    _eval_backend_numeric_expr_and_gradient!(upper_values, upper_gradients, cache, env, step.upper, 9)
    _accumulate_truncatednormal_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
        lower_values,
        lower_gradients,
        upper_values,
        upper_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

# --- mixture-of-normals gradient ----------------------------------------------

function _accumulate_mixture_normal_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    weight_values::AbstractVector{<:AbstractVector},
    weight_gradients::AbstractVector{<:AbstractMatrix},
    mu_values::AbstractVector{<:AbstractVector},
    mu_gradients::AbstractVector{<:AbstractMatrix},
    sigma_values::AbstractVector{<:AbstractVector},
    sigma_gradients::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    k = length(weight_values)
    terms = Vector{T}(undef, k)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        weights = ntuple(index -> weight_values[index][batch_index], k)
        mus = ntuple(index -> mu_values[index][batch_index], k)
        sigmas = ntuple(index -> sigma_values[index][batch_index], k)
        logpdf = _backend_mixture_normal_logpdf(weights, mus, sigmas, value)
        totals[batch_index] += logpdf
        isfinite(logpdf) || continue
        dvalue = zero(T)
        for index in 1:k
            lp = _backend_normal_logpdf(mus[index], sigmas[index], value)
            terms[index] = lp
        end
        for index in 1:k
            mu = mus[index]
            sigma = sigmas[index]
            lp = terms[index]
            responsibility = exp(log(weights[index]) + lp - logpdf)
            weight_derivative = exp(lp - logpdf)
            z = (value - mu) / sigma
            inv_sigma = 1 / sigma
            dvalue += responsibility * (-z * inv_sigma)
            dmu = responsibility * (z * inv_sigma)
            dsigma = responsibility * ((z * z - one(T)) * inv_sigma)
            component_weight_gradients = weight_gradients[index]
            component_mu_gradients = mu_gradients[index]
            component_sigma_gradients = sigma_gradients[index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] +=
                    weight_derivative * component_weight_gradients[parameter_index, batch_index] +
                    dmu * component_mu_gradients[parameter_index, batch_index] +
                    dsigma * component_sigma_gradients[parameter_index, batch_index]
            end
        end
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += dvalue * value_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendMixtureNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    k = length(step.weights)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    weight_values = [_batched_numeric_scratch!(env, index) for index in 1:k]
    weight_gradients = [_batched_backend_gradient_scratch!(cache, 1 + index) for index in 1:k]
    mu_values = [_batched_numeric_scratch!(env, k + index) for index in 1:k]
    mu_gradients = [_batched_backend_gradient_scratch!(cache, k + 1 + index) for index in 1:k]
    sigma_values = [_batched_numeric_scratch!(env, 2 * k + index) for index in 1:k]
    sigma_gradients = [_batched_backend_gradient_scratch!(cache, 2 * k + 1 + index) for index in 1:k]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot)
    for index in 1:k
        _eval_backend_numeric_expr_and_gradient!(weight_values[index], weight_gradients[index], cache, env, step.weights[index], 3 * k + 2)
        _eval_backend_numeric_expr_and_gradient!(mu_values[index], mu_gradients[index], cache, env, step.mus[index], 3 * k + 2)
        _eval_backend_numeric_expr_and_gradient!(sigma_values[index], sigma_gradients[index], cache, env, step.sigmas[index], 3 * k + 2)
    end
    _accumulate_mixture_normal_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        weight_values,
        weight_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

# --- dense multivariate normal gradient ---------------------------------------

# Solve L z = residual (forward) then Lᵀ w = z (back) so that w = L⁻ᵀ z, giving
# d logpdf / d mu_i = w_i and d logpdf / d x_i = -w_i (the scale factor is a
# constant matrix, so it contributes no gradient).
function _accumulate_mvnormaldense_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    d::Int,
    choice_values::AbstractVector{<:AbstractVector},
    choice_gradients::AbstractVector{<:AbstractMatrix},
    mu_values::AbstractVector{<:AbstractVector},
    mu_gradients::AbstractVector{<:AbstractMatrix},
    scale_storage::AbstractVector,
) where {T<:AbstractFloat}
    z = Vector{T}(undef, d)
    w = Vector{T}(undef, d)
    for batch_index in eachindex(totals)
        Lmat = _backend_mvnormaldense_scale_matrix(scale_storage[batch_index], d)
        mu = ntuple(index -> mu_values[index][batch_index], d)
        x = ntuple(index -> choice_values[index][batch_index], d)
        totals[batch_index] += _backend_mvnormaldense_logpdf(mu, Lmat, x)
        # forward substitution: L z = x - mu
        for row in 1:d
            residual = x[row] - mu[row]
            for col in 1:(row - 1)
                residual -= Lmat[row, col] * z[col]
            end
            z[row] = residual / Lmat[row, row]
        end
        # back substitution: Lᵀ w = z
        for row in d:-1:1
            accumulator = z[row]
            for col in (row + 1):d
                accumulator -= Lmat[col, row] * w[col]
            end
            w[row] = accumulator / Lmat[row, row]
        end
        for component_index in 1:d
            dmu = w[component_index]
            dvalue = -w[component_index]
            component_mu_gradients = mu_gradients[component_index]
            component_value_gradients = choice_gradients[component_index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] +=
                    dmu * component_mu_gradients[parameter_index, batch_index] +
                    dvalue * component_value_gradients[parameter_index, batch_index]
            end
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendMvNormalDenseChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    d = step.value_length
    choice_values = [_batched_numeric_scratch!(env, index) for index in 1:d]
    choice_gradients = [_batched_backend_gradient_scratch!(cache, index) for index in 1:d]
    mu_values = [_batched_numeric_scratch!(env, d + index) for index in 1:d]
    mu_gradients = [_batched_backend_gradient_scratch!(cache, d + index) for index in 1:d]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, d, params, constraints, address_parts)
    for component_index in 1:d
        _fill_choice_vector_gradient!(choice_gradients[component_index], step.value_index, component_index)
        _eval_backend_numeric_expr_and_gradient!(
            mu_values[component_index],
            mu_gradients[component_index],
            cache,
            env,
            step.mu[component_index],
            2 * d + 1,
        )
    end

    env.assigned[step.scale_tril.slot] ||
        throw(BatchedBackendFallback("mvnormaldense scale_tril slot $(step.scale_tril.slot) is not assigned"))
    _accumulate_mvnormaldense_gradient!(
        totals,
        gradients,
        d,
        choice_values,
        choice_gradients,
        mu_values,
        mu_gradients,
        env.generic_values[step.scale_tril.slot],
    )

    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, choice_gradients)
    return totals, gradients
end
