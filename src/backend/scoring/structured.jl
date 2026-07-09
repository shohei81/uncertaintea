# Backend-native scalar and batched scoring: structured families (mvnormal, mvnormaldense, dirichlet, mixture, broadcast/iid vector machinery).

function _backend_mvnormal_observed_value(value, expected_length::Int, ::Type{T}=Float64) where {T<:Real}
    values = value isa Tuple ? collect(value) : value
    values isa AbstractVector || throw(ArgumentError("mvnormal expects a vector or tuple value"))
    length(values) == expected_length ||
        throw(ArgumentError("mvnormal expects a value of length $expected_length, got $(length(values))"))
    return T[convert(T, float(item)) for item in values]
end

function _assign_backend_choice_vector_value!(
    env::BatchedPlanEnvironment,
    slot::Int,
    values::AbstractMatrix,
)
    env.generic_slots[slot] || throw(BatchedBackendFallback("mvnormal backend binding slot $slot must be generic"))
    storage = env.generic_values[slot]
    for batch_index in 1:env.batch_size
        storage[batch_index] = collect(view(values, :, batch_index))
    end
    env.assigned[slot] = true
    return env
end

function _assign_backend_choice_vector_value!(
    env::BatchedPlanEnvironment,
    slot::Int,
    values::AbstractVector{<:AbstractVector},
)
    env.generic_slots[slot] || throw(BatchedBackendFallback("mvnormal backend binding slot $slot must be generic"))
    storage = env.generic_values[slot]
    for batch_index in 1:env.batch_size
        storage[batch_index] = [component_values[batch_index] for component_values in values]
    end
    env.assigned[slot] = true
    return env
end

function _backend_mvnormal_logpdf(mu::Tuple, sigma::Tuple, x)
    (length(mu) == length(sigma) && length(mu) == length(x)) ||
        throw(ArgumentError("mvnormal requires matching mean, scale, and value lengths"))
    total = _backend_normal_logpdf(mu[1], sigma[1], x[1])
    for index in 2:length(mu)
        total += _backend_normal_logpdf(mu[index], sigma[index], x[index])
    end
    return total
end

function _score_backend_step!(
    step::BackendMvNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    mu = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.mu)...)
    sigma = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.sigma)...)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_mvnormal_logpdf(mu, sigma, value)
end

function _score_backend_step!(
    step::BackendMvNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = [_batched_numeric_scratch!(env, index) for index in 1:step.value_length]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)

    mu_values = [_batched_numeric_scratch!(env, step.value_length + index) for index in 1:step.value_length]
    sigma_values = [_batched_numeric_scratch!(env, 2 * step.value_length + index) for index in 1:step.value_length]
    for component_index in 1:step.value_length
        _eval_backend_numeric_expr!(mu_values[component_index], env, step.mu[component_index], 3 * step.value_length + 1)
        _eval_backend_numeric_expr!(sigma_values[component_index], env, step.sigma[component_index], 3 * step.value_length + 2)
        for batch_index in 1:env.batch_size
            totals[batch_index] += _backend_normal_logpdf(
                mu_values[component_index][batch_index],
                sigma_values[component_index][batch_index],
                choice_values[component_index][batch_index],
            )
        end
    end

    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end

# --- mixture of normal components ---------------------------------------------

function _backend_mixture_normal_logpdf(weights, mus, sigmas, x)
    k = length(weights)
    (length(mus) == k && length(sigmas) == k) ||
        throw(ArgumentError("mixture requires one weight per normal component"))
    xf = float(x)
    T = typeof(xf + float(weights[1]) + float(mus[1]) + float(sigmas[1]))
    m = T(-Inf)
    for index in 1:k
        term = log(weights[index]) + _backend_normal_logpdf(mus[index], sigmas[index], xf)
        term > m && (m = term)
    end
    isfinite(m) || return T(-Inf)
    total = zero(T)
    for index in 1:k
        term = log(weights[index]) + _backend_normal_logpdf(mus[index], sigmas[index], xf)
        total += exp(term - m)
    end
    return m + log(total)
end

function _score_backend_step!(
    step::BackendMixtureNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    weights = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.weights)...)
    mus = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.mus)...)
    sigmas = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.sigmas)...)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_mixture_normal_logpdf(weights, mus, sigmas, value)
end

function _score_backend_step!(
    step::BackendMixtureNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    k = length(step.weights)
    choice_values = env.observed_values
    weight_values = [_batched_numeric_scratch!(env, index) for index in 1:k]
    mu_values = [_batched_numeric_scratch!(env, k + index) for index in 1:k]
    sigma_values = [_batched_numeric_scratch!(env, 2 * k + index) for index in 1:k]
    for index in 1:k
        _eval_backend_numeric_expr!(weight_values[index], env, step.weights[index], 3 * k + 1)
        _eval_backend_numeric_expr!(mu_values[index], env, step.mus[index], 3 * k + 2)
        _eval_backend_numeric_expr!(sigma_values[index], env, step.sigmas[index], 3 * k + 3)
    end
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in 1:env.batch_size
        value = choice_values[batch_index]
        weights = ntuple(index -> weight_values[index][batch_index], k)
        mus = ntuple(index -> mu_values[index][batch_index], k)
        sigmas = ntuple(index -> sigma_values[index][batch_index], k)
        totals[batch_index] += _backend_mixture_normal_logpdf(weights, mus, sigmas, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = Int(value)
            else
                env.generic_values[step.binding_slot][batch_index] = value
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

# --- dense-covariance multivariate normal -------------------------------------

# Log density of mvnormaldense via forward substitution solving L z = x - mu,
# reading only the lower triangle of the d x d factor `Lmat`. Matches the CPU
# `MvNormalDenseDist` reference (returns -Inf on a length mismatch).
function _backend_mvnormaldense_logpdf(mu, Lmat, x)
    d = length(mu)
    (length(x) == d && size(Lmat, 1) == d) || return oftype(float(x[1]), -Inf)
    z1 = (x[1] - mu[1]) / Lmat[1, 1]
    solved = Vector{typeof(z1)}(undef, d)
    solved[1] = z1
    log_det = log(Lmat[1, 1])
    quadratic = z1 * z1
    for row in 2:d
        residual = x[row] - mu[row]
        for col in 1:(row - 1)
            residual -= Lmat[row, col] * solved[col]
        end
        z = residual / Lmat[row, row]
        solved[row] = z
        log_det += log(Lmat[row, row])
        quadratic += z * z
    end
    return -log_det - quadratic / 2 - d * log(2 * pi) / 2
end

# Read the whole scale_tril matrix from the generic slot the argument expression
# resolves to, validating it is a d x d matrix.
function _backend_mvnormaldense_scale_matrix(matrix, d::Int)
    matrix isa AbstractMatrix ||
        throw(ArgumentError("mvnormaldense scale_tril must evaluate to a matrix, got $(typeof(matrix))"))
    (size(matrix, 1) == d && size(matrix, 2) == d) ||
        throw(ArgumentError("mvnormaldense scale_tril must be $(d)x$(d), got $(size(matrix))"))
    return matrix
end

function _score_backend_step!(
    step::BackendMvNormalDenseChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    d = step.value_length
    mu = [_eval_backend_numeric_expr(env, expr) for expr in step.mu]
    Lmat = _backend_mvnormaldense_scale_matrix(_environment_value(env, step.scale_tril.slot), d)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_mvnormaldense_logpdf(mu, Lmat, value)
end

function _score_backend_step!(
    step::BackendMvNormalDenseChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    d = step.value_length
    choice_values = [_batched_numeric_scratch!(env, index) for index in 1:d]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_vector_values!(choice_values, step.value_index, d, params, constraints, address_parts)

    mu_values = [_batched_numeric_scratch!(env, d + index) for index in 1:d]
    for index in 1:d
        _eval_backend_numeric_expr!(mu_values[index], env, step.mu[index], 2 * d + 1)
    end

    env.assigned[step.scale_tril.slot] ||
        throw(BatchedBackendFallback("mvnormaldense scale_tril slot $(step.scale_tril.slot) is not assigned"))
    scale_storage = env.generic_values[step.scale_tril.slot]
    for batch_index in 1:env.batch_size
        mu = ntuple(index -> mu_values[index][batch_index], d)
        Lmat = _backend_mvnormaldense_scale_matrix(scale_storage[batch_index], d)
        x = ntuple(index -> choice_values[index][batch_index], d)
        totals[batch_index] += _backend_mvnormaldense_logpdf(mu, Lmat, x)
    end

    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end
