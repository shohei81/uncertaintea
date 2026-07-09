# Backend-native scalar and batched scoring: truncated families (truncatednormal, truncatedstudentt).

# --- truncated normal ---------------------------------------------------------

# Standard normal pdf with an explicit guard so an infinite standardized bound
# (from an unbounded truncation side) contributes a zero density, keeping the
# normalizer-gradient terms finite.
function _std_normal_pdf(z)
    zz = float(z)
    isinf(zz) && return zero(zz)
    return exp(-zz * zz / 2) / sqrt(oftype(zz, 2) * pi)
end

function _backend_truncatednormal_logpdf(mu, sigma, lower, upper, x)
    xx, mu_, sigma_, lower_, upper_ = promote(x, mu, sigma, lower, upper)
    sigma_ > zero(sigma_) || throw(ArgumentError("truncatednormal requires sigma > 0"))
    (xx < lower_ || xx > upper_) && return oftype(xx, -Inf)
    base = _backend_normal_logpdf(mu_, sigma_, xx)
    za = (lower_ - mu_) / sigma_
    zb = (upper_ - mu_) / sigma_
    return base - _log_normal_cdf_diff(za, zb)
end

function _score_backend_step!(
    step::BackendTruncatedNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    lower = _eval_backend_numeric_expr(env, step.lower)
    upper = _eval_backend_numeric_expr(env, step.upper)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_truncatednormal_logpdf(mu, sigma, lower, upper, value)
end

function _score_backend_step!(
    step::BackendTruncatedNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    lower_values = _batched_numeric_scratch!(env, 3)
    upper_values = _batched_numeric_scratch!(env, 4)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 5)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 6)
    _eval_backend_numeric_expr!(lower_values, env, step.lower, 7)
    _eval_backend_numeric_expr!(upper_values, env, step.upper, 8)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        totals[batch_index] += _backend_truncatednormal_logpdf(
            mu_values[batch_index],
            sigma_values[batch_index],
            lower_values[batch_index],
            upper_values[batch_index],
            value,
        )
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

# --- truncated Student-t ------------------------------------------------------
# `_std_t_pdf` (the standard Student-t density, with the infinite-bound guard used
# below and in the analytic gradient) lives alongside `_std_t_cdf` in
# distributions.jl.

function _backend_truncatedstudentt_logpdf(nu, mu, sigma, lower, upper, x)
    xx, nu_, mu_, sigma_, lower_, upper_ = promote(x, nu, mu, sigma, lower, upper)
    # Mirror the CPU `TruncatedStudentTDist` constructor contract: the
    # backend-native path bypasses construction, so re-validate the parameters
    # here to keep behavior (a clear ArgumentError) identical to the reference.
    nu_ > zero(nu_) || throw(ArgumentError("truncatedstudentt requires nu > 0"))
    sigma_ > zero(sigma_) || throw(ArgumentError("truncatedstudentt requires sigma > 0"))
    lower_ < upper_ || throw(ArgumentError("truncatedstudentt requires lower < upper"))
    (xx < lower_ || xx > upper_) && return oftype(xx, -Inf)
    base = _backend_studentt_logpdf(nu_, mu_, sigma_, xx)
    za = (lower_ - mu_) / sigma_
    zb = (upper_ - mu_) / sigma_
    return base - log(_std_t_cdf(zb, nu_) - _std_t_cdf(za, nu_))
end

function _score_backend_step!(
    step::BackendTruncatedStudentTChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    nu = _eval_backend_numeric_expr(env, step.nu)
    mu = _eval_backend_numeric_expr(env, step.mu)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    lower = _eval_backend_numeric_expr(env, step.lower)
    upper = _eval_backend_numeric_expr(env, step.upper)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_truncatedstudentt_logpdf(nu, mu, sigma, lower, upper, value)
end

function _score_backend_step!(
    step::BackendTruncatedStudentTChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    nu_values = _batched_numeric_scratch!(env, 1)
    mu_values = _batched_numeric_scratch!(env, 2)
    sigma_values = _batched_numeric_scratch!(env, 3)
    lower_values = _batched_numeric_scratch!(env, 4)
    upper_values = _batched_numeric_scratch!(env, 5)
    _eval_backend_numeric_expr!(nu_values, env, step.nu, 6)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 7)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 8)
    _eval_backend_numeric_expr!(lower_values, env, step.lower, 9)
    _eval_backend_numeric_expr!(upper_values, env, step.upper, 10)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        totals[batch_index] += _backend_truncatedstudentt_logpdf(
            nu_values[batch_index],
            mu_values[batch_index],
            sigma_values[batch_index],
            lower_values[batch_index],
            upper_values[batch_index],
            value,
        )
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
