# The one fused KernelAbstractions logjoint kernel plus the device-side, fully
# inlined interpreter of the device execution plan. One thread == one batch column.
#
# The plan is passed to the kernel by value (it is `isbits`); its step tuple type
# specializes the kernel, so the recursive walk unrolls at compile time. Per-column
# scratch lives in a `slots` matrix (device buffer). Observed values are read from
# `observed[cursor, col]`, with the cursor threaded through the recursion exactly as
# staging assigned rows.

# ---- expression evaluation -----------------------------------------------------

@inline _device_eval(e::DeviceLiteralExpr, slots, col) = e.value
@inline _device_eval(e::DeviceSlotExpr, slots, col) = @inbounds slots[e.slot, col]
@inline _device_eval(e::DevicePrimitiveExpr{Op}, slots, col) where {Op} =
    _device_apply(Val(Op), _device_eval_args(e.args, slots, col)...)

@inline _device_eval_args(::Tuple{}, slots, col) = ()
@inline _device_eval_args(t::Tuple, slots, col) =
    (_device_eval(first(t), slots, col), _device_eval_args(Base.tail(t), slots, col)...)

@inline _device_apply(::Val{:+}, args...) = +(args...)
@inline _device_apply(::Val{:-}, args...) = -(args...)
@inline _device_apply(::Val{:*}, args...) = *(args...)
@inline _device_apply(::Val{:/}, a, b) = a / b
@inline _device_apply(::Val{:^}, a, b) = a^b
@inline _device_apply(::Val{:exp}, a) = exp(a)
@inline _device_apply(::Val{:log}, a) = log(a)
@inline _device_apply(::Val{:log1p}, a) = log1p(a)
@inline _device_apply(::Val{:sqrt}, a) = sqrt(a)
@inline _device_apply(::Val{:abs}, a) = abs(a)
@inline _device_apply(::Val{:min}, args...) = min(args...)
@inline _device_apply(::Val{:max}, args...) = max(args...)
@inline _device_apply(::Val{:clamp}, a, b, c) = clamp(a, b, c)

# ---- choice value resolution ---------------------------------------------------

@inline function _device_store_binding!(slots, binding_slot::Int32, value, col)
    if binding_slot > Int32(0)
        @inbounds slots[binding_slot, col] = value
    end
    return nothing
end

# Returns (value, logabsdet, new_cursor).
@inline function _device_choice_value(step, params, observed, col, cursor::Int32)
    if step.value_source > Int32(0)
        u = @inbounds params[step.value_source, col]
        c, lad = _device_transform(step.transform, u)
        return (c, lad, cursor)
    else
        v = @inbounds observed[cursor, col]
        return (v, zero(v), cursor + Int32(1))
    end
end

# ---- per-step scoring ----------------------------------------------------------

@inline function _device_score_step(step::DeviceNormalChoiceStep, slots, params, observed, tc, ls, col, cursor)
    mu = _device_eval(step.mu, slots, col)
    sigma = _device_eval(step.sigma, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_normal_logpdf(mu, sigma, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceNoncenteredNormalChoiceStep, slots, params, observed, tc, ls, col, cursor)
    mu = _device_eval(step.mu, slots, col)
    sigma = _device_eval(step.sigma, slots, col)
    z = @inbounds params[step.value_source, col]
    _device_store_binding!(slots, step.binding_slot, mu + sigma * z, col)
    return (_device_normal_logpdf(zero(z), one(z), z), cursor)
end

@inline function _device_score_step(step::DeviceLognormalChoiceStep, slots, params, observed, tc, ls, col, cursor)
    mu = _device_eval(step.mu, slots, col)
    sigma = _device_eval(step.sigma, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_lognormal_logpdf(mu, sigma, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceExponentialChoiceStep, slots, params, observed, tc, ls, col, cursor)
    rate = _device_eval(step.rate, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_exponential_logpdf(rate, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceGammaChoiceStep, slots, params, observed, tc, ls, col, cursor)
    shape = _device_eval(step.shape, slots, col)
    rate = _device_eval(step.rate, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_gamma_logpdf(shape, rate, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceLaplaceChoiceStep, slots, params, observed, tc, ls, col, cursor)
    loc = _device_eval(step.loc, slots, col)
    scale = _device_eval(step.scale, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_laplace_logpdf(loc, scale, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceBetaChoiceStep, slots, params, observed, tc, ls, col, cursor)
    alpha = _device_eval(step.alpha, slots, col)
    beta = _device_eval(step.beta, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_beta_logpdf(alpha, beta, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceBernoulliChoiceStep, slots, params, observed, tc, ls, col, cursor)
    p = _device_eval(step.probability, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_bernoulli_logpdf(p, value) + lad, cur)
end

@inline function _device_score_step(step::DevicePoissonChoiceStep, slots, params, observed, tc, ls, col, cursor)
    lambda = _device_eval(step.lambda, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_poisson_logpdf(lambda, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceDeterministicStep, slots, params, observed, tc, ls, col, cursor)
    @inbounds slots[step.binding_slot, col] = _device_eval(step.expr, slots, col)
    return (zero(eltype(slots)), cursor)
end

@inline function _device_score_step(step::DeviceLoopStep, slots, params, observed, tc, ls, col, cursor)
    Tt = eltype(slots)
    count = @inbounds tc[step.loop_id]
    start = @inbounds ls[step.loop_id]
    total = zero(Tt)
    cur = cursor
    for t = Int32(0):(count-Int32(1))
        if step.iterator_slot > Int32(0)
            @inbounds slots[step.iterator_slot, col] = Tt(start + t)
        end
        contribution, cur = _device_score_steps(step.body, slots, params, observed, tc, ls, col, cur)
        total += contribution
    end
    return (total, cur)
end

# ---- recursive step-tuple walk -------------------------------------------------

@inline _device_score_steps(::Tuple{}, slots, params, observed, tc, ls, col, cursor) =
    (zero(eltype(slots)), cursor)

@inline function _device_score_steps(steps::Tuple, slots, params, observed, tc, ls, col, cursor)
    contribution, cur = _device_score_step(first(steps), slots, params, observed, tc, ls, col, cursor)
    rest, cur2 = _device_score_steps(Base.tail(steps), slots, params, observed, tc, ls, col, cur)
    return (contribution + rest, cur2)
end

# ---- the kernel ----------------------------------------------------------------

@kernel function _device_logjoint_kernel!(
    totals,
    plan,
    @Const(params),
    @Const(observed),
    slots,
    @Const(trip_counts),
    @Const(loop_starts),
)
    col = @index(Global)
    total, _ = _device_score_steps(plan.steps, slots, params, observed, trip_counts, loop_starts, col, Int32(1))
    @inbounds totals[col] = total
end
