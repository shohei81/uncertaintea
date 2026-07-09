# One fused KernelAbstractions kernel that computes the gradient of the UNCONSTRAINED
# logjoint by scalar forward-mode differentiation. The `ndrange` is 2D:
# `(parameter_index, batch_index)`. Thread `(p, b)` re-walks the SAME device plan as
# the logjoint kernel, but in `DeviceDual` numbers seeded so that only unconstrained
# parameter row `p` of column `b` carries derivative 1. Its accumulated total is a dual
# whose `.deriv` is d(logjoint)/d(param p) and whose `.value` is the logjoint itself.
#
# Cost is O(P * plan) per column -- the same FLOP class as the CPU manual forward
# accumulation in `manual_gradient_scoring.jl`, whose per-distribution derivative
# FORMULAS this differentiation reproduces exactly (forward-mode through the same
# logpdfs and the same unconstrained transform + log-abs-det).
#
# Per-thread slot scratch lives in a 3D `DeviceDual` buffer laid out as
# `slots[slot, parameter_index, batch_index]`; each `(p, b)` thread owns the
# `slots[:, p, b]` column and never races another thread. `params`/`observed` are the
# same plain-`T` buffers the logjoint kernel uses, indexed by the batch column `b`.
#
# `math.jl`'s logpdfs and `_device_transform` are reused verbatim: they are written
# `where {T}` with no `<:Real` bound, so `T` binds to `DeviceDual{...}` and the whole
# density (including the transform log-abs-det) differentiates through the duals. The
# per-choice arguments are `promote`d so the three positional arguments share the one
# `DeviceDual{...}` type the logpdf signatures require.

# ---- dual expression evaluation ------------------------------------------------

@inline _device_grad_eval(e::DeviceLiteralExpr, slots, pidx, b) = e.value
@inline _device_grad_eval(e::DeviceSlotExpr, slots, pidx, b) = @inbounds slots[e.slot, pidx, b]
@inline _device_grad_eval(e::DevicePrimitiveExpr{Op}, slots, pidx, b) where {Op} =
    _device_apply(Val(Op), _device_grad_eval_args(e.args, slots, pidx, b)...)

@inline _device_grad_eval_args(::Tuple{}, slots, pidx, b) = ()
@inline _device_grad_eval_args(t::Tuple, slots, pidx, b) =
    (_device_grad_eval(first(t), slots, pidx, b), _device_grad_eval_args(Base.tail(t), slots, pidx, b)...)

@inline function _device_grad_store_binding!(slots, binding_slot::Int32, value, pidx, b)
    if binding_slot > Int32(0)
        @inbounds slots[binding_slot, pidx, b] = value
    end
    return nothing
end

# Returns (value_dual, logabsdet_dual, new_cursor). Latent values seed derivative 1 on
# the differentiation target row `pidx`; observed values are constants (derivative 0).
@inline function _device_grad_choice_value(step, params, observed, pidx, b, cursor::Int32, ::Type{TD}) where {TD}
    T = _device_dual_basetype(TD)
    if step.value_source > Int32(0)
        raw = @inbounds params[step.value_source, b]
        seed = ifelse(step.value_source == Int32(pidx), one(T), zero(T))
        u = DeviceDual{T}(convert(T, raw), seed)
        c, lad = _device_transform(step.transform, u)
        return (c, lad, cursor)
    else
        v = @inbounds observed[cursor, b]
        z = zero(T)
        return (DeviceDual{T}(convert(T, v), z), DeviceDual{T}(z, z), cursor + Int32(1))
    end
end

# ---- per-step dual scoring ------------------------------------------------------

@inline function _device_grad_score_step(step::DeviceNormalChoiceStep, slots, params, observed, tc, ls, pidx, b, cursor)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    m, s, v = promote(mu, sigma, value)
    return (_device_normal_logpdf(m, s, v) + lad, cur)
end

@inline function _device_grad_score_step(step::DeviceNoncenteredNormalChoiceStep, slots, params, observed, tc, ls, pidx, b, cursor)
    T = _device_dual_basetype(eltype(slots))
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    raw = @inbounds params[step.value_source, b]
    seed = ifelse(step.value_source == Int32(pidx), one(T), zero(T))
    z = DeviceDual{T}(convert(T, raw), seed)
    m, s, zz = promote(mu, sigma, z)
    _device_grad_store_binding!(slots, step.binding_slot, m + s * zz, pidx, b)
    return (_device_normal_logpdf(zero(zz), one(zz), zz), cursor)
end

@inline function _device_grad_score_step(step::DeviceLognormalChoiceStep, slots, params, observed, tc, ls, pidx, b, cursor)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    m, s, v = promote(mu, sigma, value)
    return (_device_lognormal_logpdf(m, s, v) + lad, cur)
end

@inline function _device_grad_score_step(step::DeviceExponentialChoiceStep, slots, params, observed, tc, ls, pidx, b, cursor)
    rate = _device_grad_eval(step.rate, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    r, v = promote(rate, value)
    return (_device_exponential_logpdf(r, v) + lad, cur)
end

@inline function _device_grad_score_step(step::DeviceGammaChoiceStep, slots, params, observed, tc, ls, pidx, b, cursor)
    shape = _device_grad_eval(step.shape, slots, pidx, b)
    rate = _device_grad_eval(step.rate, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    sh, r, v = promote(shape, rate, value)
    return (_device_gamma_logpdf(sh, r, v) + lad, cur)
end

@inline function _device_grad_score_step(step::DeviceLaplaceChoiceStep, slots, params, observed, tc, ls, pidx, b, cursor)
    loc = _device_grad_eval(step.loc, slots, pidx, b)
    scale = _device_grad_eval(step.scale, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    l, s, v = promote(loc, scale, value)
    return (_device_laplace_logpdf(l, s, v) + lad, cur)
end

@inline function _device_grad_score_step(step::DeviceBetaChoiceStep, slots, params, observed, tc, ls, pidx, b, cursor)
    alpha = _device_grad_eval(step.alpha, slots, pidx, b)
    beta = _device_grad_eval(step.beta, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    al, be, v = promote(alpha, beta, value)
    return (_device_beta_logpdf(al, be, v) + lad, cur)
end

@inline function _device_grad_score_step(step::DeviceBernoulliChoiceStep, slots, params, observed, tc, ls, pidx, b, cursor)
    p = _device_grad_eval(step.probability, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    pp, v = promote(p, value)
    return (_device_bernoulli_logpdf(pp, v) + lad, cur)
end

@inline function _device_grad_score_step(step::DevicePoissonChoiceStep, slots, params, observed, tc, ls, pidx, b, cursor)
    lambda = _device_grad_eval(step.lambda, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    lam, v = promote(lambda, value)
    return (_device_poisson_logpdf(lam, v) + lad, cur)
end

@inline function _device_grad_score_step(step::DeviceDeterministicStep, slots, params, observed, tc, ls, pidx, b, cursor)
    @inbounds slots[step.binding_slot, pidx, b] = _device_grad_eval(step.expr, slots, pidx, b)
    return (zero(eltype(slots)), cursor)
end

@inline function _device_grad_score_step(step::DeviceLoopStep, slots, params, observed, tc, ls, pidx, b, cursor)
    TD = eltype(slots)
    count = @inbounds tc[step.loop_id]
    start = @inbounds ls[step.loop_id]
    total = zero(TD)
    cur = cursor
    for t = Int32(0):(count-Int32(1))
        if step.iterator_slot > Int32(0)
            @inbounds slots[step.iterator_slot, pidx, b] = TD(start + t)
        end
        contribution, cur = _device_grad_score_steps(step.body, slots, params, observed, tc, ls, pidx, b, cur)
        total += contribution
    end
    return (total, cur)
end

# ---- recursive dual step-tuple walk --------------------------------------------

@inline _device_grad_score_steps(::Tuple{}, slots, params, observed, tc, ls, pidx, b, cursor) =
    (zero(eltype(slots)), cursor)

@inline function _device_grad_score_steps(steps::Tuple, slots, params, observed, tc, ls, pidx, b, cursor)
    contribution, cur = _device_grad_score_step(first(steps), slots, params, observed, tc, ls, pidx, b, cursor)
    rest, cur2 = _device_grad_score_steps(Base.tail(steps), slots, params, observed, tc, ls, pidx, b, cur)
    return (contribution + rest, cur2)
end

# ---- the kernel ----------------------------------------------------------------

@kernel function _device_gradient_kernel!(
    totals,
    gradients,
    plan,
    @Const(params),
    @Const(observed),
    slots,
    @Const(trip_counts),
    @Const(loop_starts),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    total, _ = _device_grad_score_steps(
        plan.steps, slots, params, observed, trip_counts, loop_starts, pidx, b, Int32(1),
    )
    @inbounds gradients[pidx, b] = _device_dual_deriv(total)
    if pidx == 1
        @inbounds totals[b] = _device_dual_value(total)
    end
end
