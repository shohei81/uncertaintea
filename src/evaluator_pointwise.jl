# Pointwise log-likelihood extraction.
#
# This mirrors the compiled scoring walk in `evaluator.jl` (`_score_compiled_steps` /
# `_score_plan_step!`) but instead of summing every choice's logpdf it *records* the
# per-observation logpdf for every observation.
#
# The observation/latent split is taken from the CONDITIONING SIGNATURE (issue #95),
# exactly as `logjoint` and the batched/device paths do: the walk runs over the
# signature-resolved compiled plan, in which a static choice constrained at inference
# time carries no parameter slot (so it reads its value from the constraints and is
# recorded), while any unconstrained static choice is a slotted latent. This keeps the
# recorded observations identical to `observation_addresses` and to the scoring paths.
# (Correctly weighting a marginalized/enumerated latent's likelihood contribution is
# issue #88; it builds on this classification and is not implemented here.)

_record_compiled_steps(::Tuple{}, env::PlanEnvironment, params::AbstractVector, constraints::ChoiceMap, records) = records

function _record_compiled_steps(steps::Tuple, env::PlanEnvironment, params::AbstractVector, constraints::ChoiceMap, records)
    _record_plan_step!(first(steps), env, params, constraints, records)
    _record_compiled_steps(Base.tail(steps), env, params, constraints, records)
    return records
end

function _record_plan_step!(
    step::CompiledChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    records,
)
    address = _concrete_address(env, step.address)
    is_constrained = isnothing(step.parameter_value_indices)
    value = if !is_constrained
        _parameter_slot_value(step.parameter_value_indices, params)
    else
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("pointwise loglikelihood requires a provided value for choice $(address)"))
        constrained_value
    end

    dist = _compiled_distribution(step, env)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    lp = logpdf(dist, value)
    if is_constrained
        push!(records, address => lp)
    end
    return records
end

function _record_plan_step!(
    step::CompiledDeterministicPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    records,
)
    _environment_set!(env, step.binding_slot, _eval_compiled_expr(env, step.expr))
    return records
end

function _record_plan_step!(
    step::CompiledLoopPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    records,
)
    iterable = _eval_compiled_expr(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing

    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        _record_compiled_steps(step.body, env, params, constraints, records)
    end

    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return records
end

function _record_execution!(records, model::TeaModel, params::AbstractVector, args::Tuple, constraints::ChoiceMap)
    resolved = _resolve_signature_plan(model, constraints)
    plan = resolved.plan
    compiled_plan = resolved.compiled
    args = _complete_model_args(model, args)

    env = PlanEnvironment(plan.environment_layout)
    for (slot, value) in zip(plan.environment_layout.argument_slots, args)
        _environment_set!(env, slot, value)
    end

    _record_compiled_steps(compiled_plan.steps, env, params, constraints, records)
    return records
end

# Initial parameter vector matching the signature layout (not the default
# layout): observation_addresses only needs a validly-shaped latent vector to
# walk the plan and read the constrained addresses, so any prior draw of the
# signature latents will do.
function _signature_initial_parameters(
    model::TeaModel,
    args::Tuple,
    resolved,
    constraints::ChoiceMap;
    rng::AbstractRNG=Random.default_rng(),
)
    trace, _ = generate(model, args, constraints; rng=rng)
    layout = resolved.plan.parameter_layout
    params = Vector{Float64}(undef, parametervaluecount(layout))
    for slot in layout.slots
        _write_slot_value!(params, slot, trace[_static_address(slot.address)])
    end
    return params
end

"""
    observation_addresses(model, args=(), constraints=choicemap()) -> Vector

Return the ordered list of observation choice addresses in execution-plan order. Under the
constraint-driven rule (issue #95, `docs/constraint-driven-conditioning.md`) an observation
is exactly a choice whose address is constrained-and-present, regardless of whether the
choice is bound; binding is orthogonal. This order defines the column order of
`pointwise_loglikelihood` and is deterministic across draws.
"""
function observation_addresses(model::TeaModel, args::Tuple=(), constraints::ChoiceMap=choicemap())
    resolved = _resolve_signature_plan(model, constraints)
    params = _signature_initial_parameters(model, args, resolved, constraints)
    records = Pair{Any,Float64}[]
    _record_execution!(records, model, params, args, constraints)
    return Any[first(r) for r in records]
end

"""
    pointwise_loglikelihood(model, args, constraints, chains) -> Matrix{Float64}

Compute the pointwise (per-observation) log-likelihood matrix of size `S_total x N_obs`,
where `S_total` is the pooled number of posterior draws across all chains and `N_obs` is the
number of constrained (observation) choices. Each draw uses its *constrained* parameter
vector (`chain.constrained_samples`). Columns follow `observation_addresses(model, args, constraints)`.
"""
function pointwise_loglikelihood(model::TeaModel, args::Tuple, constraints::ChoiceMap, chains)
    addresses = observation_addresses(model, args, constraints)
    n_obs = length(addresses)
    column_of = Dict{Any,Int}()
    for (i, a) in enumerate(addresses)
        column_of[a] = i
    end

    total = 0
    for chain in chains.chains
        total += size(chain.constrained_samples, 2)
    end

    ll = Matrix{Float64}(undef, total, n_obs)
    records = Pair{Any,Float64}[]
    s = 0
    for chain in chains.chains
        samples = chain.constrained_samples
        for j = 1:size(samples, 2)
            s += 1
            empty!(records)
            _record_execution!(records, model, view(samples, :, j), args, constraints)
            for (address, lp) in records
                ll[s, column_of[address]] = lp
            end
        end
    end
    return ll
end
