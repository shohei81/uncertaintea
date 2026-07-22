# Lowering of static models to the backend expression IR: discrete families (bernoulli, poisson, geometric, binomial, negativebinomial, categorical).

struct BackendBernoulliChoicePlanStep{P<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    probability::P
    parameter_slot::Union{Nothing,Int}
end

struct BackendCategoricalChoicePlanStep{P<:Tuple,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    probabilities::P
    parameter_slot::Union{Nothing,Int}
end

struct BackendPoissonChoicePlanStep{L<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    lambda::L
    parameter_slot::Union{Nothing,Int}
end

function _collect_backend_slot_kinds!(
    step::BackendBernoulliChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.probability, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendCategoricalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    for probability in step.probabilities
        _mark_backend_numeric_expr_slots!(probability, numeric_slots, index_slots, generic_slots)
    end
    isnothing(step.binding_slot) || _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendPoissonChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.lambda, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

# marginalize=:enumerate (docs/discrete-enumeration.md, issue #13 PR-4): the
# step owns the lowered plan SUFFIX and scores it once per compile-time
# support value, combining with a per-column logsumexp. `probabilities` holds
# the family's lowered numeric argument exprs (bernoulli: `(p,)`; categorical:
# the K literal entries); `parameter_slot` is always `nothing` (an enumerated
# latent has no slot) and exists only for the choice-step field convention.
struct BackendMarginalizeChoicePlanStep{P<:Tuple,S<:Tuple,B<:Tuple,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    family::Symbol
    probabilities::P
    support::S
    body::B
    parameter_slot::Union{Nothing,Int}
end

# Nested enumerated latents multiply the suffix cost; the backend rejects
# support products beyond this bound honestly instead of compiling them.
const BACKEND_MARGINALIZE_SUPPORT_LIMIT = 32

function _backend_marginalize_support_product(steps)
    product = 1
    for step in steps
        if step isa BackendMarginalizeChoicePlanStep
            product *= length(step.support) * _backend_marginalize_support_product(step.body)
        elseif step isa BackendLoopPlanStep
            product *= _backend_marginalize_support_product(step.body)
        end
    end
    return product
end

function _collect_backend_slot_kinds!(
    step::BackendMarginalizeChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    for probability in step.probabilities
        _mark_backend_numeric_expr_slots!(probability, numeric_slots, index_slots, generic_slots)
    end
    if !isnothing(step.binding_slot)
        # bernoulli branches bind 0/1 numerics; categorical branches bind the
        # category itself, which integer consumers (binomial trials, loop
        # bounds) require to stay an index slot
        if step.family === :categorical
            _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
        else
            _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
        end
    end
    for inner in step.body
        _collect_backend_slot_kinds!(inner, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _backend_lower_marginalize_choice_step(
    model::TeaModel,
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    suffix::AbstractVector,
    issues::Vector{String},
)
    isnothing(step.parameter_slot) || begin
        _backend_issue!(issues, "marginalized choices carry no parameter slot")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    family = step.rhs.family
    probabilities = if family === :bernoulli
        length(step.rhs.arguments) == 1 || begin
            _backend_issue!(issues, "bernoulli expects exactly 1 backend argument")
            return nothing
        end
        probability = _backend_lower_expr(model, layout, step.rhs.arguments[1], issues, "bernoulli probability")
        isnothing(probability) && return nothing
        (probability,)
    elseif family === :categorical
        lowered = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[1], issues, "categorical probabilities")
        isnothing(lowered) && return nothing
        lowered
    else
        _backend_issue!(
            issues,
            "marginalize=:enumerate supports bernoulli and categorical in backend lowering, got `$family`",
        )
        return nothing
    end
    support = family === :bernoulli ? (false, true) : ntuple(identity, length(probabilities))

    body = _backend_lower_steps(model, layout, parameter_layout, suffix, issues)
    any(isnothing, body) && return nothing
    body_steps = tuple(body...)
    support_product = length(support) * _backend_marginalize_support_product(body_steps)
    support_product <= BACKEND_MARGINALIZE_SUPPORT_LIMIT || begin
        _backend_issue!(
            issues,
            "nested marginalize=:enumerate support product $support_product exceeds the backend limit " *
            "$BACKEND_MARGINALIZE_SUPPORT_LIMIT",
        )
        return nothing
    end
    return BackendMarginalizeChoicePlanStep(
        step.binding_slot,
        address,
        family,
        probabilities,
        support,
        body_steps,
        nothing,
    )
end

struct BackendBinomialChoicePlanStep{N<:AbstractBackendExpr,P<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    trials::N
    probability::P
    parameter_slot::Union{Nothing,Int}
end

struct BackendGeometricChoicePlanStep{P<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    probability::P
    parameter_slot::Union{Nothing,Int}
end

struct BackendNegativeBinomialChoicePlanStep{R<:AbstractBackendExpr,P<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    successes::R
    probability::P
    parameter_slot::Union{Nothing,Int}
end

function _collect_backend_slot_kinds!(
    step::BackendBinomialChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_index_expr_slots!(step.trials, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.probability, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendGeometricChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.probability, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendNegativeBinomialChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.successes, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.probability, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end
