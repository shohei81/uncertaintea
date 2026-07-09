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
