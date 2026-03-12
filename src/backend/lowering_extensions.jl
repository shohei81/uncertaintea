struct BackendLaplaceChoicePlanStep{M<:AbstractBackendExpr,S<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    scale::S
    parameter_slot::Union{Nothing,Int}
end

struct BackendInverseGammaChoicePlanStep{SH<:AbstractBackendExpr,SC<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    shape::SH
    scale::SC
    parameter_slot::Union{Nothing,Int}
end

struct BackendWeibullChoicePlanStep{SH<:AbstractBackendExpr,SC<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    shape::SH
    scale::SC
    parameter_slot::Union{Nothing,Int}
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

struct BackendMvNormalChoicePlanStep{M<:Tuple,S<:Tuple,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
    value_index::Union{Nothing,Int}
    value_length::Int
end

function _backend_lower_tuple_argument(model::TeaModel, layout::EnvironmentLayout, expr, issues::Vector{String}, context::String)
    if expr isa Expr && expr.head in (:vect, :tuple)
        lowered = map(arg -> _backend_lower_expr(model, layout, arg, issues, context), expr.args)
        any(isnothing, lowered) && return nothing
        return tuple(lowered...)
    elseif expr isa QuoteNode && (expr.value isa Tuple || expr.value isa AbstractVector)
        lowered = map(arg -> _backend_lower_expr(model, layout, QuoteNode(arg), issues, context), collect(expr.value))
        any(isnothing, lowered) && return nothing
        return tuple(lowered...)
    elseif expr isa Tuple || expr isa AbstractVector
        lowered = map(arg -> _backend_lower_expr(model, layout, QuoteNode(arg), issues, context), collect(expr))
        any(isnothing, lowered) && return nothing
        return tuple(lowered...)
    end

    lowered = _backend_lower_expr(model, layout, expr, issues, context)
    lowered isa BackendTupleExpr || begin
        _backend_issue!(issues, "mvnormal requires tuple-like backend arguments")
        return nothing
    end
    return lowered.arguments
end

function _backend_lower_mvnormal_choice_step(model::TeaModel, layout::EnvironmentLayout, step::ChoicePlanStep, issues::Vector{String})
    length(step.rhs.arguments) == 2 || begin
        _backend_issue!(issues, "mvnormal expects exactly 2 backend arguments")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    mu = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[1], issues, "mvnormal mean")
    sigma = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[2], issues, "mvnormal scale")
    (isnothing(mu) || isnothing(sigma)) && return nothing
    length(mu) == length(sigma) || begin
        _backend_issue!(issues, "mvnormal requires mean and scale vectors with the same backend length")
        return nothing
    end
    isempty(mu) && begin
        _backend_issue!(issues, "mvnormal requires at least one backend dimension")
        return nothing
    end

    value_index = nothing
    if !isnothing(step.parameter_slot)
        slot = parameterlayout(model).slots[step.parameter_slot]
        slot.transform isa VectorIdentityTransform || begin
            _backend_issue!(issues, "mvnormal backend lowering expects a vector identity transform")
            return nothing
        end
        slot.value_length == length(mu) || begin
            _backend_issue!(issues, "mvnormal backend lowering requires a parameter slot with matching vector length")
            return nothing
        end
        value_index = slot.value_index
    end
    return BackendMvNormalChoicePlanStep(
        step.binding_slot,
        address,
        mu,
        sigma,
        step.parameter_slot,
        value_index,
        length(mu),
    )
end

function _collect_backend_slot_kinds!(
    step::BackendLaplaceChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.scale, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendInverseGammaChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.shape, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.scale, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendWeibullChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.shape, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.scale, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
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

function _collect_backend_slot_kinds!(
    step::BackendMvNormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    for expr in step.mu
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    for expr in step.sigma
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end
