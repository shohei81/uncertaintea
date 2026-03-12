_backend_package_target_name(::Val{:gpu}) = "GPU"

function _backend_package_target_name(target::Symbol)
    target === :gpu || throw(ArgumentError("unsupported backend package target $(target)"))
    return _backend_package_target_name(Val(target))
end

function _backend_package_symbol(model::TeaModel, target::Symbol)
    return Symbol(
        "UncertainTea",
        _backend_package_target_name(target),
        "BackendPackage__",
        model.name,
    )
end

function _backend_bundle_symbol(model::TeaModel, target::Symbol)
    return Symbol(
        "UncertainTea",
        _backend_package_target_name(target),
        "BackendBundle__",
        model.name,
    )
end

function _backend_module_symbol(model::TeaModel, target::Symbol)
    return Symbol(
        "UncertainTea",
        _backend_package_target_name(target),
        "BackendModule__",
        model.name,
    )
end

function _backend_module_filename(model::TeaModel, target::Symbol)
    target === :gpu || throw(ArgumentError("unsupported backend module target $(target)"))
    return string(lowercase(String(_backend_module_symbol(model, target))), ".jl")
end

_backend_bool_literal(value::Bool) = value ? "true" : "false"

function _backend_bool_vector_literal(bits::BitVector)
    return string("[", join((_backend_bool_literal(bit) for bit in bits), ", "), "]")
end

_backend_expr_stub(expr::BackendLiteralExpr) = repr(expr.value)
_backend_expr_stub(expr::BackendSlotExpr) = string("slot(", expr.slot, ")")

function _backend_expr_stub(expr::BackendPrimitiveExpr)
    arguments = join((_backend_expr_stub(arg) for arg in expr.arguments), ", ")
    return string(expr.op, "(", arguments, ")")
end

_backend_expr_stub(expr::BackendTupleExpr) =
    string("tuple(", join((_backend_expr_stub(arg) for arg in expr.arguments), ", "), ")")

_backend_expr_stub(expr::BackendBlockExpr) =
    string("block(", join((_backend_expr_stub(arg) for arg in expr.arguments), "; "), ")")

_backend_address_part_stub(part::BackendAddressLiteralPart) = repr(part.value)
_backend_address_part_stub(part::BackendAddressExprPart) = _backend_expr_stub(part.expr)

function _backend_address_stub(address::BackendAddressSpec)
    return string(
        "(",
        join((_backend_address_part_stub(part) for part in address.parts), " => "),
        ")",
    )
end

function _backend_step_stub(step::BackendNormalChoicePlanStep)
    return string(
        "choice normal address=",
        _backend_address_stub(step.address),
        " mu=",
        _backend_expr_stub(step.mu),
        " sigma=",
        _backend_expr_stub(step.sigma),
        " parameter_slot=",
        repr(step.parameter_slot),
    )
end

function _backend_step_stub(step::BackendLognormalChoicePlanStep)
    return string(
        "choice lognormal address=",
        _backend_address_stub(step.address),
        " mu=",
        _backend_expr_stub(step.mu),
        " sigma=",
        _backend_expr_stub(step.sigma),
        " parameter_slot=",
        repr(step.parameter_slot),
    )
end

function _backend_step_stub(step::BackendBernoulliChoicePlanStep)
    return string(
        "choice bernoulli address=",
        _backend_address_stub(step.address),
        " probability=",
        _backend_expr_stub(step.probability),
        " parameter_slot=",
        repr(step.parameter_slot),
    )
end

function _backend_step_stub(step::BackendDeterministicPlanStep)
    return string(
        "deterministic slot=",
        step.binding_slot,
        " expr=",
        _backend_expr_stub(step.expr),
    )
end

function _backend_step_stub(step::BackendLoopPlanStep)
    return string(
        "loop iterator_slot=",
        step.iterator_slot,
        " iterable=",
        _backend_expr_stub(step.iterable),
        " body_steps=",
        length(step.body),
    )
end

function _backend_module_source_lines(
    model::TeaModel,
    plan::BackendExecutionPlan,
    module_symbol::Symbol,
    entry_symbol::Symbol,
)
    lines = String[
        string("module ", module_symbol),
        string("const TARGET = :", plan.target),
        string("const MODEL = :", model.name),
        string("const STEP_COUNT = ", length(plan.steps)),
        string("const NUMERIC_SLOTS = ", _backend_bool_vector_literal(plan.numeric_slots)),
        string("const INDEX_SLOTS = ", _backend_bool_vector_literal(plan.index_slots)),
        string("const GENERIC_SLOTS = ", _backend_bool_vector_literal(plan.generic_slots)),
        "",
        "# Lowered backend plan",
    ]
    for (index, step) in enumerate(plan.steps)
        push!(lines, string("# ", index, ". ", _backend_step_stub(step)))
    end
    push!(lines, "")
    push!(lines, string("function ", entry_symbol, "()"))
    push!(lines, "    return nothing")
    push!(lines, "end")
    push!(lines, "end")
    return Tuple(lines)
end

function _backend_manifest_contents(
    model::TeaModel,
    target::Symbol,
    bundle_symbol::Symbol,
    stage_name::Symbol,
    stage_filename::String,
)
    lines = String[
        string("model = \"", model.name, "\""),
        string("target = \"", target, "\""),
        string("bundle = \"", bundle_symbol, "\""),
        "count = 1",
        "[[stage]]",
        "index = 1",
        string("name = \"", stage_name, "\""),
        string("file = \"", stage_filename, "\""),
    ]
    return join(lines, "\n")
end

function _backend_bundle_layout(
    model::TeaModel,
    plan::BackendExecutionPlan,
    target::Symbol,
    root_dir::String,
)
    bundle_symbol = _backend_bundle_symbol(model, target)
    module_symbol = _backend_module_symbol(model, target)
    stage_name = model.name
    entry_symbol = Symbol("execute_backend__", model.name)
    stage_filename = _backend_module_filename(model, target)
    return gpu_backend_bundle_layout(
        target,
        bundle_symbol,
        GPUBackendManifestFile(
            joinpath(
                root_dir,
                string(lowercase(String(bundle_symbol)), "__manifest.toml"),
            ),
            _backend_manifest_contents(
                model,
                target,
                bundle_symbol,
                stage_name,
                stage_filename,
            ),
        ),
        (
            GPUBackendStageFile(
                stage_name,
                joinpath(root_dir, stage_filename),
                join(
                    _backend_module_source_lines(
                        model,
                        plan,
                        module_symbol,
                        entry_symbol,
                    ),
                    "\n",
                ),
            ),
        ),
    )
end

function backend_package_layout(model::TeaModel; target::Symbol=:gpu)
    plan = backend_execution_plan(model; target=target)
    package_symbol = _backend_package_symbol(model, target)
    root_dir = lowercase(String(package_symbol))
    return gpu_backend_package_layout(
        target,
        package_symbol,
        root_dir,
        (_backend_bundle_layout(model, plan, target, root_dir),),
    )
end

function emit_backend_package(
    model::TeaModel,
    output_root::AbstractString;
    target::Symbol=:gpu,
    overwrite::Bool=false,
)
    return emit_gpu_backend_package(
        backend_package_layout(model; target=target),
        output_root;
        overwrite=overwrite,
    )
end
