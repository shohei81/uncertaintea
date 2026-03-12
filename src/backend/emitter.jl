function _backend_package_symbol(model::TeaModel, target::Symbol)
    return Symbol(
        "UncertainTea",
        gpu_backend_target_name(target),
        "BackendPackage__",
        model.name,
    )
end

function _backend_bundle_symbol(model::TeaModel, target::Symbol)
    return Symbol(
        "UncertainTea",
        gpu_backend_target_name(target),
        "BackendBundle__",
        model.name,
    )
end

function _backend_module_symbol(model::TeaModel, target::Symbol)
    return Symbol(
        "UncertainTea",
        gpu_backend_target_name(target),
        "BackendModule__",
        model.name,
    )
end

function _backend_module_filename(model::TeaModel, target::Symbol)
    return gpu_backend_module_filename(_backend_module_symbol(model, target), nothing, target)
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

function _backend_step_stub(step::BackendExponentialChoicePlanStep)
    return string(
        "choice exponential address=",
        _backend_address_stub(step.address),
        " rate=",
        _backend_expr_stub(step.rate),
        " parameter_slot=",
        repr(step.parameter_slot),
    )
end

function _backend_step_stub(step::BackendGammaChoicePlanStep)
    return string(
        "choice gamma address=",
        _backend_address_stub(step.address),
        " shape=",
        _backend_expr_stub(step.shape),
        " rate=",
        _backend_expr_stub(step.rate),
        " parameter_slot=",
        repr(step.parameter_slot),
    )
end

function _backend_step_stub(step::BackendBetaChoicePlanStep)
    return string(
        "choice beta address=",
        _backend_address_stub(step.address),
        " alpha=",
        _backend_expr_stub(step.alpha),
        " beta=",
        _backend_expr_stub(step.beta),
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

function _backend_step_stub(step::BackendCategoricalChoicePlanStep)
    return string(
        "choice categorical address=",
        _backend_address_stub(step.address),
        " probabilities=",
        _backend_expr_stub(BackendTupleExpr(step.probabilities)),
        " parameter_slot=",
        repr(step.parameter_slot),
    )
end

function _backend_step_stub(step::BackendPoissonChoicePlanStep)
    return string(
        "choice poisson address=",
        _backend_address_stub(step.address),
        " lambda=",
        _backend_expr_stub(step.lambda),
        " parameter_slot=",
        repr(step.parameter_slot),
    )
end

function _backend_step_stub(step::BackendStudentTChoicePlanStep)
    return string(
        "choice studentt address=",
        _backend_address_stub(step.address),
        " nu=",
        _backend_expr_stub(step.nu),
        " mu=",
        _backend_expr_stub(step.mu),
        " sigma=",
        _backend_expr_stub(step.sigma),
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
    target::Symbol,
    module_symbol::Symbol,
    entry_symbol::Symbol,
)
    return gpu_backend_stage_source_lines(
        module_symbol,
        entry_symbol,
        (),
        :backend_execute,
        target;
        metadata_lines=(string("# model = ", model.name),),
        preamble_lines=(
        string("const TARGET = :", target),
        string("const MODEL = :", model.name),
        string("const STEP_COUNT = ", length(plan.steps)),
        string("const NUMERIC_SLOTS = ", _backend_bool_vector_literal(plan.numeric_slots)),
        string("const INDEX_SLOTS = ", _backend_bool_vector_literal(plan.index_slots)),
        string("const GENERIC_SLOTS = ", _backend_bool_vector_literal(plan.generic_slots)),
        "",
        "# Lowered backend plan",
        (
            string("# ", index, ". ", _backend_step_stub(step)) for
            (index, step) in enumerate(plan.steps)
        )...,
        "",
    ),
    )
end

function _backend_bundle_layout(
    model::TeaModel,
    plan::BackendExecutionPlan,
    target::Symbol,
)
    bundle_symbol = _backend_bundle_symbol(model, target)
    module_symbol = _backend_module_symbol(model, target)
    stage_name = model.name
    entry_symbol = Symbol("execute_backend__", model.name)
    stage_filename = _backend_module_filename(model, target)
    return gpu_backend_codegen_bundle(
        target,
        bundle_symbol,
        (
            GPUBackendCodegenStage(
                stage_name,
                :backend_execute,
                entry_symbol,
                stage_filename,
                join(
                    _backend_module_source_lines(
                        model,
                        plan,
                        target,
                        module_symbol,
                        entry_symbol,
                    ),
                    "\n",
                ),
            ),
        );
        manifest_lines=(string("model = \"", model.name, "\""),),
    )
end

function backend_package_layout(model::TeaModel; target::Symbol=:gpu)
    _gpu_backend_require_target(target)
    plan = backend_execution_plan(model; target=:gpu)
    package_symbol = _backend_package_symbol(model, target)
    root_dir = lowercase(String(package_symbol))
    return gpu_backend_codegen_package_layout(
        target,
        package_symbol,
        root_dir,
        (_backend_bundle_layout(model, plan, target),),
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
