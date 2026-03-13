_tempered_smc_nuts_bool_literal(value::Bool) = value ? "true" : "false"

function _tempered_smc_nuts_bool_vector_literal(bits::BitVector)
    return string("[", join((_tempered_smc_nuts_bool_literal(bit) for bit in bits), ", "), "]")
end

function _tempered_smc_nuts_int_vector_literal(values::AbstractVector{<:Integer})
    return string("[", join((string(value) for value in values), ", "), "]")
end

function _tempered_smc_nuts_package_symbol(model::TeaModel, target::Symbol)
    return Symbol(
        "UncertainTea",
        gpu_backend_target_name(target),
        "TemperedSMCNUTSPackage__",
        model.name,
    )
end

function _tempered_smc_nuts_bundle_symbol(model::TeaModel, target::Symbol)
    return Symbol(
        "UncertainTea",
        gpu_backend_target_name(target),
        "TemperedSMCNUTSBundle__",
        model.name,
    )
end

function _tempered_smc_nuts_module_symbol(
    model::TeaModel,
    stage_kind::Symbol,
    target::Symbol,
)
    return Symbol(
        "UncertainTea",
        gpu_backend_target_name(target),
        "TemperedSMCNUTSModule__",
        model.name,
        "__",
        stage_kind,
    )
end

function _tempered_smc_nuts_entry_symbol(
    model::TeaModel,
    stage_kind::Symbol,
)
    return Symbol("execute_", stage_kind, "__", model.name)
end

_tempered_smc_nuts_stage_kind(::TemperedNUTSIdleDescriptor) = :smc_tempered_nuts_idle
_tempered_smc_nuts_stage_kind(::TemperedNUTSExpandDescriptor) = :smc_tempered_nuts_expand
_tempered_smc_nuts_stage_kind(::TemperedNUTSMergeDescriptor) = :smc_tempered_nuts_merge
_tempered_smc_nuts_stage_kind(::TemperedNUTSDoneDescriptor) = :smc_tempered_nuts_done

function _tempered_smc_nuts_descriptor_metadata_lines(
    descriptor::TemperedNUTSIdleDescriptor,
    workspace::TemperedNUTSMoveWorkspace,
)
    return (
        string("const PARAMETER_COUNT = ", workspace.parameter_total),
        string("const PARTICLE_COUNT = ", workspace.num_particles),
        "const SCHEDULER_PHASE = :idle",
    )
end

function _tempered_smc_nuts_descriptor_metadata_lines(
    descriptor::TemperedNUTSExpandDescriptor,
    workspace::TemperedNUTSMoveWorkspace,
)
    return (
        string("const PARAMETER_COUNT = ", workspace.parameter_total),
        string("const PARTICLE_COUNT = ", workspace.num_particles),
        "const SCHEDULER_PHASE = :expand",
        string("const ACTIVE_DEPTH = ", descriptor.block.ir.active_depth),
        string("const ACTIVE_DEPTH_COUNT = ", descriptor.block.ir.active_depth_count),
        string("const REMAINING_STEPS = ", descriptor.remaining_steps),
        string(
            "const ACTIVE_PARTICLES = ",
            _tempered_smc_nuts_bool_vector_literal(descriptor.active_particles),
        ),
        string(
            "const STEP_DIRECTIONS = ",
            _tempered_smc_nuts_int_vector_literal(descriptor.directions),
        ),
    )
end

function _tempered_smc_nuts_descriptor_metadata_lines(
    descriptor::TemperedNUTSMergeDescriptor,
    workspace::TemperedNUTSMoveWorkspace,
)
    return (
        string("const PARAMETER_COUNT = ", workspace.parameter_total),
        string("const PARTICLE_COUNT = ", workspace.num_particles),
        "const SCHEDULER_PHASE = :merge",
        string("const ACTIVE_DEPTH = ", descriptor.block.ir.active_depth),
        string("const ACTIVE_DEPTH_COUNT = ", descriptor.block.ir.active_depth_count),
        string(
            "const ACTIVE_PARTICLES = ",
            _tempered_smc_nuts_bool_vector_literal(descriptor.active_particles),
        ),
    )
end

function _tempered_smc_nuts_descriptor_metadata_lines(
    descriptor::TemperedNUTSDoneDescriptor,
    workspace::TemperedNUTSMoveWorkspace,
)
    return (
        string("const PARAMETER_COUNT = ", workspace.parameter_total),
        string("const PARTICLE_COUNT = ", workspace.num_particles),
        "const SCHEDULER_PHASE = :done",
    )
end

function _tempered_smc_nuts_argument_declarations(target::Symbol)
    return (
        gpu_backend_buffer_argument_declaration(target, :control),
        gpu_backend_buffer_argument_declaration(target, :particles),
        gpu_backend_buffer_argument_declaration(target, :scratch),
    )
end

function _tempered_smc_nuts_codegen_stage(
    model::TeaModel,
    workspace::TemperedNUTSMoveWorkspace,
    descriptor::AbstractTemperedNUTSSchedulerDescriptor,
    target::Symbol,
)
    stage_kind = _tempered_smc_nuts_stage_kind(descriptor)
    entry_symbol = _tempered_smc_nuts_entry_symbol(model, stage_kind)
    module_symbol = _tempered_smc_nuts_module_symbol(model, stage_kind, target)
    filename = gpu_backend_module_filename(module_symbol, entry_symbol, target)
    source_blob = gpu_backend_stage_source_blob(
        module_symbol,
        entry_symbol,
        _tempered_smc_nuts_argument_declarations(target),
        stage_kind,
        target;
        metadata_lines=_tempered_smc_nuts_descriptor_metadata_lines(descriptor, workspace),
        body_lines=(
            string("    # tempered SMC NUTS stage stub for ", model.name),
            "    return nothing",
        ),
    )
    return GPUBackendCodegenStage(stage_kind, stage_kind, entry_symbol, filename, source_blob)
end

function _tempered_smc_nuts_all_descriptors(workspace::TemperedNUTSMoveWorkspace)
    idle = _tempered_nuts_scheduler_descriptor(TemperedNUTSIdleBlock(TemperedNUTSIdleIR()))
    expand_ir = TemperedNUTSExpandIR(
        workspace.scheduler.active_depth,
        workspace.scheduler.active_depth_count,
        workspace.scheduler.remaining_steps,
        copy(workspace.scheduler.cohort_active),
        copy(workspace.control.directions),
    )
    expand = _tempered_nuts_scheduler_descriptor(_tempered_nuts_scheduler_block(expand_ir))
    merge_ir = TemperedNUTSMergeIR(
        workspace.scheduler.active_depth,
        workspace.scheduler.active_depth_count,
        copy(workspace.scheduler.cohort_active),
    )
    merge = _tempered_nuts_scheduler_descriptor(_tempered_nuts_scheduler_block(merge_ir))
    done = _tempered_nuts_scheduler_descriptor(TemperedNUTSDoneBlock(TemperedNUTSDoneIR()))
    return (idle, expand, merge, done)
end

function tempered_smc_nuts_codegen_bundle(
    model::TeaModel,
    workspace::TemperedNUTSMoveWorkspace;
    target::Symbol=:gpu,
)
    descriptors = _tempered_smc_nuts_all_descriptors(workspace)
    stages = Tuple(
        _tempered_smc_nuts_codegen_stage(model, workspace, descriptor, target)
        for descriptor in descriptors
    )
    return gpu_backend_codegen_bundle(
        target,
        _tempered_smc_nuts_bundle_symbol(model, target),
        stages;
        manifest_lines=(
            string("model = \"", model.name, "\""),
            string("current_phase = \"", workspace.scheduler.phase, "\""),
        ),
    )
end

function tempered_smc_nuts_package_layout(
    model::TeaModel,
    workspace::TemperedNUTSMoveWorkspace;
    target::Symbol=:gpu,
)
    return gpu_backend_codegen_package_layout(
        target,
        _tempered_smc_nuts_package_symbol(model, target),
        lowercase(String(_tempered_smc_nuts_package_symbol(model, target))),
        (tempered_smc_nuts_codegen_bundle(model, workspace; target=target),),
    )
end

function tempered_smc_nuts_package_layout(
    model::TeaModel,
    particles::AbstractMatrix,
    args=(),
    constraints=choicemap();
    target::Symbol=:gpu,
)
    workspace = TemperedNUTSMoveWorkspace(model, particles, args, constraints)
    return tempered_smc_nuts_package_layout(model, workspace; target=target)
end

function emit_tempered_smc_nuts_package(
    model::TeaModel,
    workspace::TemperedNUTSMoveWorkspace,
    output_root::AbstractString;
    target::Symbol=:gpu,
    overwrite::Bool=false,
)
    return emit_gpu_backend_package(
        tempered_smc_nuts_package_layout(model, workspace; target=target),
        output_root;
        overwrite=overwrite,
    )
end

function emit_tempered_smc_nuts_package(
    model::TeaModel,
    particles::AbstractMatrix,
    output_root::AbstractString,
    args=(),
    constraints=choicemap();
    target::Symbol=:gpu,
    overwrite::Bool=false,
)
    workspace = TemperedNUTSMoveWorkspace(model, particles, args, constraints)
    return emit_tempered_smc_nuts_package(
        model,
        workspace,
        output_root;
        target=target,
        overwrite=overwrite,
    )
end
