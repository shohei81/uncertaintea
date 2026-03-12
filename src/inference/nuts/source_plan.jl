@enum BatchedNUTSKernelSourceBackend::UInt8 begin
    NUTSKernelCPUSource = 0
    NUTSKernelMetalSource = 1
    NUTSKernelCUDASource = 2
end

struct BatchedNUTSKernelSourceArgument
    argument::BatchedNUTSKernelArtifactArgument
    declaration::String
end

struct BatchedNUTSKernelSourceStage{S,C,D,H,L}
    artifact_stage::S
    backend::BatchedNUTSKernelSourceBackend
    source_module::Symbol
    source_entry::Symbol
    constant_arguments::C
    device_arguments::D
    shared_arguments::H
    source_lines::L
end

struct BatchedNUTSKernelSourcePlan{P,S}
    target::Symbol
    artifact_plan::P
    backend::BatchedNUTSKernelSourceBackend
    source_module::Symbol
    stages::S
end

_batched_nuts_source_artifact_plan(plan::BatchedNUTSKernelSourcePlan) = plan.artifact_plan
_batched_nuts_source_backend(plan::BatchedNUTSKernelSourcePlan) = plan.backend
_batched_nuts_source_module(plan::BatchedNUTSKernelSourcePlan) = plan.source_module
_batched_nuts_source_stages(plan::BatchedNUTSKernelSourcePlan) = plan.stages

_batched_nuts_source_artifact_stage(stage::BatchedNUTSKernelSourceStage) = stage.artifact_stage
_batched_nuts_source_backend(stage::BatchedNUTSKernelSourceStage) = stage.backend
_batched_nuts_source_module(stage::BatchedNUTSKernelSourceStage) = stage.source_module
_batched_nuts_source_entry(stage::BatchedNUTSKernelSourceStage) = stage.source_entry
_batched_nuts_source_constant_arguments(stage::BatchedNUTSKernelSourceStage) =
    stage.constant_arguments
_batched_nuts_source_device_arguments(stage::BatchedNUTSKernelSourceStage) =
    stage.device_arguments
_batched_nuts_source_shared_arguments(stage::BatchedNUTSKernelSourceStage) =
    stage.shared_arguments
_batched_nuts_source_lines(stage::BatchedNUTSKernelSourceStage) = stage.source_lines

function _batched_nuts_source_argument_symbol(argument::BatchedNUTSKernelSourceArgument)
    return _batched_nuts_artifact_argument_symbol(argument.argument)
end

function _batched_nuts_source_stage_kind(
    artifact_stage::BatchedNUTSKernelArtifactStage,
)
    return _batched_nuts_executor_kernel_symbol(
        _batched_nuts_codegen_executor_stage(
            _batched_nuts_artifact_codegen_stage(artifact_stage),
        ),
    )
end

_batched_nuts_source_backend(::Val{:gpu}) = NUTSKernelCPUSource
_batched_nuts_source_backend(::Val{:metal}) = NUTSKernelMetalSource
_batched_nuts_source_backend(::Val{:cuda}) = NUTSKernelCUDASource

function _batched_nuts_source_backend(target::Symbol)
    return _batched_nuts_source_backend(Val(_gpu_backend_require_target(target)))
end

_batched_nuts_source_module(::Val{:gpu}) = :UncertainTeaCPUSources
_batched_nuts_source_module(::Val{:metal}) = :UncertainTeaMetalSources
_batched_nuts_source_module(::Val{:cuda}) = :UncertainTeaCUDASources

function _batched_nuts_source_module(target::Symbol)
    return _batched_nuts_source_module(Val(_gpu_backend_require_target(target)))
end

function _batched_nuts_source_argument_declaration(
    target::Symbol,
    argument::BatchedNUTSKernelArtifactArgument,
)
    return gpu_backend_buffer_argument_declaration(
        target,
        _batched_nuts_artifact_argument_symbol(argument),
    )
end

function _batched_nuts_source_arguments(target::Symbol, arguments::Tuple)
    return Tuple(
        BatchedNUTSKernelSourceArgument(
            argument,
            _batched_nuts_source_argument_declaration(target, argument),
        ) for argument in arguments
    )
end

function _batched_nuts_source_lines(
    source_module::Symbol,
    source_entry::Symbol,
    stage_kind::Symbol,
    target::Symbol,
    constant_arguments::Tuple,
    device_arguments::Tuple,
    shared_arguments::Tuple,
)
    return gpu_backend_stage_source_lines(
        source_module,
        source_entry,
        (
            map(argument -> argument.declaration, constant_arguments)...,
            map(argument -> argument.declaration, device_arguments)...,
            map(argument -> argument.declaration, shared_arguments)...,
        ),
        stage_kind,
        target;
        metadata_lines=(
            string("# constant_args = ", length(constant_arguments)),
            string("# device_args = ", length(device_arguments)),
            string("# shared_args = ", length(shared_arguments)),
        ),
    )
end

function _batched_nuts_source_stage(
    plan::BatchedNUTSKernelSourcePlan,
    artifact_stage::BatchedNUTSKernelArtifactStage,
)
    constant_arguments = _batched_nuts_source_arguments(
        plan.target,
        _batched_nuts_artifact_constant_arguments(artifact_stage),
    )
    device_arguments = _batched_nuts_source_arguments(
        plan.target,
        _batched_nuts_artifact_device_arguments(artifact_stage),
    )
    shared_arguments = _batched_nuts_source_arguments(
        plan.target,
        _batched_nuts_artifact_shared_arguments(artifact_stage),
    )
    source_entry = Symbol(
        lowercase(String(plan.source_module)),
        :__,
        _batched_nuts_artifact_symbol(artifact_stage),
        :__,
        :stub,
    )
    stage_kind = _batched_nuts_source_stage_kind(artifact_stage)
    return BatchedNUTSKernelSourceStage(
        artifact_stage,
        plan.backend,
        plan.source_module,
        source_entry,
        constant_arguments,
        device_arguments,
        shared_arguments,
        _batched_nuts_source_lines(
            plan.source_module,
            source_entry,
            stage_kind,
            plan.target,
            constant_arguments,
            device_arguments,
            shared_arguments,
        ),
    )
end

function _batched_nuts_source_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    artifact_plan = _batched_nuts_artifact_plan(program; target=target)
    backend = _batched_nuts_source_backend(target)
    source_module = _batched_nuts_source_module(target)
    plan = BatchedNUTSKernelSourcePlan(
        target,
        artifact_plan,
        backend,
        source_module,
        (),
    )
    stages = Tuple(
        _batched_nuts_source_stage(plan, stage) for
        stage in _batched_nuts_artifact_stages(artifact_plan)
    )
    return BatchedNUTSKernelSourcePlan(
        target,
        artifact_plan,
        backend,
        source_module,
        stages,
    )
end
