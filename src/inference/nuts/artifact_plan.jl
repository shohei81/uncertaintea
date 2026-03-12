@enum BatchedNUTSKernelArtifactBackend::UInt8 begin
    NUTSKernelCPUArtifact = 0
    NUTSKernelMetalArtifact = 1
    NUTSKernelCUDAArtifact = 2
end

struct BatchedNUTSKernelArtifactArgument
    argument::BatchedNUTSKernelCodegenArgument
    slot::Int
end

struct BatchedNUTSKernelArtifactStage{S,C,D,H,B}
    codegen_stage::S
    backend::BatchedNUTSKernelArtifactBackend
    module_symbol::Symbol
    artifact_symbol::Symbol
    constant_arguments::C
    device_arguments::D
    shared_arguments::H
    barriers_after::B
end

struct BatchedNUTSKernelArtifactPlan{P,S}
    target::Symbol
    codegen_plan::P
    backend::BatchedNUTSKernelArtifactBackend
    module_symbol::Symbol
    stages::S
end

_batched_nuts_artifact_codegen_plan(plan::BatchedNUTSKernelArtifactPlan) = plan.codegen_plan
_batched_nuts_artifact_backend(plan::BatchedNUTSKernelArtifactPlan) = plan.backend
_batched_nuts_artifact_module_symbol(plan::BatchedNUTSKernelArtifactPlan) = plan.module_symbol
_batched_nuts_artifact_stages(plan::BatchedNUTSKernelArtifactPlan) = plan.stages

_batched_nuts_artifact_codegen_stage(stage::BatchedNUTSKernelArtifactStage) = stage.codegen_stage
_batched_nuts_artifact_backend(stage::BatchedNUTSKernelArtifactStage) = stage.backend
_batched_nuts_artifact_module_symbol(stage::BatchedNUTSKernelArtifactStage) = stage.module_symbol
_batched_nuts_artifact_symbol(stage::BatchedNUTSKernelArtifactStage) = stage.artifact_symbol
_batched_nuts_artifact_constant_arguments(stage::BatchedNUTSKernelArtifactStage) =
    stage.constant_arguments
_batched_nuts_artifact_device_arguments(stage::BatchedNUTSKernelArtifactStage) =
    stage.device_arguments
_batched_nuts_artifact_shared_arguments(stage::BatchedNUTSKernelArtifactStage) =
    stage.shared_arguments
_batched_nuts_artifact_barriers_after(stage::BatchedNUTSKernelArtifactStage) =
    stage.barriers_after

function _batched_nuts_artifact_argument_symbol(argument::BatchedNUTSKernelArtifactArgument)
    return argument.argument.symbol
end

_batched_nuts_artifact_backend(::Val{:gpu}) = NUTSKernelCPUArtifact
_batched_nuts_artifact_backend(::Val{:metal}) = NUTSKernelMetalArtifact
_batched_nuts_artifact_backend(::Val{:cuda}) = NUTSKernelCUDAArtifact

function _batched_nuts_artifact_backend(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS artifact target $(target)"))
    return _batched_nuts_artifact_backend(Val(target))
end

_batched_nuts_artifact_module_symbol(::Val{:gpu}) = :UncertainTeaCPUArtifacts
_batched_nuts_artifact_module_symbol(::Val{:metal}) = :UncertainTeaMetalArtifacts
_batched_nuts_artifact_module_symbol(::Val{:cuda}) = :UncertainTeaCUDAArtifacts

function _batched_nuts_artifact_module_symbol(target::Symbol)
    target in (:gpu, :metal, :cuda) ||
        throw(ArgumentError("unsupported NUTS artifact target $(target)"))
    return _batched_nuts_artifact_module_symbol(Val(target))
end

function _batched_nuts_artifact_arguments(arguments::Tuple)
    return Tuple(
        BatchedNUTSKernelArtifactArgument(argument, slot) for
        (slot, argument) in enumerate(arguments)
    )
end

function _batched_nuts_artifact_symbol(
    module_symbol::Symbol,
    codegen_stage::BatchedNUTSKernelCodegenStage,
    stage_index::Int,
)
    return Symbol(
        lowercase(String(module_symbol)),
        :__,
        _batched_nuts_codegen_entry_symbol(codegen_stage),
        :__,
        :artifact_,
        stage_index,
    )
end

function _batched_nuts_artifact_stage(
    plan::BatchedNUTSKernelArtifactPlan,
    codegen_stage::BatchedNUTSKernelCodegenStage,
    stage_index::Int,
)
    return BatchedNUTSKernelArtifactStage(
        codegen_stage,
        plan.backend,
        plan.module_symbol,
        _batched_nuts_artifact_symbol(plan.module_symbol, codegen_stage, stage_index),
        _batched_nuts_artifact_arguments(
            _batched_nuts_codegen_constant_arguments(codegen_stage),
        ),
        _batched_nuts_artifact_arguments(
            _batched_nuts_codegen_device_arguments(codegen_stage),
        ),
        _batched_nuts_artifact_arguments(
            _batched_nuts_codegen_shared_arguments(codegen_stage),
        ),
        _batched_nuts_codegen_barriers_after(codegen_stage),
    )
end

function _batched_nuts_artifact_plan(
    program::AbstractBatchedNUTSKernelProgram;
    target::Symbol=:gpu,
)
    codegen_plan = _batched_nuts_codegen_plan(program; target=target)
    backend = _batched_nuts_artifact_backend(target)
    module_symbol = _batched_nuts_artifact_module_symbol(target)
    plan = BatchedNUTSKernelArtifactPlan(
        target,
        codegen_plan,
        backend,
        module_symbol,
        (),
    )
    stages = Tuple(
        _batched_nuts_artifact_stage(plan, stage, stage_index) for
        (stage_index, stage) in enumerate(_batched_nuts_codegen_stages(codegen_plan))
    )
    return BatchedNUTSKernelArtifactPlan(
        target,
        codegen_plan,
        backend,
        module_symbol,
        stages,
    )
end
