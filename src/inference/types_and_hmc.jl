struct HMCMassAdaptationWindowSummary
    window_index::Int
    iteration_start::Int
    iteration_end::Int
    window_length::Int
    pooled_samples::Int
    weight_sum::Float64
    effective_count::Float64
    mean_weight::Float64
    clip_scale_start::Float64
    clip_scale_end::Float64
    updated::Bool
    mass_mean::Float64
    mass_min::Float64
    mass_max::Float64
end

struct HMCMassAdaptationSummary
    window_index::Int
    iteration_start::Int
    iteration_end::Int
    window_length::Int
    chains::Int
    num_updated::Int
    mean_pooled_samples::Float64
    mean_weight_sum::Float64
    mean_effective_count::Float64
    min_effective_count::Float64
    max_effective_count::Float64
    mean_weight::Float64
    mean_clip_scale_end::Float64
    mean_mass::Float64
    min_mass::Float64
    max_mass::Float64
end

struct HMCDiagnosticsSummary
    acceptance_rate::Float64
    divergence_rate::Float64
    step_sizes::Vector{Float64}
    mean_step_size::Float64
    mass_adaptation_windows::Vector{HMCMassAdaptationSummary}
end

struct HMCChain
    sampler::Symbol
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    unconstrained_samples::Matrix{Float64}
    constrained_samples::Matrix{Float64}
    logjoint_values::Vector{Float64}
    acceptance_stats::Vector{Float64}
    energies::Vector{Float64}
    energy_errors::Vector{Float64}
    accepted::BitVector
    divergent::BitVector
    step_size::Float64
    mass_matrix::Vector{Float64}
    num_leapfrog_steps::Int
    max_tree_depth::Int
    tree_depths::Vector{Int}
    integration_steps::Vector{Int}
    target_accept::Float64
    mass_adaptation_windows::Vector{HMCMassAdaptationWindowSummary}
end

struct HMCChains{A,C}
    model::TeaModel
    args::A
    constraints::C
    chains::Vector{HMCChain}
end

struct HMCParameterSummary
    index::Int
    binding::Symbol
    address::Any
    mean::Float64
    sd::Float64
    quantiles::Vector{Float64}
    rhat::Float64
    ess::Float64
end

struct HMCSummary
    model::TeaModel
    space::Symbol
    quantile_probs::Vector{Float64}
    diagnostics::HMCDiagnosticsSummary
    parameters::Vector{HMCParameterSummary}
end

mutable struct DualAveragingState
    target_accept::Float64
    gamma::Float64
    t0::Float64
    kappa::Float64
    mu::Float64
    log_step_size::Float64
    log_step_size_avg::Float64
    hbar::Float64
    iteration::Int
end

mutable struct RunningVarianceState
    mean::Vector{Float64}
    m2::Vector{Float64}
    clipped_sample::Vector{Float64}
    window_length::Int
    count::Int
    weight_sum::Float64
    weight_square_sum::Float64
end

const _RUNNING_VARIANCE_CLIP_START = 4
const _RUNNING_VARIANCE_CLIP_SCALE_EARLY = 8.0
const _RUNNING_VARIANCE_CLIP_SCALE_LATE = 5.0
const _RUNNING_VARIANCE_REJECTION_WEIGHT_EARLY = 1.0
const _RUNNING_VARIANCE_FLOOR = 1e-3

struct WarmupSchedule
    initial_buffer::Int
    slow_window_ends::Vector{Int}
    terminal_buffer::Int
end

mutable struct NUTSState{P<:AbstractVector{Float64}, M<:AbstractVector{Float64}, G<:AbstractVector{Float64}}
    position::P
    momentum::M
    logjoint::Float64
    gradient::G
end

mutable struct NUTSSubtreeMetadataState
    log_weight::Float64
    accept_stat_sum::Float64
    accept_stat_count::Int
    integration_steps::Int
    proposed_energy::Float64
    delta_energy::Float64
    proposal_energy::Float64
    proposal_energy_error::Float64
    accept_prob::Float64
    candidate_log_weight::Float64
    combined_log_weight::Float64
    turning::Bool
    divergent::Bool
end

mutable struct NUTSSubtreeWorkspace{
    C<:NUTSState,
    N<:NUTSState,
    L<:NUTSState,
    R<:NUTSState,
    P<:NUTSState,
    S<:NUTSSubtreeMetadataState,
}
    current::C
    next::N
    left::L
    right::R
    proposal::P
    summary::S
end

mutable struct NUTSContinuationState{L<:NUTSState,R<:NUTSState,P<:NUTSState}
    left::L
    right::R
    proposal::P
    proposal_energy::Float64
    proposal_energy_error::Float64
    log_weight::Float64
    accept_stat_sum::Float64
    accept_stat_count::Int
    integration_steps::Int
    tree_depth::Int
    turning::Bool
    divergent::Bool
end

@enum BatchedNUTSSchedulerPhase::UInt8 begin
    NUTSSchedulerIdle = 0
    NUTSSchedulerExpand = 1
    NUTSSchedulerMerge = 2
    NUTSSchedulerDone = 3
end

mutable struct BatchedNUTSSchedulerState
    continuation_active::BitVector
    subtree_started::BitVector
    active_depth::Int
    active_depth_count::Int
    phase::BatchedNUTSSchedulerPhase
    remaining_steps::Int
end

mutable struct BatchedNUTSControlState
    accepted_step::BitVector
    divergent_step::BitVector
    continuation_turning::BitVector
    step_valid::BitVector
    step_direction::Vector{Int}
    tree_depths::Vector{Int}
    integration_steps::Vector{Int}
    scheduler::BatchedNUTSSchedulerState
end

abstract type AbstractBatchedNUTSControlIR end

struct BatchedNUTSIdleIR <: AbstractBatchedNUTSControlIR end

struct BatchedNUTSExpandIR <: AbstractBatchedNUTSControlIR
    active_depth::Int
    active_depth_count::Int
    remaining_steps::Int
    active_chains::BitVector
    step_direction::Vector{Int}
end

struct BatchedNUTSMergeIR <: AbstractBatchedNUTSControlIR
    active_depth::Int
    active_depth_count::Int
    started_chains::BitVector
    merge_active::BitVector
end

struct BatchedNUTSDoneIR <: AbstractBatchedNUTSControlIR end

abstract type AbstractBatchedNUTSControlBlock end

struct BatchedNUTSIdleControlBlock <: AbstractBatchedNUTSControlBlock
    ir::BatchedNUTSIdleIR
end

struct BatchedNUTSExpandControlBlock <: AbstractBatchedNUTSControlBlock
    ir::BatchedNUTSExpandIR
    active_chains::BitVector
    step_direction::Vector{Int}
end

struct BatchedNUTSMergeControlBlock <: AbstractBatchedNUTSControlBlock
    ir::BatchedNUTSMergeIR
    started_chains::BitVector
    merge_active::BitVector
end

struct BatchedNUTSDoneControlBlock <: AbstractBatchedNUTSControlBlock
    ir::BatchedNUTSDoneIR
end

abstract type AbstractBatchedNUTSStepDescriptor end

struct BatchedNUTSIdleStepDescriptor <: AbstractBatchedNUTSStepDescriptor
    block::BatchedNUTSIdleControlBlock
end

struct BatchedNUTSExpandStepDescriptor <: AbstractBatchedNUTSStepDescriptor
    block::BatchedNUTSExpandControlBlock
    copy_left::BitVector
    copy_right::BitVector
    select_proposal::BitVector
    turning::BitVector
end

struct BatchedNUTSMergeStepDescriptor <: AbstractBatchedNUTSStepDescriptor
    block::BatchedNUTSMergeControlBlock
    select_proposal::BitVector
    merged_turning::BitVector
end

struct BatchedNUTSDoneStepDescriptor <: AbstractBatchedNUTSStepDescriptor
    block::BatchedNUTSDoneControlBlock
end

abstract type AbstractBatchedNUTSStepState end

struct BatchedNUTSIdleStepState <: AbstractBatchedNUTSStepState
    descriptor::BatchedNUTSIdleStepDescriptor
end

struct BatchedNUTSExpandStepState <: AbstractBatchedNUTSStepState
    descriptor::BatchedNUTSExpandStepDescriptor
    log_weight::Vector{Float64}
    proposed_energy::Vector{Float64}
    delta_energy::Vector{Float64}
    proposal_energy::Vector{Float64}
    proposal_energy_error::Vector{Float64}
    accept_prob::Vector{Float64}
    candidate_log_weight::Vector{Float64}
    combined_log_weight::Vector{Float64}
end

struct BatchedNUTSMergeStepState <: AbstractBatchedNUTSStepState
    descriptor::BatchedNUTSMergeStepDescriptor
    proposal_energy::Vector{Float64}
    proposal_energy_error::Vector{Float64}
    candidate_log_weight::Vector{Float64}
    combined_log_weight::Vector{Float64}
end

struct BatchedNUTSDoneStepState <: AbstractBatchedNUTSStepState
    descriptor::BatchedNUTSDoneStepDescriptor
end

abstract type AbstractBatchedNUTSKernelFrame end

struct BatchedNUTSIdleKernelFrame <: AbstractBatchedNUTSKernelFrame
    state::BatchedNUTSIdleStepState
end

struct BatchedNUTSExpandKernelFrame <: AbstractBatchedNUTSKernelFrame
    state::BatchedNUTSExpandStepState
    current_position::Matrix{Float64}
    current_momentum::Matrix{Float64}
    current_gradient::Matrix{Float64}
    current_logjoint::Vector{Float64}
    next_position::Matrix{Float64}
    next_momentum::Matrix{Float64}
    next_gradient::Matrix{Float64}
    proposed_logjoint::Vector{Float64}
    left_position::Matrix{Float64}
    left_momentum::Matrix{Float64}
    left_gradient::Matrix{Float64}
    left_logjoint::Vector{Float64}
    right_position::Matrix{Float64}
    right_momentum::Matrix{Float64}
    right_gradient::Matrix{Float64}
    right_logjoint::Vector{Float64}
    proposal_position::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    proposal_logjoint::Vector{Float64}
    current_energy::Vector{Float64}
end

struct BatchedNUTSMergeKernelFrame <: AbstractBatchedNUTSKernelFrame
    state::BatchedNUTSMergeStepState
    left_position::Matrix{Float64}
    left_momentum::Matrix{Float64}
    left_gradient::Matrix{Float64}
    left_logjoint::Vector{Float64}
    right_position::Matrix{Float64}
    right_momentum::Matrix{Float64}
    right_gradient::Matrix{Float64}
    right_logjoint::Vector{Float64}
    tree_proposal_position::Matrix{Float64}
    tree_proposal_momentum::Matrix{Float64}
    tree_proposal_gradient::Matrix{Float64}
    tree_proposal_logjoint::Vector{Float64}
    proposal_position::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    proposed_logjoint::Vector{Float64}
    continuation_proposal_logjoint::Vector{Float64}
    continuation_log_weight::Vector{Float64}
    current_energy::Vector{Float64}
end

struct BatchedNUTSDoneKernelFrame <: AbstractBatchedNUTSKernelFrame
    state::BatchedNUTSDoneStepState
end

abstract type AbstractBatchedNUTSKernelProgram end

@enum BatchedNUTSKernelOp::UInt8 begin
    NUTSKernelReloadControl = 0
    NUTSKernelLeapfrog = 1
    NUTSKernelHamiltonian = 2
    NUTSKernelAdvance = 3
    NUTSKernelActivateMerge = 4
    NUTSKernelMerge = 5
    NUTSKernelTransitionPhase = 6
end

mutable struct BatchedNUTSKernelExecutionState
    any_active::Bool
end

abstract type AbstractBatchedNUTSKernelStep end

struct BatchedNUTSReloadControlStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSLeapfrogStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSHamiltonianStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSAdvanceStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSActivateMergeStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSMergeStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSTransitionPhaseStep <: AbstractBatchedNUTSKernelStep end

struct BatchedNUTSIdleKernelProgram <: AbstractBatchedNUTSKernelProgram
    frame::BatchedNUTSIdleKernelFrame
    ops::NTuple{1,BatchedNUTSKernelOp}
end

struct BatchedNUTSExpandKernelProgram <: AbstractBatchedNUTSKernelProgram
    frame::BatchedNUTSExpandKernelFrame
    ops::NTuple{5,BatchedNUTSKernelOp}
end

struct BatchedNUTSMergeKernelProgram <: AbstractBatchedNUTSKernelProgram
    frame::BatchedNUTSMergeKernelFrame
    ops::NTuple{4,BatchedNUTSKernelOp}
end

struct BatchedNUTSDoneKernelProgram <: AbstractBatchedNUTSKernelProgram
    frame::BatchedNUTSDoneKernelFrame
    ops::NTuple{1,BatchedNUTSKernelOp}
end

const _BATCHED_NUTS_IDLE_KERNEL_OPS = (NUTSKernelReloadControl,)
const _BATCHED_NUTS_EXPAND_KERNEL_OPS = (
    NUTSKernelReloadControl,
    NUTSKernelLeapfrog,
    NUTSKernelHamiltonian,
    NUTSKernelAdvance,
    NUTSKernelTransitionPhase,
)
const _BATCHED_NUTS_MERGE_KERNEL_OPS = (
    NUTSKernelReloadControl,
    NUTSKernelActivateMerge,
    NUTSKernelMerge,
    NUTSKernelTransitionPhase,
)
const _BATCHED_NUTS_DONE_KERNEL_OPS = (NUTSKernelReloadControl,)
const _BATCHED_NUTS_IDLE_KERNEL_STEPS = (BatchedNUTSReloadControlStep(),)
const _BATCHED_NUTS_EXPAND_KERNEL_STEPS = (
    BatchedNUTSReloadControlStep(),
    BatchedNUTSLeapfrogStep(),
    BatchedNUTSHamiltonianStep(),
    BatchedNUTSAdvanceStep(),
    BatchedNUTSTransitionPhaseStep(),
)
const _BATCHED_NUTS_MERGE_KERNEL_STEPS = (
    BatchedNUTSReloadControlStep(),
    BatchedNUTSActivateMergeStep(),
    BatchedNUTSMergeStep(),
    BatchedNUTSTransitionPhaseStep(),
)
const _BATCHED_NUTS_DONE_KERNEL_STEPS = (BatchedNUTSReloadControlStep(),)

mutable struct BatchedHMCWorkspace
    logjoint_workspace::BatchedLogjointWorkspace
    gradient_cache::BatchedLogjointGradientCache
    current_gradient::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    momentum::Matrix{Float64}
    proposal_position::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    valid::BitVector
    current_hamiltonian::Vector{Float64}
    proposed_hamiltonian::Vector{Float64}
    proposed_logjoint::Vector{Float64}
    log_accept_ratio::Vector{Float64}
    energy_error::Vector{Float64}
    accept_prob::Vector{Float64}
    accepted_step::BitVector
    divergent_step::BitVector
    mass_adaptation_weights::Vector{Float64}
    constrained_position::Matrix{Float64}
    sqrt_inverse_mass_matrix::Vector{Float64}
end

mutable struct BatchedNUTSWorkspace
    gradient_cache::BatchedLogjointGradientCache
    current_gradient::Matrix{Float64}
    tree_current_position::Matrix{Float64}
    tree_next_position::Matrix{Float64}
    tree_left_position::Matrix{Float64}
    tree_right_position::Matrix{Float64}
    tree_proposal_position::Matrix{Float64}
    left_position::Matrix{Float64}
    proposal_position::Matrix{Float64}
    right_position::Matrix{Float64}
    tree_current_momentum::Matrix{Float64}
    tree_next_momentum::Matrix{Float64}
    tree_left_momentum::Matrix{Float64}
    tree_right_momentum::Matrix{Float64}
    tree_proposal_momentum::Matrix{Float64}
    left_momentum::Matrix{Float64}
    current_momentum::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    right_momentum::Matrix{Float64}
    tree_current_gradient::Matrix{Float64}
    tree_next_gradient::Matrix{Float64}
    tree_left_gradient::Matrix{Float64}
    tree_right_gradient::Matrix{Float64}
    tree_proposal_gradient::Matrix{Float64}
    tree_current_logjoint::Vector{Float64}
    tree_left_logjoint::Vector{Float64}
    tree_right_logjoint::Vector{Float64}
    tree_proposal_logjoint::Vector{Float64}
    left_gradient::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    right_gradient::Matrix{Float64}
    left_logjoint::Vector{Float64}
    continuation_proposal_logjoint::Vector{Float64}
    right_logjoint::Vector{Float64}
    proposed_logjoint::Vector{Float64}
    current_energy::Vector{Float64}
    proposed_energy::Vector{Float64}
    continuation_log_weight::Vector{Float64}
    continuation_accept_stat_sum::Vector{Float64}
    continuation_accept_stat_count::Vector{Int}
    continuation_proposed_energy::Vector{Float64}
    continuation_delta_energy::Vector{Float64}
    continuation_accept_prob::Vector{Float64}
    continuation_candidate_log_weight::Vector{Float64}
    continuation_combined_log_weight::Vector{Float64}
    continuation_select_proposal::BitVector
    subtree_log_weight::Vector{Float64}
    subtree_accept_stat_sum::Vector{Float64}
    subtree_accept_stat_count::Vector{Int}
    subtree_integration_steps::Vector{Int}
    subtree_proposed_energy::Vector{Float64}
    subtree_delta_energy::Vector{Float64}
    subtree_proposal_energy::Vector{Float64}
    subtree_proposal_energy_error::Vector{Float64}
    subtree_accept_prob::Vector{Float64}
    subtree_candidate_log_weight::Vector{Float64}
    subtree_combined_log_weight::Vector{Float64}
    energy_error::Vector{Float64}
    accept_prob::Vector{Float64}
    subtree_turning::BitVector
    subtree_merged_turning::BitVector
    subtree_divergent::BitVector
    subtree_active::BitVector
    subtree_copy_left::BitVector
    subtree_copy_right::BitVector
    subtree_select_proposal::BitVector
    control::BatchedNUTSControlState
    mass_adaptation_weights::Vector{Float64}
    constrained_position::Matrix{Float64}
    column_gradient_caches::Vector{LogjointGradientCache}
    column_tree_workspaces::Vector{NUTSSubtreeWorkspace}
    column_continuation_states::Vector{NUTSContinuationState}
end

function BatchedNUTSSchedulerState(num_chains::Int)
    return BatchedNUTSSchedulerState(
        falses(num_chains),
        falses(num_chains),
        0,
        0,
        NUTSSchedulerIdle,
        0,
    )
end

function BatchedNUTSControlState(num_chains::Int)
    return BatchedNUTSControlState(
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        zeros(Int, num_chains),
        zeros(Int, num_chains),
        zeros(Int, num_chains),
        BatchedNUTSSchedulerState(num_chains),
    )
end

function BatchedHMCWorkspace(
    model::TeaModel,
    position::AbstractMatrix,
    args=(),
    constraints=choicemap(),
    inverse_mass_matrix::AbstractVector=ones(size(position, 1)),
)
    num_params, num_chains = size(position)
    batch_args = _validate_batched_args(args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    length(inverse_mass_matrix) == num_params ||
        throw(DimensionMismatch("expected inverse mass matrix of length $num_params, got $(length(inverse_mass_matrix))"))

    return BatchedHMCWorkspace(
        BatchedLogjointWorkspace(model),
        BatchedLogjointGradientCache(model, position, batch_args, batch_constraints),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        falses(num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        falses(num_chains),
        falses(num_chains),
        Vector{Float64}(undef, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        sqrt.(Float64.(inverse_mass_matrix)),
    )
end

function BatchedNUTSWorkspace(
    model::TeaModel,
    position::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    num_params, num_chains = size(position)
    batch_args = _validate_batched_args(args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    gradient_cache = BatchedLogjointGradientCache(model, position, batch_args, batch_constraints)
    tree_current_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_next_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_left_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_right_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_proposal_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_current_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_next_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_left_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_right_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_proposal_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_current_gradient = Matrix{Float64}(undef, num_params, num_chains)
    tree_next_gradient = Matrix{Float64}(undef, num_params, num_chains)
    tree_left_gradient = Matrix{Float64}(undef, num_params, num_chains)
    tree_right_gradient = Matrix{Float64}(undef, num_params, num_chains)
    tree_proposal_gradient = Matrix{Float64}(undef, num_params, num_chains)
    column_gradient_caches = _batched_nuts_column_gradient_caches(
        model,
        position,
        batch_args,
        batch_constraints,
        tree_next_gradient,
    )
    column_tree_workspaces = [
        NUTSSubtreeWorkspace(
            NUTSState(view(tree_current_position, :, chain_index), view(tree_current_momentum, :, chain_index), 0.0, view(tree_current_gradient, :, chain_index)),
            NUTSState(view(tree_next_position, :, chain_index), view(tree_next_momentum, :, chain_index), 0.0, view(tree_next_gradient, :, chain_index)),
            NUTSState(view(tree_left_position, :, chain_index), view(tree_left_momentum, :, chain_index), 0.0, view(tree_left_gradient, :, chain_index)),
            NUTSState(view(tree_right_position, :, chain_index), view(tree_right_momentum, :, chain_index), 0.0, view(tree_right_gradient, :, chain_index)),
            NUTSState(view(tree_proposal_position, :, chain_index), view(tree_proposal_momentum, :, chain_index), 0.0, view(tree_proposal_gradient, :, chain_index)),
            NUTSSubtreeMetadataState(-Inf, 0.0, 0, 0, Inf, Inf, Inf, Inf, 0.0, -Inf, -Inf, false, false),
        ) for chain_index in 1:num_chains
    ]
    left_position = Matrix{Float64}(undef, num_params, num_chains)
    right_position = Matrix{Float64}(undef, num_params, num_chains)
    left_momentum = Matrix{Float64}(undef, num_params, num_chains)
    right_momentum = Matrix{Float64}(undef, num_params, num_chains)
    left_gradient = Matrix{Float64}(undef, num_params, num_chains)
    right_gradient = Matrix{Float64}(undef, num_params, num_chains)
    proposal_position = Matrix{Float64}(undef, num_params, num_chains)
    proposal_momentum = Matrix{Float64}(undef, num_params, num_chains)
    proposal_gradient = Matrix{Float64}(undef, num_params, num_chains)
    proposed_logjoint = Vector{Float64}(undef, num_chains)
    control = BatchedNUTSControlState(num_chains)
    column_continuation_states = [
        NUTSContinuationState(
            NUTSState(view(left_position, :, chain_index), view(left_momentum, :, chain_index), 0.0, view(left_gradient, :, chain_index)),
            NUTSState(view(right_position, :, chain_index), view(right_momentum, :, chain_index), 0.0, view(right_gradient, :, chain_index)),
            _batched_nuts_state(proposal_position, proposal_momentum, proposed_logjoint, proposal_gradient, chain_index),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            0,
            false,
            false,
        ) for chain_index in 1:num_chains
    ]
    return BatchedNUTSWorkspace(
        gradient_cache,
        Matrix{Float64}(undef, num_params, num_chains),
        tree_current_position,
        tree_next_position,
        tree_left_position,
        tree_right_position,
        tree_proposal_position,
        left_position,
        proposal_position,
        right_position,
        tree_current_momentum,
        tree_next_momentum,
        tree_left_momentum,
        tree_right_momentum,
        tree_proposal_momentum,
        left_momentum,
        Matrix{Float64}(undef, num_params, num_chains),
        proposal_momentum,
        right_momentum,
        tree_current_gradient,
        tree_next_gradient,
        tree_left_gradient,
        tree_right_gradient,
        tree_proposal_gradient,
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        left_gradient,
        proposal_gradient,
        right_gradient,
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        proposed_logjoint,
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        zeros(Int, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        falses(num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        zeros(Int, num_chains),
        zeros(Int, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        control,
        Vector{Float64}(undef, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        column_gradient_caches,
        column_tree_workspaces,
        column_continuation_states,
    )
end

function _batched_nuts_column_gradient_caches(
    model::TeaModel,
    position::AbstractMatrix,
    batch_args::Tuple,
    batch_constraints::ChoiceMap,
    gradient_buffer::AbstractMatrix,
)
    num_chains = size(position, 2)
    num_chains == 0 && return LogjointGradientCache[]
    first_cache = _logjoint_gradient_cache(
        model,
        collect(view(position, :, 1)),
        batch_args,
        batch_constraints,
        view(gradient_buffer, :, 1),
    )
    caches = Vector{typeof(first_cache)}(undef, num_chains)
    caches[1] = first_cache
    for chain_index in 2:num_chains
        caches[chain_index] = LogjointGradientCache(
            first_cache.objective,
            first_cache.config,
            view(gradient_buffer, :, chain_index),
        )
    end
    return caches
end

function _batched_nuts_column_gradient_caches(
    model::TeaModel,
    position::AbstractMatrix,
    batch_args,
    batch_constraints,
    gradient_buffer::AbstractMatrix,
)
    num_chains = size(position, 2)
    caches = Vector{LogjointGradientCache}(undef, num_chains)
    for chain_index in 1:num_chains
        caches[chain_index] = _logjoint_gradient_cache(
            model,
            collect(view(position, :, chain_index)),
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            view(gradient_buffer, :, chain_index),
        )
    end
    return caches
end

Base.length(chain::HMCChain) = size(chain.unconstrained_samples, 2)
Base.length(chains::HMCChains) = length(chains.chains)
Base.length(summary::HMCSummary) = length(summary.parameters)
Base.getindex(chains::HMCChains, index::Int) = chains.chains[index]
Base.getindex(summary::HMCSummary, index::Int) = summary.parameters[index]
Base.firstindex(chains::HMCChains) = firstindex(chains.chains)
Base.firstindex(summary::HMCSummary) = firstindex(summary.parameters)
Base.lastindex(chains::HMCChains) = lastindex(chains.chains)
Base.lastindex(summary::HMCSummary) = lastindex(summary.parameters)
Base.iterate(chains::HMCChains, state...) = iterate(chains.chains, state...)
Base.iterate(summary::HMCSummary, state...) = iterate(summary.parameters, state...)

_sampler_label(chain::HMCChain) = uppercase(String(chain.sampler))

function _sampler_label(chains::HMCChains)
    isempty(chains.chains) && return "HMC"
    sampler = first(chains.chains).sampler
    return all(chain -> chain.sampler === sampler, chains.chains) ? uppercase(String(sampler)) : "MIXED"
end

function _summary_float(value::Real; digits::Int=4)
    if isnan(value)
        return "NaN"
    elseif !isfinite(value)
        return value > 0 ? "Inf" : "-Inf"
    end
    return string(round(Float64(value); digits=digits))
end

function _show_mass_adaptation_summary_line(io::IO, summary::HMCMassAdaptationSummary; indent::AbstractString="")
    print(
        io,
        indent,
        "window ",
        summary.window_index,
        " [",
        summary.iteration_start,
        ":",
        summary.iteration_end,
        "]",
        " updated=",
        summary.num_updated,
        "/",
        summary.chains,
        " eff=",
        _summary_float(summary.mean_effective_count; digits=2),
        " mass=",
        _summary_float(summary.mean_mass),
        " clip=",
        _summary_float(summary.mean_clip_scale_end; digits=2),
    )
end

function Base.show(io::IO, summary::HMCMassAdaptationWindowSummary)
    print(
        io,
        "HMCMassAdaptationWindowSummary(window=",
        summary.window_index,
        ", iterations=",
        summary.iteration_start,
        ":",
        summary.iteration_end,
        ", effective_count=",
        round(summary.effective_count; digits=2),
        ", updated=",
        summary.updated,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", summary::HMCMassAdaptationWindowSummary)
    println(io, "HMCMassAdaptationWindowSummary")
    println(io, "  window: ", summary.window_index)
    println(io, "  iterations: ", summary.iteration_start, ":", summary.iteration_end)
    println(io, "  window_length: ", summary.window_length)
    println(io, "  pooled_samples: ", summary.pooled_samples)
    println(io, "  effective_count: ", _summary_float(summary.effective_count; digits=2))
    println(io, "  mean_weight: ", _summary_float(summary.mean_weight; digits=3))
    println(io, "  clip_scale: ", _summary_float(summary.clip_scale_start; digits=2), " -> ", _summary_float(summary.clip_scale_end; digits=2))
    println(io, "  updated: ", summary.updated)
    print(io, "  mass: mean=", _summary_float(summary.mass_mean), " min=", _summary_float(summary.mass_min), " max=", _summary_float(summary.mass_max))
end

function Base.show(io::IO, summary::HMCMassAdaptationSummary)
    print(
        io,
        "HMCMassAdaptationSummary(window=",
        summary.window_index,
        ", chains=",
        summary.chains,
        ", effective_count=",
        round(summary.mean_effective_count; digits=2),
        ", updated=",
        summary.num_updated,
        "/",
        summary.chains,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", summary::HMCMassAdaptationSummary)
    println(io, "HMCMassAdaptationSummary")
    println(io, "  window: ", summary.window_index, " (", summary.iteration_start, ":", summary.iteration_end, ")")
    println(io, "  chains: ", summary.chains, " updated=", summary.num_updated, "/", summary.chains)
    println(io, "  window_length: ", summary.window_length)
    println(io, "  pooled_samples_mean: ", _summary_float(summary.mean_pooled_samples; digits=2))
    println(io, "  weight_sum_mean: ", _summary_float(summary.mean_weight_sum; digits=2))
    println(io, "  effective_count: mean=", _summary_float(summary.mean_effective_count; digits=2),
        " min=", _summary_float(summary.min_effective_count; digits=2),
        " max=", _summary_float(summary.max_effective_count; digits=2))
    println(io, "  mean_weight: ", _summary_float(summary.mean_weight; digits=3))
    println(io, "  clip_scale_end_mean: ", _summary_float(summary.mean_clip_scale_end; digits=2))
    print(io, "  mass: mean=", _summary_float(summary.mean_mass), " min=", _summary_float(summary.min_mass), " max=", _summary_float(summary.max_mass))
end

function Base.show(io::IO, diagnostics::HMCDiagnosticsSummary)
    print(
        io,
        "HMCDiagnosticsSummary(acceptance_rate=",
        round(diagnostics.acceptance_rate; digits=3),
        ", divergence_rate=",
        round(diagnostics.divergence_rate; digits=3),
        ", mass_windows=",
        length(diagnostics.mass_adaptation_windows),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", diagnostics::HMCDiagnosticsSummary)
    println(io, "HMCDiagnosticsSummary")
    println(io, "  acceptance_rate: ", _summary_float(diagnostics.acceptance_rate; digits=3))
    println(io, "  divergence_rate: ", _summary_float(diagnostics.divergence_rate; digits=3))
    println(
        io,
        "  step_size: mean=",
        _summary_float(diagnostics.mean_step_size),
        " min=",
        _summary_float(isempty(diagnostics.step_sizes) ? 0.0 : minimum(diagnostics.step_sizes)),
        " max=",
        _summary_float(isempty(diagnostics.step_sizes) ? 0.0 : maximum(diagnostics.step_sizes)),
    )
    if isempty(diagnostics.mass_adaptation_windows)
        print(io, "  mass_adaptation_windows: none")
        return nothing
    end
    println(io, "  mass_adaptation_windows:")
    for summary in diagnostics.mass_adaptation_windows
        _show_mass_adaptation_summary_line(io, summary; indent="    ")
        println(io)
    end
    return nothing
end

function Base.show(io::IO, chain::HMCChain)
    print(io, "HMCChain(", lowercase(String(chain.sampler)), ", ", chain.model.name)
    print(io, ", samples=", length(chain))
    print(io, ", acceptance_rate=", round(acceptancerate(chain); digits=3))
    print(io, ", divergences=", count(identity, chain.divergent))
    print(io, ", step_size=", round(chain.step_size; digits=4))
    if chain.sampler === :nuts
        print(io, ", max_tree_depth=", chain.max_tree_depth)
    else
        print(io, ", num_leapfrog_steps=", chain.num_leapfrog_steps)
    end
    print(io, ", mass_windows=", length(chain.mass_adaptation_windows), ")")
end

function Base.show(io::IO, chains::HMCChains)
    print(
        io,
        "HMCChains(",
        lowercase(_sampler_label(chains)),
        ", ",
        chains.model.name,
        ", chains=",
        length(chains),
        ", samples=",
        numsamples(chains),
        ", acceptance_rate=",
        round(acceptancerate(chains); digits=3),
        ", divergences=",
        sum(count(identity, chain.divergent) for chain in chains.chains),
        ")",
    )
end

function Base.show(io::IO, summary::HMCSummary)
    print(
        io,
        "HMCSummary(",
        summary.model.name,
        ", space=",
        summary.space,
        ", parameters=",
        length(summary),
        ", acceptance_rate=",
        round(summary.diagnostics.acceptance_rate; digits=3),
        ", divergence_rate=",
        round(summary.diagnostics.divergence_rate; digits=3),
        ", mass_windows=",
        length(summary.diagnostics.mass_adaptation_windows),
        ", quantiles=",
        summary.quantile_probs,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", summary::HMCSummary)
    println(io, "HMCSummary(", summary.model.name, ")")
    println(io, "  space: ", summary.space)
    println(io, "  quantiles: ", summary.quantile_probs)
    println(io, "  parameters: ", length(summary))
    max_parameters = min(length(summary.parameters), 5)
    for parameter_index in 1:max_parameters
        parameter = summary.parameters[parameter_index]
        mid_quantile = parameter.quantiles[cld(length(parameter.quantiles), 2)]
        println(
            io,
            "    ",
            parameter.binding,
            " @ ",
            parameter.address,
            ": mean=",
            _summary_float(parameter.mean),
            " sd=",
            _summary_float(parameter.sd),
            " median=",
            _summary_float(mid_quantile),
            " rhat=",
            _summary_float(parameter.rhat; digits=3),
            " ess=",
            _summary_float(parameter.ess; digits=1),
        )
    end
    if length(summary.parameters) > max_parameters
        println(io, "    ... ", length(summary.parameters) - max_parameters, " more parameters")
    end
    println(io, "  diagnostics:")
    print(io, "    acceptance_rate: ", _summary_float(summary.diagnostics.acceptance_rate; digits=3))
    println(io)
    print(io, "    divergence_rate: ", _summary_float(summary.diagnostics.divergence_rate; digits=3))
    println(io)
    print(
        io,
        "    step_size: mean=",
        _summary_float(summary.diagnostics.mean_step_size),
        " min=",
        _summary_float(isempty(summary.diagnostics.step_sizes) ? 0.0 : minimum(summary.diagnostics.step_sizes)),
        " max=",
        _summary_float(isempty(summary.diagnostics.step_sizes) ? 0.0 : maximum(summary.diagnostics.step_sizes)),
    )
    println(io)
    if isempty(summary.diagnostics.mass_adaptation_windows)
        print(io, "    mass_adaptation_windows: none")
        return nothing
    end
    println(io, "    mass_adaptation_windows:")
    for window_summary in summary.diagnostics.mass_adaptation_windows
        _show_mass_adaptation_summary_line(io, window_summary; indent="      ")
        println(io)
    end
    return nothing
end

function acceptancerate(chain::HMCChain)
    isempty(chain.acceptance_stats) && return 0.0
    return _sample_mean(chain.acceptance_stats)
end

function acceptancerate(chains::HMCChains)
    total_samples = sum(length(chain.acceptance_stats) for chain in chains.chains)
    total_samples == 0 && return 0.0
    return sum(sum(chain.acceptance_stats) for chain in chains.chains) / total_samples
end

function divergencerate(chain::HMCChain)
    isempty(chain.divergent) && return 0.0
    return count(identity, chain.divergent) / length(chain.divergent)
end

function divergencerate(chains::HMCChains)
    total_samples = sum(length(chain.divergent) for chain in chains.chains)
    total_samples == 0 && return 0.0
    return sum(count(identity, chain.divergent) for chain in chains.chains) / total_samples
end

function acceptancerate(summary::HMCSummary)
    return summary.diagnostics.acceptance_rate
end

function divergencerate(summary::HMCSummary)
    return summary.diagnostics.divergence_rate
end

function massadaptationwindows(chain::HMCChain)
    return chain.mass_adaptation_windows
end

function massadaptationwindows(chains::HMCChains)
    return [chain.mass_adaptation_windows for chain in chains.chains]
end

function massadaptationwindows(summary::HMCSummary)
    return summary.diagnostics.mass_adaptation_windows
end

function treedepths(chain::HMCChain)
    return chain.tree_depths
end

function treedepths(chains::HMCChains)
    return [chain.tree_depths for chain in chains.chains]
end

function integrationsteps(chain::HMCChain)
    return chain.integration_steps
end

function integrationsteps(chains::HMCChains)
    return [chain.integration_steps for chain in chains.chains]
end

function _diagnostic_space_samples(chain::HMCChain, space::Symbol)
    if space === :constrained
        return chain.constrained_samples
    elseif space === :unconstrained
        return chain.unconstrained_samples
    end

    throw(ArgumentError("diagnostic space must be :constrained or :unconstrained"))
end

function nchains(chains::HMCChains)
    return length(chains)
end

function numsamples(chains::HMCChains)
    isempty(chains.chains) && return 0
    return length(first(chains.chains))
end

function _summary_address(address::AddressSpec)
    parts = Any[part isa AddressLiteralPart ? part.value : part.value for part in address.parts]
    return length(parts) == 1 ? first(parts) : Tuple(parts)
end

function _validate_hmc_diagnostics(chains::HMCChains, space::Symbol)
    length(chains) >= 2 || throw(ArgumentError("multi-chain diagnostics require at least 2 chains"))
    num_samples = numsamples(chains)
    num_samples >= 4 || throw(ArgumentError("multi-chain diagnostics require at least 4 samples per chain"))

    first_samples = _diagnostic_space_samples(first(chains.chains), space)
    num_params = size(first_samples, 1)
    for chain in chains.chains
        chain_samples = _diagnostic_space_samples(chain, space)
        size(chain_samples, 1) == num_params ||
            throw(DimensionMismatch("all chains must have the same parameter dimension"))
        size(chain_samples, 2) == num_samples ||
            throw(DimensionMismatch("all chains must have the same number of samples"))
    end

    return num_params, num_samples
end

function _validate_summary_quantiles(quantile_probs)
    isempty(quantile_probs) && throw(ArgumentError("summary quantiles must be non-empty"))
    probabilities = Float64[Float64(prob) for prob in quantile_probs]
    for prob in probabilities
        0.0 <= prob <= 1.0 || throw(ArgumentError("summary quantiles must lie in [0, 1]"))
    end
    return probabilities
end

function _mass_adaptation_diagnostics(chains::HMCChains)
    groups = Dict{NTuple{4, Int}, Vector{HMCMassAdaptationWindowSummary}}()
    for chain in chains.chains
        for window in chain.mass_adaptation_windows
            key = (window.window_index, window.iteration_start, window.iteration_end, window.window_length)
            push!(get!(groups, key, HMCMassAdaptationWindowSummary[]), window)
        end
    end

    summaries = HMCMassAdaptationSummary[]
    for key in sort!(collect(keys(groups)); by=identity)
        windows = groups[key]
        push!(
            summaries,
            HMCMassAdaptationSummary(
                key[1],
                key[2],
                key[3],
                key[4],
                length(windows),
                count(window -> window.updated, windows),
                _sample_mean([window.pooled_samples for window in windows]),
                _sample_mean([window.weight_sum for window in windows]),
                _sample_mean([window.effective_count for window in windows]),
                minimum(window.effective_count for window in windows),
                maximum(window.effective_count for window in windows),
                _sample_mean([window.mean_weight for window in windows]),
                _sample_mean([window.clip_scale_end for window in windows]),
                _sample_mean([window.mass_mean for window in windows]),
                minimum(window.mass_min for window in windows),
                maximum(window.mass_max for window in windows),
            ),
        )
    end
    return summaries
end

function _diagnostics_summary(chains::HMCChains)
    step_sizes = Float64[chain.step_size for chain in chains.chains]
    return HMCDiagnosticsSummary(
        acceptancerate(chains),
        divergencerate(chains),
        step_sizes,
        isempty(step_sizes) ? 0.0 : _sample_mean(step_sizes),
        _mass_adaptation_diagnostics(chains),
    )
end

function _sample_mean(values::AbstractVector)
    return sum(values) / length(values)
end

function _sample_variance(values::AbstractVector, mean_value::Real=_sample_mean(values))
    length(values) > 1 || return 0.0
    return sum((value - mean_value)^2 for value in values) / (length(values) - 1)
end

function _sample_sd(values::AbstractVector, mean_value::Real=_sample_mean(values))
    return sqrt(max(_sample_variance(values, mean_value), 0.0))
end

function _pooled_parameter_draws(chains::HMCChains, parameter_index::Int, space::Symbol)
    _, num_samples = _validate_hmc_diagnostics(chains, space)
    pooled = Vector{Float64}(undef, length(chains) * num_samples)
    offset = 1
    for chain in chains.chains
        samples = _diagnostic_space_samples(chain, space)
        pooled[offset:(offset + num_samples - 1)] = samples[parameter_index, :]
        offset += num_samples
    end
    return pooled
end

function _quantile(sorted_values::AbstractVector, probability::Float64)
    num_values = length(sorted_values)
    num_values == 0 && throw(ArgumentError("quantile requires at least one value"))
    num_values == 1 && return Float64(sorted_values[1])

    position = 1 + (num_values - 1) * probability
    lower = floor(Int, position)
    upper = ceil(Int, position)
    lower == upper && return Float64(sorted_values[lower])
    weight = position - lower
    return (1 - weight) * sorted_values[lower] + weight * sorted_values[upper]
end

function _quantiles(values::AbstractVector, probabilities::AbstractVector{Float64})
    sorted_values = sort(collect(values))
    return Float64[_quantile(sorted_values, probability) for probability in probabilities]
end

function _split_chain_parameter_draws(chains::HMCChains, parameter_index::Int, space::Symbol)
    _, num_samples = _validate_hmc_diagnostics(chains, space)
    split_samples = fld(num_samples, 2)
    even_samples = 2 * split_samples
    split_draws = Matrix{Float64}(undef, 2 * length(chains), split_samples)

    for (chain_index, chain) in enumerate(chains.chains)
        samples = _diagnostic_space_samples(chain, space)
        split_draws[2 * chain_index - 1, :] = samples[parameter_index, 1:split_samples]
        split_draws[2 * chain_index, :] = samples[parameter_index, split_samples + 1:even_samples]
    end

    return split_draws
end

function _chain_draw_statistics(draws::AbstractMatrix)
    num_chains, num_samples = size(draws)
    chain_means = Vector{Float64}(undef, num_chains)
    chain_variances = Vector{Float64}(undef, num_chains)
    for chain_index in 1:num_chains
        chain_draws = view(draws, chain_index, :)
        chain_means[chain_index] = _sample_mean(chain_draws)
        chain_variances[chain_index] = _sample_variance(chain_draws, chain_means[chain_index])
    end

    within_variance = _sample_mean(chain_variances)
    between_variance = num_samples > 1 ? num_samples * _sample_variance(chain_means) : 0.0
    var_plus = ((num_samples - 1) / num_samples) * within_variance + between_variance / num_samples
    return chain_means, chain_variances, within_variance, between_variance, var_plus
end

function _split_rhat(draws::AbstractMatrix)
    _, _, within_variance, _, var_plus = _chain_draw_statistics(draws)
    if within_variance == 0
        return var_plus == 0 ? 1.0 : Inf
    end

    return sqrt(max(var_plus / within_variance, 1.0))
end

function _autocovariance(draws::AbstractVector, lag::Int, mean_value::Real)
    num_samples = length(draws)
    total = 0.0
    for index in 1:(num_samples - lag)
        total += (draws[index] - mean_value) * (draws[index + lag] - mean_value)
    end
    return total / num_samples
end

function _split_ess(draws::AbstractMatrix)
    num_chains, num_samples = size(draws)
    chain_means, _, within_variance, _, var_plus = _chain_draw_statistics(draws)
    total_draws = num_chains * num_samples

    if within_variance == 0 && var_plus == 0
        return Float64(total_draws)
    elseif var_plus <= 0
        return 0.0
    end

    pair_sums = Float64[]
    autocovariance_means = Vector{Float64}(undef, num_chains)
    for pair_start in 0:2:(num_samples - 1)
        pair_sum = 0.0
        for lag in pair_start:min(pair_start + 1, num_samples - 1)
            for chain_index in 1:num_chains
                autocovariance_means[chain_index] = _autocovariance(view(draws, chain_index, :), lag, chain_means[chain_index])
            end
            mean_autocovariance = _sample_mean(autocovariance_means)
            rho_hat = lag == 0 ? 1.0 : 1 - (within_variance - mean_autocovariance) / var_plus
            pair_sum += min(rho_hat, 1.0)
        end

        pair_sum > 0 || break
        push!(pair_sums, pair_sum)
    end

    for index in 2:length(pair_sums)
        pair_sums[index] = min(pair_sums[index], pair_sums[index - 1])
    end

    tau_hat = -1 + 2 * sum(pair_sums)
    tau_hat = max(tau_hat, 1.0)
    return min(Float64(total_draws), Float64(total_draws) / tau_hat)
end

function rhat(chains::HMCChains; space::Symbol=:constrained)
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    values = Vector{Float64}(undef, num_params)
    for parameter_index in 1:num_params
        values[parameter_index] = _split_rhat(_split_chain_parameter_draws(chains, parameter_index, space))
    end
    return values
end

function ess(chains::HMCChains; space::Symbol=:constrained)
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    values = Vector{Float64}(undef, num_params)
    for parameter_index in 1:num_params
        values[parameter_index] = _split_ess(_split_chain_parameter_draws(chains, parameter_index, space))
    end
    return values
end

function summarize(chains::HMCChains; space::Symbol=:constrained, quantiles=(0.05, 0.5, 0.95))
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    quantile_probs = _validate_summary_quantiles(quantiles)
    rhats = rhat(chains; space=space)
    ess_values = ess(chains; space=space)
    diagnostics = _diagnostics_summary(chains)
    layout = parameterlayout(chains.model)
    parametercount(layout) == num_params ||
        throw(DimensionMismatch("summary expected $num_params parameters in layout, got $(parametercount(layout))"))

    parameters = Vector{HMCParameterSummary}(undef, num_params)
    for slot in layout.slots
        draws = _pooled_parameter_draws(chains, slot.index, space)
        mean_value = _sample_mean(draws)
        parameters[slot.index] = HMCParameterSummary(
            slot.index,
            slot.binding,
            _summary_address(slot.address),
            mean_value,
            _sample_sd(draws, mean_value),
            _quantiles(draws, quantile_probs),
            rhats[slot.index],
            ess_values[slot.index],
        )
    end

    return HMCSummary(chains.model, space, quantile_probs, diagnostics, parameters)
end

function _validate_hmc_arguments(
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
)
    num_params > 0 || throw(ArgumentError("HMC requires at least one parameterized latent choice"))
    num_samples > 0 || throw(ArgumentError("HMC requires num_samples > 0"))
    num_warmup >= 0 || throw(ArgumentError("HMC requires num_warmup >= 0"))
    step_size > 0 || throw(ArgumentError("HMC requires step_size > 0"))
    num_leapfrog_steps > 0 || throw(ArgumentError("HMC requires num_leapfrog_steps > 0"))
    0 < target_accept < 1 || throw(ArgumentError("HMC requires 0 < target_accept < 1"))
    divergence_threshold > 0 || throw(ArgumentError("HMC requires divergence_threshold > 0"))
    mass_matrix_regularization > 0 || throw(ArgumentError("HMC requires mass_matrix_regularization > 0"))
    mass_matrix_min_samples > 0 || throw(ArgumentError("HMC requires mass_matrix_min_samples > 0"))
    return nothing
end

function _validate_nuts_arguments(
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    max_tree_depth::Int,
    target_accept::Real,
    max_delta_energy::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
)
    num_params > 0 || throw(ArgumentError("NUTS requires at least one parameterized latent choice"))
    num_samples > 0 || throw(ArgumentError("NUTS requires num_samples > 0"))
    num_warmup >= 0 || throw(ArgumentError("NUTS requires num_warmup >= 0"))
    step_size > 0 || throw(ArgumentError("NUTS requires step_size > 0"))
    max_tree_depth > 0 || throw(ArgumentError("NUTS requires max_tree_depth > 0"))
    0 < target_accept < 1 || throw(ArgumentError("NUTS requires 0 < target_accept < 1"))
    max_delta_energy > 0 || throw(ArgumentError("NUTS requires max_delta_energy > 0"))
    mass_matrix_regularization > 0 || throw(ArgumentError("NUTS requires mass_matrix_regularization > 0"))
    mass_matrix_min_samples > 0 || throw(ArgumentError("NUTS requires mass_matrix_min_samples > 0"))
    return nothing
end

function _validate_hmc_chains_arguments(num_chains::Int, sampler_name::AbstractString="HMC")
    num_chains > 0 || throw(ArgumentError("$sampler_name requires num_chains > 0"))
    return nothing
end

function _validate_batched_hmc_arguments(
    num_chains::Int,
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    args,
    constraints,
)
    _validate_hmc_chains_arguments(num_chains)
    _validate_hmc_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )
    _validate_batched_args(args, num_chains)
    _validate_batched_constraints(constraints, num_chains)
    return nothing
end

function _validate_batched_nuts_arguments(
    num_chains::Int,
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    max_tree_depth::Int,
    target_accept::Real,
    max_delta_energy::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    args,
    constraints,
)
    _validate_hmc_chains_arguments(num_chains, "NUTS")
    _validate_nuts_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        max_tree_depth,
        target_accept,
        max_delta_energy,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )
    _validate_batched_args(args, num_chains)
    _validate_batched_constraints(constraints, num_chains)
    return nothing
end

function _initial_hmc_position(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    initial_params,
    rng::AbstractRNG,
)
    if isnothing(initial_params)
        trace, _ = generate(model, args, constraints; rng=rng)
        return transform_to_unconstrained(trace)
    end

    expected = parametercount(parameterlayout(model))
    length(initial_params) == expected || throw(DimensionMismatch("expected $expected initial parameters, got $(length(initial_params))"))
    return Float64[value for value in initial_params]
end

function _sample_momentum(rng::AbstractRNG, inverse_mass_matrix::AbstractVector)
    return randn(rng, length(inverse_mass_matrix)) ./ sqrt.(inverse_mass_matrix)
end

function _initial_batched_hmc_positions(
    model::TeaModel,
    args,
    constraints,
    initial_params,
    rng::AbstractRNG,
    num_params::Int,
    num_chains::Int,
)
    batch_args = _validate_batched_args(args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    positions = Matrix{Float64}(undef, num_params, num_chains)
    seeds = rand(rng, UInt, num_chains)

    for chain_index in 1:num_chains
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, num_chains)
        chain_rng = MersenneTwister(seeds[chain_index])
        positions[:, chain_index] = _initial_hmc_position(
            model,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            chain_initial_params,
            chain_rng,
        )
    end

    return positions
end

function _sample_batched_momentum(
    rng::AbstractRNG,
    inverse_mass_matrix::AbstractVector,
    num_chains::Int,
)
    momentum = Matrix{Float64}(undef, length(inverse_mass_matrix), num_chains)
    _sample_batched_momentum!(momentum, rng, sqrt.(Float64.(inverse_mass_matrix)))
    return momentum
end

function _sample_batched_momentum!(
    destination::AbstractMatrix,
    rng::AbstractRNG,
    sqrt_inverse_mass_matrix::AbstractVector,
)
    size(destination, 1) == length(sqrt_inverse_mass_matrix) ||
        throw(DimensionMismatch("expected momentum matrix with $(length(sqrt_inverse_mass_matrix)) rows, got $(size(destination, 1))"))

    for chain_index in axes(destination, 2)
        for parameter_index in eachindex(sqrt_inverse_mass_matrix)
            destination[parameter_index, chain_index] =
                randn(rng) / sqrt_inverse_mass_matrix[parameter_index]
        end
    end
    return destination
end

function _update_sqrt_inverse_mass_matrix!(
    destination::AbstractVector,
    inverse_mass_matrix::AbstractVector,
)
    length(destination) == length(inverse_mass_matrix) ||
        throw(DimensionMismatch("expected inverse mass matrix of length $(length(destination)), got $(length(inverse_mass_matrix))"))
    for index in eachindex(destination, inverse_mass_matrix)
        destination[index] = sqrt(Float64(inverse_mass_matrix[index]))
    end
    return destination
end

function _batched_kinetic_energy(
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    energy = Vector{Float64}(undef, size(momentum, 2))
    return _batched_kinetic_energy!(energy, momentum, inverse_mass_matrix)
end

function _batched_kinetic_energy!(
    destination::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    size(momentum, 1) == length(inverse_mass_matrix) ||
        throw(DimensionMismatch("expected momentum matrix with $(length(inverse_mass_matrix)) rows, got $(size(momentum, 1))"))
    size(momentum, 2) == length(destination) ||
        throw(DimensionMismatch("expected kinetic-energy destination of length $(size(momentum, 2)), got $(length(destination))"))

    for chain_index in axes(momentum, 2)
        kinetic_energy = 0.0
        for parameter_index in eachindex(inverse_mass_matrix)
            momentum_value = momentum[parameter_index, chain_index]
            kinetic_energy += momentum_value^2 * inverse_mass_matrix[parameter_index]
        end
        destination[chain_index] = kinetic_energy / 2
    end
    return destination
end

function _batched_hamiltonian(
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    hamiltonian = Vector{Float64}(undef, length(logjoint_values))
    return _batched_hamiltonian!(hamiltonian, logjoint_values, momentum, inverse_mass_matrix)
end

function _batched_hamiltonian!(
    destination::AbstractVector,
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    length(logjoint_values) == length(destination) ||
        throw(DimensionMismatch("expected hamiltonian inputs of length $(length(destination)), got $(length(logjoint_values))"))

    _batched_kinetic_energy!(destination, momentum, inverse_mass_matrix)
    for chain_index in eachindex(destination)
        destination[chain_index] -= Float64(logjoint_values[chain_index])
    end
    return destination
end

function _batched_acceptance_probability(log_accept_ratio::AbstractVector)
    probabilities = Vector{Float64}(undef, length(log_accept_ratio))
    return _batched_acceptance_probability!(probabilities, log_accept_ratio)
end

function _batched_acceptance_probability!(
    destination::AbstractVector,
    log_accept_ratio::AbstractVector,
)
    length(destination) == length(log_accept_ratio) ||
        throw(DimensionMismatch("expected acceptance-probability destination of length $(length(log_accept_ratio)), got $(length(destination))"))

    for index in eachindex(log_accept_ratio)
        destination[index] = _acceptance_probability(log_accept_ratio[index])
    end
    return destination
end

function _mean_acceptance_stats!(
    destination::AbstractVector,
    accept_sum::AbstractVector,
    accept_count::AbstractVector{Int},
)
    length(destination) == length(accept_sum) == length(accept_count) ||
        throw(DimensionMismatch("expected acceptance-stat inputs of matching length, got $(length(destination)), $(length(accept_sum)), and $(length(accept_count))"))
    for index in eachindex(destination)
        destination[index] = accept_count[index] == 0 ? 0.0 : accept_sum[index] / accept_count[index]
    end
    return destination
end

function _mean_acceptance_stat(
    accept_sum::Real,
    accept_count::Integer,
)
    return accept_count == 0 ? 0.0 : Float64(accept_sum) / accept_count
end

function _energy_errors!(
    destination::AbstractVector,
    proposed_energy::AbstractVector,
    current_energy::AbstractVector,
)
    length(destination) == length(proposed_energy) == length(current_energy) ||
        throw(DimensionMismatch("expected energy-error inputs of matching length, got $(length(destination)), $(length(proposed_energy)), and $(length(current_energy))"))
    for index in eachindex(destination)
        destination[index] = proposed_energy[index] - current_energy[index]
    end
    return destination
end

function _position_moved(
    proposal_position::AbstractVector,
    current_position::AbstractVector,
)
    length(proposal_position) == length(current_position) ||
        throw(DimensionMismatch("expected moved-position inputs of matching length, got $(length(proposal_position)) and $(length(current_position))"))
    for index in eachindex(proposal_position, current_position)
        proposal_position[index] == current_position[index] || return true
    end
    return false
end

function _nuts_proposal_summary(
    continuation::NUTSContinuationState,
    current_position::AbstractVector,
)
    proposed_energy = continuation.proposal_energy
    energy_error = continuation.proposal_energy_error
    accept_stat = _mean_acceptance_stat(
        continuation.accept_stat_sum,
        continuation.accept_stat_count,
    )
    moved = _position_moved(continuation.proposal.position, current_position)
    return accept_stat, proposed_energy, energy_error, moved
end

function _batched_positions_moved!(
    destination::AbstractVector{Bool},
    proposal_position::AbstractMatrix,
    current_position::AbstractMatrix,
)
    size(proposal_position) == size(current_position) ||
        throw(DimensionMismatch("expected moved-position inputs of matching size, got $(size(proposal_position)) and $(size(current_position))"))
    length(destination) == size(proposal_position, 2) ||
        throw(DimensionMismatch("expected moved-position destination of length $(size(proposal_position, 2)), got $(length(destination))"))
    for chain_index in eachindex(destination)
        moved = false
        for parameter_index in axes(proposal_position, 1)
            if proposal_position[parameter_index, chain_index] != current_position[parameter_index, chain_index]
                moved = true
                break
            end
        end
        destination[chain_index] = moved
    end
    return destination
end

function _finalize_batched_nuts_proposals!(
    workspace::BatchedNUTSWorkspace,
    position::AbstractMatrix,
)
    copyto!(workspace.proposed_logjoint, workspace.continuation_proposal_logjoint)
    copyto!(workspace.proposed_energy, workspace.continuation_proposed_energy)
    copyto!(workspace.energy_error, workspace.continuation_delta_energy)
    _mean_acceptance_stats!(
        workspace.accept_prob,
        workspace.continuation_accept_stat_sum,
        workspace.continuation_accept_stat_count,
    )
    _batched_positions_moved!(workspace.control.accepted_step, workspace.proposal_position, position)
    return workspace
end

function _mean_acceptance_probability(accept_prob::AbstractVector)
    isempty(accept_prob) && return 0.0
    return sum(accept_prob) / length(accept_prob)
end

function _mean_batched_adaptation_probability(
    accept_prob::AbstractVector,
    divergent::AbstractVector,
)
    length(accept_prob) == length(divergent) ||
        throw(DimensionMismatch("expected acceptance and divergence vectors of matching length, got $(length(accept_prob)) and $(length(divergent))"))
    isempty(accept_prob) && return 0.0

    total = 0.0
    for index in eachindex(accept_prob, divergent)
        total += divergent[index] ? 0.0 : accept_prob[index]
    end
    return total / length(accept_prob)
end

function _sample_nuts_direction(rng::AbstractRNG)
    return rand(rng, Bool) ? 1 : -1
end

function _sample_batched_nuts_directions!(
    destination::AbstractVector{Int},
    rng::AbstractRNG,
    active::AbstractVector{Bool},
)
    length(destination) == length(active) ||
        throw(DimensionMismatch("expected NUTS direction destination of length $(length(active)), got $(length(destination))"))
    for index in eachindex(destination, active)
        active[index] || continue
        destination[index] = _sample_nuts_direction(rng)
    end
    return destination
end

function _nuts_continuation_active(
    tree_depth::Integer,
    max_tree_depth::Integer,
    divergent::Bool,
    turning::Bool,
)
    return tree_depth < max_tree_depth && !divergent && !turning
end

function _nuts_subtree_start_state(
    continuation::NUTSContinuationState,
    direction::Int,
)
    return direction < 0 ? continuation.left : continuation.right
end

function _batched_leapfrog!(
    workspace::BatchedHMCWorkspace,
    model::TeaModel,
    position::Matrix{Float64},
    current_gradient::Matrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    num_leapfrog_steps::Int,
)
    q = workspace.proposal_position
    p = workspace.proposal_momentum
    proposed_gradient = workspace.proposal_gradient
    valid = workspace.valid
    num_chains = size(q, 2)
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(current_gradient) == size(position) ||
        throw(DimensionMismatch("expected current gradient workspace of size $(size(position)), got $(size(current_gradient))"))
    size(proposed_gradient) == size(position) ||
        throw(DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(proposed_gradient))"))

    copyto!(q, position)
    copyto!(p, workspace.momentum)
    fill!(valid, true)
    gradient = current_gradient

    for chain_index in 1:num_chains
        if !all(isfinite, view(gradient, :, chain_index))
            valid[chain_index] = false
        else
            p[:, chain_index] .+= (step_size / 2) .* gradient[:, chain_index]
        end
    end

    for leapfrog_step in 1:num_leapfrog_steps
        for chain_index in 1:num_chains
            valid[chain_index] || continue
            q[:, chain_index] .+= step_size .* (inverse_mass_matrix .* p[:, chain_index])
        end

        if leapfrog_step < num_leapfrog_steps
            gradient = batched_logjoint_gradient_unconstrained!(workspace.gradient_cache, q)
            for chain_index in 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(gradient, :, chain_index))
                    valid[chain_index] = false
                else
                    p[:, chain_index] .+= step_size .* gradient[:, chain_index]
                end
            end
        else
            proposed_logjoint, gradient = _batched_logjoint_and_gradient_unconstrained!(
                workspace.proposed_logjoint,
                workspace.gradient_cache,
                q,
            )
            copyto!(proposed_gradient, gradient)
            for chain_index in 1:num_chains
                valid[chain_index] || continue
                if !all(isfinite, view(proposed_gradient, :, chain_index)) || !isfinite(proposed_logjoint[chain_index])
                    valid[chain_index] = false
                end
            end
        end
    end

    for chain_index in 1:num_chains
        valid[chain_index] || continue
        p[:, chain_index] .+= (step_size / 2) .* proposed_gradient[:, chain_index]
        p[:, chain_index] .*= -1
    end

    return q, p, workspace.proposed_logjoint, proposed_gradient, valid
end

function _leapfrog(
    model::TeaModel,
    position::Vector{Float64},
    momentum::Vector{Float64},
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Vector{Float64},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    num_leapfrog_steps::Int,
)
    q = copy(position)
    p = copy(momentum)

    gradient = _logjoint_gradient!(gradient_cache, q)
    all(isfinite, gradient) || return nothing
    p .+= (step_size / 2) .* gradient

    for leapfrog_step in 1:num_leapfrog_steps
        q .+= step_size .* (inverse_mass_matrix .* p)
        gradient = _logjoint_gradient!(gradient_cache, q)
        all(isfinite, gradient) || return nothing

        if leapfrog_step < num_leapfrog_steps
            p .+= step_size .* gradient
        end
    end

    p .+= (step_size / 2) .* gradient
    p .*= -1

    proposed_logjoint = logjoint_unconstrained(model, q, args, constraints)
    isfinite(proposed_logjoint) || return nothing
    return q, p, proposed_logjoint
end

function _kinetic_energy(momentum::AbstractVector, inverse_mass_matrix::AbstractVector)
    return sum((momentum .^ 2) .* inverse_mass_matrix) / 2
end

function _hamiltonian(logjoint_value::Float64, momentum::AbstractVector, inverse_mass_matrix::AbstractVector)
    return -logjoint_value + _kinetic_energy(momentum, inverse_mass_matrix)
end

