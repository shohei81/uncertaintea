abstract type AbstractTemperedNUTSSchedulerIR end

struct TemperedNUTSIdleIR <: AbstractTemperedNUTSSchedulerIR end

struct TemperedNUTSExpandIR <: AbstractTemperedNUTSSchedulerIR
    active_depth::Int
    active_depth_count::Int
    remaining_steps::Int
    active_particles::BitVector
    directions::Vector{Int}
end

struct TemperedNUTSMergeIR <: AbstractTemperedNUTSSchedulerIR
    active_depth::Int
    active_depth_count::Int
    active_particles::BitVector
end

struct TemperedNUTSDoneIR <: AbstractTemperedNUTSSchedulerIR end

function _begin_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    max_tree_depth::Int,
    rng::AbstractRNG,
)
    scheduler = workspace.scheduler
    active_depth, active_depth_count = _select_tempered_nuts_depth_cohort!(
        workspace,
        continuations,
        max_tree_depth,
    )
    if active_depth_count == 0
        fill!(scheduler.cohort_active, false)
        scheduler.phase = TemperedNUTSSchedulerDone
        scheduler.remaining_steps = 0
        return scheduler.phase
    end

    copyto!(
        scheduler.cohort_active,
        _activate_tempered_nuts_depth_cohort!(workspace, continuations, active_depth),
    )
    scheduler.phase = TemperedNUTSSchedulerExpand
    scheduler.remaining_steps = 1 << active_depth
    _initialize_tempered_nuts_depth_cohort!(workspace, continuations, active_depth, rng)
    return scheduler.phase
end

function _tempered_nuts_scheduler_ir(workspace::TemperedNUTSMoveWorkspace)
    scheduler = workspace.scheduler
    if scheduler.phase === TemperedNUTSSchedulerIdle
        return TemperedNUTSIdleIR()
    elseif scheduler.phase === TemperedNUTSSchedulerExpand
        return TemperedNUTSExpandIR(
            scheduler.active_depth,
            scheduler.active_depth_count,
            scheduler.remaining_steps,
            copy(scheduler.cohort_active),
            copy(workspace.control.directions),
        )
    elseif scheduler.phase === TemperedNUTSSchedulerMerge
        return TemperedNUTSMergeIR(
            scheduler.active_depth,
            scheduler.active_depth_count,
            copy(scheduler.cohort_active),
        )
    end
    return TemperedNUTSDoneIR()
end

function _step_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    ::TemperedNUTSIdleIR,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    _begin_tempered_nuts_cohort_scheduler!(workspace, continuations, max_tree_depth, rng)
    return workspace.scheduler.phase
end

function _step_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    ir::TemperedNUTSExpandIR,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    scheduler = workspace.scheduler
    scheduler.active_depth = ir.active_depth
    scheduler.active_depth_count = ir.active_depth_count
    scheduler.remaining_steps = ir.remaining_steps
    copyto!(scheduler.cohort_active, ir.active_particles)
    copyto!(workspace.control.active, ir.active_particles)
    copyto!(workspace.control.directions, ir.directions)
    _expand_tempered_nuts_depth_cohort!(
        workspace,
        model,
        cache,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
        step_size,
        max_delta_energy,
        initial_hamiltonian,
        inverse_mass_matrix,
        rng,
    )
    scheduler.phase = TemperedNUTSSchedulerMerge
    scheduler.remaining_steps = 0
    return scheduler.phase
end

function _step_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    ir::TemperedNUTSMergeIR,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    scheduler = workspace.scheduler
    scheduler.active_depth = ir.active_depth
    scheduler.active_depth_count = ir.active_depth_count
    copyto!(scheduler.cohort_active, ir.active_particles)
    copyto!(workspace.control.active, ir.active_particles)
    _merge_tempered_nuts_depth_cohort!(workspace, continuations, rng)
    _begin_tempered_nuts_cohort_scheduler!(workspace, continuations, max_tree_depth, rng)
    return scheduler.phase
end

function _step_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    ::TemperedNUTSDoneIR,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    workspace.scheduler.phase = TemperedNUTSSchedulerDone
    workspace.scheduler.remaining_steps = 0
    return workspace.scheduler.phase
end

function _continue_batched_tempered_nuts_cohorts!(
    workspace::TemperedNUTSMoveWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    _begin_tempered_nuts_cohort_scheduler!(workspace, continuations, max_tree_depth, rng)
    while workspace.scheduler.phase !== TemperedNUTSSchedulerDone
        ir = _tempered_nuts_scheduler_ir(workspace)
        _step_tempered_nuts_cohort_scheduler!(
            workspace,
            ir,
            continuations,
            model,
            cache,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
            step_size,
            max_tree_depth,
            max_delta_energy,
            initial_hamiltonian,
            inverse_mass_matrix,
            rng,
        )
    end
    return continuations
end
