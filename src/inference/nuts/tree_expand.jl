# Shared per-particle NUTS tree math used by both the model-target batched
# cohort loops (nuts/kernel.jl) and the tempered SMC cohort loops
# (smc/moves_nuts.jl). Only the algorithmic core lives here; each caller keeps
# its own buffer plumbing. RNG draws happen in exactly the same order and count
# as the original inline code: a rand() is consumed only on the proposal-
# selection branch, and only when the running log weight is finite.

# Result of advancing one leaf for a single particle. isbits, so returning it
# does not allocate. When `divergent` is true only `delta_energy` is meaningful.
struct _NUTSLeafAdvance
    divergent::Bool
    delta_energy::Float64
    accept_prob::Float64
    candidate_log_weight::Float64
    combined_log_weight::Float64
    select_proposal::Bool
end

# Per-particle leaf advance: delta-energy divergence check, accept-prob, and the
# multinomial progressive proposal selection (logaddexp + one rand draw). This
# is the single insertion point for future dyadic U-turn checkpoint logic.
function _advance_tree_leaf(
    proposed_energy::Float64,
    reference_energy::Float64,
    max_delta_energy::Float64,
    log_weight::Float64,
    rng::AbstractRNG,
)
    delta_energy = proposed_energy - reference_energy
    if !isfinite(delta_energy) || delta_energy > max_delta_energy
        return _NUTSLeafAdvance(true, delta_energy, 0.0, -Inf, log_weight, false)
    end
    accept_prob = min(1.0, exp(min(0.0, -delta_energy)))
    candidate_log_weight = -proposed_energy
    combined_log_weight = _logaddexp(log_weight, candidate_log_weight)
    select_proposal =
        !isfinite(log_weight) ||
        log(rand(rng)) < candidate_log_weight - combined_log_weight
    return _NUTSLeafAdvance(
        false,
        delta_energy,
        accept_prob,
        candidate_log_weight,
        combined_log_weight,
        select_proposal,
    )
end

# Result of merging one subtree into its continuation for a single particle.
struct _NUTSSubtreeMerge
    candidate_log_weight::Float64
    combined_log_weight::Float64
    select_proposal::Bool
end

# Generalized (dyadic) U-turn checkpoint math, shared by the scalar
# `_build_nuts_subtree` and both batched cohort expand loops. `leaf_index` is
# the 0-based index of the leaf just produced within the current subtree
# (`completed-steps-in-subtree - 1`). Checkpoint columns are indexed by
# `count_ones(block_start) + 1`; callers pass per-particle checkpoint views.

# Store the (position, momentum) of an even leaf into its checkpoint slot.
# Caller must ensure `iseven(leaf_index)`. The slot is `count_ones(leaf_index)+1`.
function _store_tree_checkpoint!(
    checkpoint_positions::AbstractMatrix,
    checkpoint_momenta::AbstractMatrix,
    leaf_index::Int,
    position::AbstractVector,
    momentum::AbstractVector,
)
    slot = count_ones(leaf_index) + 1
    copyto!(view(checkpoint_positions, :, slot), position)
    copyto!(view(checkpoint_momenta, :, slot), momentum)
    return nothing
end

# Odd-leaf dyadic U-turn test: for each dyadic block ending at `leaf_index`,
# compare the block's start checkpoint against the current endpoint. `direction`
# selects the argument orientation (>0 forward, <0 backward). Returns true as
# soon as any block turns. Caller must ensure `isodd(leaf_index)`.
function _dyadic_turning(
    checkpoint_positions::AbstractMatrix,
    checkpoint_momenta::AbstractMatrix,
    leaf_index::Int,
    position::AbstractVector,
    momentum::AbstractVector,
    direction::Int,
)
    for k = 1:trailing_ones(leaf_index)
        block_start = leaf_index - (1 << k) + 1
        slot = count_ones(block_start) + 1
        ckpt_position = view(checkpoint_positions, :, slot)
        ckpt_momentum = view(checkpoint_momenta, :, slot)
        turned = if direction > 0
            _is_turning(ckpt_position, position, ckpt_momentum, momentum)
        else
            _is_turning(position, ckpt_position, momentum, ckpt_momentum)
        end
        turned && return true
    end
    return false
end

# Per-particle subtree-into-continuation merge: combined log weight and the
# proposal swap decision (one rand draw, consumed only when the subtree log
# weight is finite).
function _merge_subtree_stats(
    continuation_log_weight::Float64,
    subtree_log_weight::Float64,
    rng::AbstractRNG,
)
    if !isfinite(subtree_log_weight)
        return _NUTSSubtreeMerge(-Inf, continuation_log_weight, false)
    end
    combined_log_weight = _logaddexp(continuation_log_weight, subtree_log_weight)
    select_proposal =
        log(rand(rng)) < subtree_log_weight - combined_log_weight
    return _NUTSSubtreeMerge(subtree_log_weight, combined_log_weight, select_proposal)
end
