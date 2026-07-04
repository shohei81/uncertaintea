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
