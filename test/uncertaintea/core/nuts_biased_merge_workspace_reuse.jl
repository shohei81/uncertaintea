# Issue #159: Stan-style biased progressive sampling at doubling merges and
# per-iteration workspace hoisting for the single-chain NUTS proposal.
@testset "nuts_biased_merge_workspace_reuse" begin
    @testset "doubling merge uses Stan's biased progressive swap" begin
        # A subtree at least as heavy as the continuation is ALWAYS selected
        # (P = min(1, w_new/w_old) = 1), independent of the RNG draw.
        for seed = 1:32
            merge_equal = UncertainTea._merge_subtree_stats(
                -3.0,
                -3.0,
                MersenneTwister(seed),
            )
            @test merge_equal.select_proposal
            merge_heavier = UncertainTea._merge_subtree_stats(
                -3.0,
                -2.0,
                MersenneTwister(seed),
            )
            @test merge_heavier.select_proposal
        end

        # Combined log weight is untouched by the bias: still logaddexp.
        merge_stats = UncertainTea._merge_subtree_stats(-1.0, -2.0, MersenneTwister(7))
        @test merge_stats.combined_log_weight ≈
              UncertainTea._logaddexp(-1.0, -2.0)
        @test merge_stats.candidate_log_weight == -2.0

        # A -Inf subtree never merges and never consumes a draw.
        merge_rng = MersenneTwister(11)
        merge_invalid = UncertainTea._merge_subtree_stats(-1.0, -Inf, merge_rng)
        @test !merge_invalid.select_proposal
        @test merge_invalid.combined_log_weight == -1.0
        @test merge_rng == MersenneTwister(11)

        # Empirical swap rate for w_new/w_old = 1/2: biased P = 0.5 (the
        # unbiased rule would give 1/3). 20k draws => 3-sigma band ~ +-0.011.
        rate_rng = MersenneTwister(20260723)
        swap_count = 0
        num_draws = 20_000
        for _ = 1:num_draws
            merge = UncertainTea._merge_subtree_stats(0.0, -log(2.0), rate_rng)
            swap_count += merge.select_proposal
        end
        @test abs(swap_count / num_draws - 0.5) < 0.015

        # WITHIN-subtree leaf selection stays unbiased: candidate with equal
        # weight is selected with P = 1/2 (biased would be 1), and a candidate
        # with half the running weight with P = 1/3.
        leaf_rng = MersenneTwister(20260724)
        leaf_select_equal = 0
        leaf_select_half = 0
        for _ = 1:num_draws
            leaf_equal = UncertainTea._advance_tree_leaf(1.0, 1.0, 1000.0, -1.0, leaf_rng)
            leaf_select_equal += leaf_equal.select_proposal
            leaf_half =
                UncertainTea._advance_tree_leaf(1.0, 1.0, 1000.0, -1.0 + log(2.0), leaf_rng)
            leaf_select_half += leaf_half.select_proposal
        end
        @test abs(leaf_select_equal / num_draws - 0.5) < 0.015
        @test abs(leaf_select_half / num_draws - 1.0 / 3.0) < 0.015
    end

    @testset "hoisted workspace reuse is bitwise-identical to fresh allocation" begin
        reuse_constraints = choicemap((:y, 0.4f0))
        reuse_position = [0.3]
        reuse_cache = UncertainTea._logjoint_gradient_cache(
            gaussian_mean,
            reuse_position,
            (),
            reuse_constraints,
        )
        reuse_target = UncertainTea.ModelDensityTarget(
            gaussian_mean,
            (),
            reuse_constraints,
            reuse_cache,
        )
        reuse_imm = [1.0]
        reuse_max_depth = 5

        # (a) hoisted workspaces reused across sequential transitions.
        reuse_rng = MersenneTwister(20260723)
        reuse_tree_workspace = UncertainTea.NUTSSubtreeWorkspace(1, reuse_max_depth)
        reuse_continuation = UncertainTea.NUTSContinuationState(1)
        reuse_state_position = copy(reuse_position)
        reuse_logjoint = UncertainTea.logjoint_unconstrained(
            gaussian_mean,
            reuse_state_position,
            (),
            reuse_constraints,
        )
        reuse_gradient =
            copy(UncertainTea._logjoint_gradient!(reuse_cache, reuse_state_position))
        reuse_trace = Vector{Float64}[]
        for _ = 1:20
            proposal, _, _, _, _, _, _, moved = UncertainTea._nuts_proposal(
                reuse_tree_workspace,
                reuse_continuation,
                reuse_target,
                reuse_state_position,
                reuse_logjoint,
                reuse_gradient,
                reuse_imm,
                0.25,
                reuse_max_depth,
                1000.0,
                reuse_rng,
            )
            if moved
                copyto!(reuse_state_position, proposal.position)
                reuse_logjoint = proposal.logjoint
                copyto!(reuse_gradient, proposal.gradient)
            end
            push!(reuse_trace, copy(reuse_state_position))
        end

        # (b) the allocating wrapper (fresh workspaces per call), same seed.
        fresh_rng = MersenneTwister(20260723)
        fresh_state_position = copy(reuse_position)
        fresh_logjoint = UncertainTea.logjoint_unconstrained(
            gaussian_mean,
            fresh_state_position,
            (),
            reuse_constraints,
        )
        fresh_gradient =
            copy(UncertainTea._logjoint_gradient!(reuse_cache, fresh_state_position))
        fresh_trace = Vector{Float64}[]
        for _ = 1:20
            proposal, _, _, _, _, _, _, moved = UncertainTea._nuts_proposal(
                reuse_target,
                fresh_state_position,
                fresh_logjoint,
                fresh_gradient,
                reuse_imm,
                0.25,
                reuse_max_depth,
                1000.0,
                fresh_rng,
            )
            if moved
                fresh_state_position = copy(proposal.position)
                fresh_logjoint = proposal.logjoint
                fresh_gradient = copy(proposal.gradient)
            end
            push!(fresh_trace, copy(fresh_state_position))
        end

        @test reuse_trace == fresh_trace
    end
end
