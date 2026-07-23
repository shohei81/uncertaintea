# Issue #159: Stan-style biased progressive sampling at doubling merges and
# per-iteration workspace hoisting for the single-chain NUTS proposal.
@testset "nuts_biased_merge_workspace_reuse" begin
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
