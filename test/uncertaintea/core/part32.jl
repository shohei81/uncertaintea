    @tea static function dyadic_model()
        x ~ normal(0.0f0, 1.0f0)
        y ~ normal(0.0f0, 3.0f0)
        return x
    end

    @tea static function dyadic_model_1d()
        x ~ normal(0.0f0, 1.0f0)
        return x
    end

    function dyadic_new_state(num_params)
        return UncertainTea.NUTSState(zeros(num_params), zeros(num_params), 0.0, zeros(num_params))
    end

    function dyadic_leapfrog(target, state, imm, signed_step)
        dest = dyadic_new_state(length(state.position))
        ok = UncertainTea.leapfrog_step!(dest, target, state, imm, signed_step)
        return dest, ok
    end

    function dyadic_build_tree(target, state, imm, step_size, direction, depth)
        if depth == 0
            dest, _ = dyadic_leapfrog(target, state, imm, direction * step_size)
            return (leftmost=dest, rightmost=dest, turning=false, steps=1)
        end
        left = dyadic_build_tree(target, state, imm, step_size, direction, depth - 1)
        if left.turning
            return (leftmost=left.leftmost, rightmost=left.rightmost, turning=true, steps=left.steps)
        end
        right = dyadic_build_tree(target, left.rightmost, imm, step_size, direction, depth - 1)
        steps = left.steps + right.steps
        if right.turning
            return (leftmost=left.leftmost, rightmost=right.rightmost, turning=true, steps=steps)
        end
        if direction > 0
            turned = UncertainTea._is_turning(
                left.leftmost.position,
                right.rightmost.position,
                left.leftmost.momentum,
                right.rightmost.momentum,
            )
        else
            turned = UncertainTea._is_turning(
                right.rightmost.position,
                left.leftmost.position,
                right.rightmost.momentum,
                left.leftmost.momentum,
            )
        end
        return (leftmost=left.leftmost, rightmost=right.rightmost, turning=turned, steps=steps)
    end

    function dyadic_old_criterion(target, start_state, imm, step_size, direction, depth)
        state = start_state
        steps = 0
        turning = false
        for _ in 1:(1 << depth)
            dest, _ = dyadic_leapfrog(target, state, imm, direction * step_size)
            state = dest
            steps += 1
            if direction > 0
                turning = UncertainTea._is_turning(
                    start_state.position,
                    state.position,
                    start_state.momentum,
                    state.momentum,
                )
            else
                turning = UncertainTea._is_turning(
                    state.position,
                    start_state.position,
                    state.momentum,
                    start_state.momentum,
                )
            end
            turning && break
        end
        return (turning=turning, steps=steps)
    end

    dyadic_position = [1.6, -1.2]
    dyadic_momentum = [0.5, 0.9]
    dyadic_imm = [1.0, 1.0]
    dyadic_cache = UncertainTea._logjoint_gradient_cache(
        dyadic_model,
        dyadic_position,
        (),
        choicemap(),
    )
    dyadic_target = UncertainTea.ModelDensityTarget(
        dyadic_model,
        (),
        choicemap(),
        dyadic_cache,
    )
    dyadic_v0, dyadic_g0 = UncertainTea.target_logdensity_and_gradient!(dyadic_target, dyadic_position)
    dyadic_start_state = UncertainTea.NUTSState(
        copy(dyadic_position),
        copy(dyadic_momentum),
        dyadic_v0,
        copy(dyadic_g0),
    )
    dyadic_initial_hamiltonian = UncertainTea._hamiltonian(
        dyadic_start_state.logjoint,
        dyadic_start_state.momentum,
        dyadic_imm,
    )
    dyadic_max_delta_energy = 1.0e10
    dyadic_direction = 1

    dyadic_step_sizes = [0.3, 0.6, 0.9, 1.2, 1.5]
    dyadic_depths = [2, 3, 4]
    dyadic_turning_count, dyadic_diff_from_old = let
    turning_count = 0
    diff_from_old = 0
    for dyadic_step_size in dyadic_step_sizes
        for dyadic_depth in dyadic_depths
            dyadic_reference = dyadic_build_tree(
                dyadic_target,
                dyadic_start_state,
                dyadic_imm,
                dyadic_step_size,
                dyadic_direction,
                dyadic_depth,
            )
            dyadic_workspace = UncertainTea.NUTSSubtreeWorkspace(length(dyadic_position), 5)
            dyadic_summary = UncertainTea._build_nuts_subtree(
                dyadic_workspace,
                dyadic_target,
                dyadic_start_state,
                dyadic_imm,
                dyadic_step_size,
                dyadic_direction,
                dyadic_depth,
                dyadic_initial_hamiltonian,
                dyadic_max_delta_energy,
                MersenneTwister(hash((dyadic_step_size, dyadic_depth))),
            )
            @test dyadic_summary.divergent == false
            @test dyadic_summary.turning == dyadic_reference.turning
            @test dyadic_summary.integration_steps == dyadic_reference.steps
            dyadic_old = dyadic_old_criterion(
                dyadic_target,
                dyadic_start_state,
                dyadic_imm,
                dyadic_step_size,
                dyadic_direction,
                dyadic_depth,
            )
            if dyadic_reference.turning
                turning_count += 1
            end
            if dyadic_summary.turning != dyadic_old.turning ||
                dyadic_summary.integration_steps != dyadic_old.steps
                diff_from_old += 1
            end
        end
    end
    (turning_count, diff_from_old)
    end
    @test dyadic_turning_count >= 4
    @test dyadic_diff_from_old >= 1

    dyadic_e2e_chains = nuts_chains(
        dyadic_model,
        (),
        choicemap();
        num_chains=2,
        num_samples=200,
        num_warmup=200,
        step_size=0.4,
        max_tree_depth=8,
        rng=MersenneTwister(777),
    )
    dyadic_e2e_summary = summarize(dyadic_e2e_chains)
    for dyadic_row_index in 1:length(dyadic_e2e_summary)
        @test isfinite(dyadic_e2e_summary[dyadic_row_index].mean)
        @test isfinite(dyadic_e2e_summary[dyadic_row_index].sd)
        @test 1.0 <= dyadic_e2e_summary[dyadic_row_index].rhat <= 1.2
    end

    dyadic_depth_chain = nuts(
        dyadic_model_1d,
        (),
        choicemap();
        num_samples=300,
        num_warmup=300,
        step_size=0.4,
        rng=MersenneTwister(99),
    )
    dyadic_depth_values = treedepths(dyadic_depth_chain)
    @test all(isfinite, dyadic_depth_chain.unconstrained_samples)
    @test (sum(dyadic_depth_values) / length(dyadic_depth_values)) <= 5
