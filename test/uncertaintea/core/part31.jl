    @tea static function integrator_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 0.8f0)
        return mu
    end

    integ_constraints = choicemap((:y, 0.4f0))
    integ_position = [0.3]
    integ_momentum = [0.7]
    integ_imm = [1.0]
    integ_cache = UncertainTea._logjoint_gradient_cache(
        integrator_model,
        integ_position,
        (),
        integ_constraints,
    )
    integ_target = UncertainTea.ModelDensityTarget(
        integrator_model,
        (),
        integ_constraints,
        integ_cache,
    )

    integ_forward = UncertainTea.leapfrog_trajectory(
        integ_target,
        integ_position,
        integ_momentum,
        integ_imm,
        0.1,
        8,
    )
    @test !isnothing(integ_forward)
    integ_q1, integ_p1, integ_v1 = integ_forward
    @test isfinite(integ_v1)
    integ_back = UncertainTea.leapfrog_trajectory(
        integ_target,
        integ_q1,
        integ_p1,
        integ_imm,
        0.1,
        8,
    )
    @test !isnothing(integ_back)
    integ_q0, integ_p0, _ = integ_back
    @test integ_q0 ≈ integ_position atol=1e-10
    @test integ_p0 ≈ integ_momentum atol=1e-10

    integ_v0, integ_g0 = UncertainTea.target_logdensity_and_gradient!(integ_target, integ_position)
    integ_state = UncertainTea.NUTSState(copy(integ_position), copy(integ_momentum), integ_v0, copy(integ_g0))
    integ_dest_model = UncertainTea.NUTSState(zeros(1), zeros(1), 0.0, zeros(1))
    integ_dest_tempered = UncertainTea.NUTSState(zeros(1), zeros(1), 0.0, zeros(1))
    integ_tempered_target = UncertainTea.TemperedDensityTarget(
        integrator_model,
        (),
        integ_constraints,
        integ_cache,
        [0.0],
        [0.0],
        zeros(1),
        zeros(1),
        1.0,
    )

    @test UncertainTea.leapfrog_step!(integ_dest_model, integ_target, integ_state, integ_imm, 0.05)
    @test UncertainTea.leapfrog_step!(integ_dest_tempered, integ_tempered_target, integ_state, integ_imm, 0.05)
    @test integ_dest_tempered.position ≈ integ_dest_model.position atol=1e-12
    @test integ_dest_tempered.momentum ≈ integ_dest_model.momentum atol=1e-12
    @test integ_dest_tempered.logjoint ≈ integ_dest_model.logjoint atol=1e-12

    integ_batch_positions = reshape([0.1, 0.3, -0.5], 1, 3)
    integ_batch_momentum = reshape([0.4, -0.2, 0.9], 1, 3)
    integ_batch_cache = BatchedLogjointGradientCache(
        integrator_model,
        integ_batch_positions,
        (),
        integ_constraints,
    )
    integ_batch_target = UncertainTea.BatchedModelDensityTarget(integ_batch_cache)
    integ_batch_gradient = copy(UncertainTea.batched_target_gradient!(
        similar(integ_batch_positions),
        integ_batch_target,
        integ_batch_positions,
    ))
    integ_dest_position = similar(integ_batch_positions)
    integ_dest_momentum = similar(integ_batch_positions)
    integ_dest_gradient = similar(integ_batch_positions)
    integ_dest_logjoint = zeros(3)
    integ_valid = fill(false, 3)
    integ_directions = [1, -1, 1]

    UncertainTea.batched_leapfrog_step_to!(
        integ_dest_position,
        integ_dest_momentum,
        integ_dest_gradient,
        integ_dest_logjoint,
        integ_valid,
        integ_batch_positions,
        integ_batch_momentum,
        integ_batch_gradient,
        integ_batch_target,
        integ_imm,
        0.05,
        integ_directions,
        trues(3),
    )
    @test all(integ_valid)

    for index in 1:3
        column_state = UncertainTea.NUTSState(
            [integ_batch_positions[1, index]],
            [integ_batch_momentum[1, index]],
            0.0,
            [integ_batch_gradient[1, index]],
        )
        column_dest = UncertainTea.NUTSState(zeros(1), zeros(1), 0.0, zeros(1))
        @test UncertainTea.leapfrog_step!(
            column_dest,
            integ_target,
            column_state,
            integ_imm,
            integ_directions[index] * 0.05,
        )
        @test column_dest.position[1] ≈ integ_dest_position[1, index] atol=1e-10
        @test column_dest.momentum[1] ≈ integ_dest_momentum[1, index] atol=1e-10
        @test column_dest.logjoint ≈ integ_dest_logjoint[index] atol=1e-10
    end
