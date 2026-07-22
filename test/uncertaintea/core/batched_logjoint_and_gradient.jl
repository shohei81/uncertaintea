@testset "batched_logjoint_and_gradient" begin
    positive_step_spec = modelspec(observed_positive_step)
    positive_step_params = parameter_vector(positive_step_trace)
    positive_step_reconstrained = transform_to_constrained(observed_positive_step, positive_step_unconstrained)
    gaussian_batch_params_shifted = gaussian_batch_params .+ 0.15
    gaussian_batch_gradient_shifted = batched_logjoint_gradient_unconstrained(
        gaussian_batch_gradient_cache,
        gaussian_batch_params_shifted,
    )
    # PR-4: the batched workspace is keyed off the conditioning signature, so it
    # is constructed with the constraints it will score against.
    gaussian_workspace = UncertainTea.BatchedLogjointWorkspace(gaussian_mean, gaussian_batch_constraints)
    heterogeneous_iid_args = Any[(Int32(2),), (3,)]
    heterogeneous_iid_gradient_cache = BatchedLogjointGradientCache(
        iid_model,
        iid_batch_params,
        heterogeneous_iid_args,
        iid_batch_constraints,
    )
    iid_batch_logjoint = batched_logjoint(iid_model, iid_batch_params, iid_batch_args, iid_batch_constraints)
    iid_shared_batch_params = reshape(Float64[-0.15, 0.25], 1, 2)
    iid_shared_batch_constraints = [
        choicemap((:y => 1, 0.0f0), (:y => 2, -0.1f0), (:y => 3, 0.3f0)),
        choicemap((:y => 1, 0.4f0), (:y => 2, 0.2f0), (:y => 3, -0.2f0)),
    ]
    iid_shared_workspace = UncertainTea.BatchedLogjointWorkspace(iid_model, iid_shared_batch_constraints)
    iid_shared_batch_logjoint = batched_logjoint(
        iid_model,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints,
    )
    iid_shared_single_constraint_logjoint = batched_logjoint(
        iid_model,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints[1],
    )
    iid_shared_gradient_cache = BatchedLogjointGradientCache(
        iid_model,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints,
    )
    shifted_batch_params = reshape(Float64[-0.1, 0.3], 1, 2)
    shifted_batch_constraints = [
        choicemap((:y => 2, 0.0f0), (:y => 3, -0.1f0), (:y => 4, 0.2f0)),
        choicemap((:y => 2, 0.5f0), (:y => 3, 0.1f0), (:y => 4, -0.3f0)),
    ]
    shifted_batch_logjoint = batched_logjoint(
        shifted_iid_model,
        shifted_batch_params,
        (3,),
        shifted_batch_constraints,
    )
    offset_batch_params = reshape(Float64[-0.25, 0.15], 1, 2)
    offset_batch_args = [(3, 1), (3, 2)]
    offset_batch_constraints = [
        choicemap((:y => 2, 0.0f0), (:y => 3, -0.1f0), (:y => 4, 0.2f0)),
        choicemap((:y => 3, 0.5f0), (:y => 4, 0.1f0), (:y => 5, -0.3f0)),
    ]
    offset_batch_logjoint = batched_logjoint(
        offset_iid_model,
        offset_batch_params,
        offset_batch_args,
        offset_batch_constraints,
    )
    indexed_scale_batch_params = reshape(Float64[-0.2, 0.35], 1, 2)
    indexed_scale_batch_constraints = [
        choicemap((:y => 1, 0.2f0), (:y => 2, -0.1f0), (:y => 3, 0.5f0)),
        choicemap((:y => 1, -0.4f0), (:y => 2, 0.1f0), (:y => 3, 0.3f0)),
    ]
    indexed_scale_batch_logjoint = batched_logjoint(
        indexed_scale_model,
        indexed_scale_batch_params,
        (3,),
        indexed_scale_batch_constraints,
    )
    deterministic_batch_params = [
        deterministic_params[1] deterministic_params[1] + 0.15 deterministic_params[1] - 0.2
        deterministic_params[2] deterministic_params[2] - 0.05 deterministic_params[2] + 0.1
    ]
    deterministic_batch_constraints = [
        choicemap((:y, 0.4f0)),
        choicemap((:y, -0.1f0)),
        choicemap((:y, 0.8f0)),
    ]
    deterministic_batch_logjoint = batched_logjoint(
        deterministic_scale,
        deterministic_batch_params,
        (),
        deterministic_batch_constraints,
    )
    deterministic_workspace =
        UncertainTea.BatchedLogjointWorkspace(deterministic_scale, deterministic_batch_constraints)
    offset_workspace = UncertainTea.BatchedLogjointWorkspace(offset_iid_model, offset_batch_constraints)
    positive_batch_logjoint = batched_logjoint_unconstrained(
        observed_positive_step,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    )
    positive_batch_gradient = batched_logjoint_gradient_unconstrained(
        observed_positive_step,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    )
    positive_batch_gradient_cache = BatchedLogjointGradientCache(
        observed_positive_step,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    )
    abs_scale_batch_params = reshape(
        [abs_scale_params[1] - 0.25, abs_scale_params[1], abs_scale_params[1] + 0.35],
        1,
        3,
    )
    abs_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.6f0)),
        choicemap((:y, 1.1f0)),
    ]
    abs_scale_gradient = batched_logjoint_gradient_unconstrained(
        abs_scale_model,
        abs_scale_batch_params,
        (),
        abs_scale_batch_constraints,
    )
    abs_scale_gradient_cache = BatchedLogjointGradientCache(
        abs_scale_model,
        abs_scale_batch_params,
        (),
        abs_scale_batch_constraints,
    )
    power_scale_batch_params = reshape(
        [power_scale_params[1] - 0.2, power_scale_params[1], power_scale_params[1] + 0.3],
        1,
        3,
    )
    power_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.4f0)),
        choicemap((:y, 0.9f0)),
    ]
    power_scale_gradient = batched_logjoint_gradient_unconstrained(
        power_scale_model,
        power_scale_batch_params,
        (),
        power_scale_batch_constraints,
    )
    power_scale_gradient_cache = BatchedLogjointGradientCache(
        power_scale_model,
        power_scale_batch_params,
        (),
        power_scale_batch_constraints,
    )
    min_scale_batch_params = reshape(
        [min_scale_params[1] - 0.35, 0.0, min_scale_params[1] + 0.45],
        1,
        3,
    )
    min_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.5f0)),
        choicemap((:y, 0.9f0)),
    ]
    min_scale_gradient = batched_logjoint_gradient_unconstrained(
        min_scale_model,
        min_scale_batch_params,
        (),
        min_scale_batch_constraints,
    )
    min_scale_gradient_cache = BatchedLogjointGradientCache(
        min_scale_model,
        min_scale_batch_params,
        (),
        min_scale_batch_constraints,
    )
    max_scale_batch_params = reshape(
        [max_scale_params[1] - 0.45, -0.1, max_scale_params[1] + 0.35],
        1,
        3,
    )
    max_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.4f0)),
        choicemap((:y, 0.8f0)),
    ]
    max_scale_gradient = batched_logjoint_gradient_unconstrained(
        max_scale_model,
        max_scale_batch_params,
        (),
        max_scale_batch_constraints,
    )
    max_scale_gradient_cache = BatchedLogjointGradientCache(
        max_scale_model,
        max_scale_batch_params,
        (),
        max_scale_batch_constraints,
    )
    mod_scale_batch_params = reshape(
        [0.1, 0.6, 0.85],
        1,
        3,
    )
    mod_scale_batch_constraints = [
        choicemap((:y, 0.2f0)),
        choicemap((:y, 0.4f0)),
        choicemap((:y, 0.8f0)),
    ]
    mod_scale_gradient = batched_logjoint_gradient_unconstrained(
        mod_scale_model,
        mod_scale_batch_params,
        (),
        mod_scale_batch_constraints,
    )
    mod_scale_gradient_cache = BatchedLogjointGradientCache(
        mod_scale_model,
        mod_scale_batch_params,
        (),
        mod_scale_batch_constraints,
    )
    clamp_scale_batch_params = reshape(
        [clamp_scale_params[1] - 0.55, 0.05, clamp_scale_params[1] + 0.45],
        1,
        3,
    )
    clamp_scale_batch_constraints = [
        choicemap((:y, 0.25f0)),
        choicemap((:y, 0.5f0)),
        choicemap((:y, 0.85f0)),
    ]
    clamp_scale_gradient = batched_logjoint_gradient_unconstrained(
        clamp_scale_model,
        clamp_scale_batch_params,
        (),
        clamp_scale_batch_constraints,
    )
    clamp_scale_gradient_cache = BatchedLogjointGradientCache(
        clamp_scale_model,
        clamp_scale_batch_params,
        (),
        clamp_scale_batch_constraints,
    )

    @test parametercount(positive_step_spec.parameter_layout) == 1
    @test positive_step_spec.parameter_layout.slots[1].transform isa LogTransform
    @test positive_step_params[1] == Float64(positive_step_trace[:state=>:sigma])
    @test positive_step_unconstrained[1] ≈ log(positive_step_params[1])
    @test positive_step_reconstrained ≈ positive_step_params
    @test logjoint_unconstrained(observed_positive_step, positive_step_unconstrained, (), choicemap((:y, 1.2f0))) ≈
          logjoint(observed_positive_step, positive_step_params, (), choicemap((:y, 1.2f0))) +
          positive_step_unconstrained[1] atol=1e-6
    @test logjoint(observed_positive_step, positive_step_params, (), choicemap((:y, 1.2f0))) ≈
          assess(
        observed_positive_step,
        (),
        choicemap((:state => :sigma, positive_step_trace[:state=>:sigma]), (:y, 1.2f0)),
    ) atol=1e-6
    @test gaussian_batch_logjoint ≈ [
        logjoint(gaussian_mean, gaussian_batch_params[:, index], (), gaussian_batch_constraints[index]) for index = 1:3
    ] atol=1e-8
    @test gaussian_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(gaussian_mean, gaussian_batch_params[:, index], (), gaussian_batch_constraints[index])
            for
            index = 1:3
        ]...,
    ) atol=1e-8
    @test batched_logjoint_gradient_unconstrained!(gaussian_batch_gradient_cache, gaussian_batch_params) ≈
          gaussian_batch_gradient atol=1e-8
    @test !isnothing(gaussian_batch_gradient_cache.backend_cache)
    @test isnothing(gaussian_batch_gradient_cache.flat_cache)
    @test isempty(gaussian_batch_gradient_cache.column_caches)
    @test gaussian_batch_gradient_shifted ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                gaussian_mean,
                gaussian_batch_params_shifted[:, index],
                (),
                gaussian_batch_constraints[index],
            ) for
            index = 1:3
        ]...,
    ) atol=1e-8
    gaussian_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        gaussian_workspace,
        gaussian_batch_params,
        (),
        gaussian_batch_constraints,
    )
    gaussian_workspace_env = gaussian_workspace.batched_environment[]
    gaussian_workspace_totals = gaussian_workspace.batched_totals_buffer[]
    gaussian_workspace_observed = gaussian_workspace_env.observed_values
    @test gaussian_workspace_values ≈ gaussian_batch_logjoint atol=1e-8
    @test UncertainTea._logjoint_with_batched_backend!(
        gaussian_workspace,
        gaussian_batch_params_shifted,
        (),
        gaussian_batch_constraints,
    ) ≈ [
        logjoint(gaussian_mean, gaussian_batch_params_shifted[:, index], (), gaussian_batch_constraints[index]) for index = 1:3
    ] atol=1e-8
    @test gaussian_workspace.batched_environment[] === gaussian_workspace_env
    @test gaussian_workspace.batched_totals_buffer[] === gaussian_workspace_totals
    @test gaussian_workspace_env.observed_values === gaussian_workspace_observed
    @test iid_batch_logjoint ≈ [
        logjoint(iid_model, iid_batch_params[:, index], iid_batch_args[index], iid_batch_constraints[index]) for index = 1:2
    ] atol=1e-8
    @test batched_logjoint_gradient_unconstrained!(
        heterogeneous_iid_gradient_cache,
        iid_batch_params,
    ) ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                iid_model,
                iid_batch_params[:, index],
                heterogeneous_iid_args[index],
                iid_batch_constraints[index],
            ) for index = 1:2
        ]...,
    ) atol=1e-8
    @test isnothing(heterogeneous_iid_gradient_cache.backend_cache)
    @test !isnothing(heterogeneous_iid_gradient_cache.flat_cache)
    @test isempty(heterogeneous_iid_gradient_cache.column_caches)
    @test iid_shared_batch_logjoint ≈ [
        logjoint(iid_model, iid_shared_batch_params[:, index], (3,), iid_shared_batch_constraints[index]) for index = 1:2
    ] atol=1e-8
    @test !isnothing(iid_shared_gradient_cache.backend_cache)
    @test isnothing(iid_shared_gradient_cache.flat_cache)
    @test isempty(iid_shared_gradient_cache.column_caches)
    @test iid_shared_single_constraint_logjoint ≈ [
        logjoint(iid_model, iid_shared_batch_params[:, index], (3,), iid_shared_batch_constraints[1]) for index = 1:2
    ] atol=1e-8
    iid_shared_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        iid_shared_workspace,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints,
    )
    iid_shared_workspace_env = iid_shared_workspace.batched_environment[]
    iid_shared_iterable_scratch = iid_shared_workspace_env.index_scratch[1]
    iid_shared_observed_values = iid_shared_workspace_env.observed_values
    @test iid_shared_workspace_values ≈ iid_shared_batch_logjoint atol=1e-8
    @test UncertainTea._logjoint_with_batched_backend!(
        iid_shared_workspace,
        iid_shared_batch_params,
        (3,),
        iid_shared_batch_constraints[1],
    ) ≈ iid_shared_single_constraint_logjoint atol=1e-8
    @test !isempty(iid_shared_workspace_env.index_scratch)
    @test UncertainTea._logjoint_with_batched_backend!(
        iid_shared_workspace,
        iid_shared_batch_params .+ 0.1,
        (3,),
        iid_shared_batch_constraints,
    ) ≈ [
        logjoint(iid_model, (iid_shared_batch_params .+ 0.1)[:, index], (3,), iid_shared_batch_constraints[index]) for
        index = 1:2
    ] atol=1e-8
    @test iid_shared_workspace.batched_environment[] === iid_shared_workspace_env
    @test iid_shared_workspace_env.index_scratch[1] === iid_shared_iterable_scratch
    @test iid_shared_workspace_env.observed_values === iid_shared_observed_values
    @test shifted_batch_logjoint ≈ [
        logjoint(shifted_iid_model, shifted_batch_params[:, index], (3,), shifted_batch_constraints[index]) for index = 1:2
    ] atol=1e-8
    @test offset_batch_logjoint ≈ [
        logjoint(offset_iid_model, offset_batch_params[:, index], offset_batch_args[index], offset_batch_constraints[index]) for
        index = 1:2
    ] atol=1e-8
    offset_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        offset_workspace,
        offset_batch_params,
        offset_batch_args,
        offset_batch_constraints,
    )
    offset_workspace_env = offset_workspace.batched_environment[]
    offset_workspace_scratch = offset_workspace_env.index_scratch[1]
    @test offset_workspace_values ≈ offset_batch_logjoint atol=1e-8
    @test !isempty(offset_workspace_env.index_scratch)
    @test UncertainTea._logjoint_with_batched_backend!(
        offset_workspace,
        offset_batch_params .+ 0.05,
        offset_batch_args,
        offset_batch_constraints,
    ) ≈ [
        logjoint(
            offset_iid_model,
            (offset_batch_params .+ 0.05)[:, index],
            offset_batch_args[index],
            offset_batch_constraints[index],
        ) for index = 1:2
    ] atol=1e-8
    @test offset_workspace.batched_environment[] === offset_workspace_env
    @test offset_workspace_env.index_scratch[1] === offset_workspace_scratch
    @test indexed_scale_batch_logjoint ≈ [
        logjoint(
            indexed_scale_model,
            indexed_scale_batch_params[:, index],
            (3,),
            indexed_scale_batch_constraints[index],
        ) for index = 1:2
    ] atol=1e-8
    deterministic_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        deterministic_workspace,
        deterministic_batch_params,
        (),
        deterministic_batch_constraints,
    )
    deterministic_workspace_env = deterministic_workspace.batched_environment[]
    deterministic_workspace_scratch = deterministic_workspace_env.numeric_scratch[1]
    @test deterministic_batch_logjoint ≈ [
        logjoint(
            deterministic_scale,
            deterministic_batch_params[:, index],
            (),
            deterministic_batch_constraints[index],
        ) for index = 1:3
    ] atol=1e-8
    @test deterministic_workspace_values ≈ deterministic_batch_logjoint atol=1e-8
    @test !isempty(deterministic_workspace_env.numeric_scratch)
    @test UncertainTea._logjoint_with_batched_backend!(
        deterministic_workspace,
        deterministic_batch_params .+ [0.05; -0.02],
        (),
        deterministic_batch_constraints,
    ) ≈ [
        logjoint(
            deterministic_scale,
            (deterministic_batch_params .+ [0.05; -0.02])[:, index],
            (),
            deterministic_batch_constraints[index],
        ) for index = 1:3
    ] atol=1e-8
    @test deterministic_workspace.batched_environment[] === deterministic_workspace_env
    @test deterministic_workspace_env.numeric_scratch[1] === deterministic_workspace_scratch
    @test positive_batch_logjoint ≈ [
        logjoint_unconstrained(
            observed_positive_step,
            positive_batch_unconstrained[:, index],
            (),
            positive_batch_constraints[index],
        ) for
        index = 1:3
    ] atol=1e-8
    @test positive_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                observed_positive_step,
                positive_batch_unconstrained[:, index],
                (),
                positive_batch_constraints[index],
            ) for index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(positive_batch_gradient_cache.backend_cache)
    @test isnothing(positive_batch_gradient_cache.flat_cache)
    @test isempty(positive_batch_gradient_cache.column_caches)
    gaussian_combined_values = fill(-1.0, 3)
    gaussian_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        gaussian_combined_values,
        gaussian_batch_gradient_cache,
        gaussian_batch_params,
    )[2]
    @test gaussian_combined_values ≈ gaussian_batch_logjoint atol=1e-8
    @test gaussian_combined_gradient === gaussian_batch_gradient_cache.gradient_buffer
    @test gaussian_combined_gradient ≈ gaussian_batch_gradient atol=1e-8
    @test abs_scale_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                abs_scale_model,
                abs_scale_batch_params[:, index],
                (),
                abs_scale_batch_constraints[index],
            ) for
            index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(abs_scale_gradient_cache.backend_cache)
    @test isnothing(abs_scale_gradient_cache.flat_cache)
    @test isempty(abs_scale_gradient_cache.column_caches)
    abs_scale_combined_values = fill(-1.0, 3)
    abs_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        abs_scale_combined_values,
        abs_scale_gradient_cache,
        abs_scale_batch_params,
    )[2]
    @test abs_scale_combined_values ≈ [
        logjoint_unconstrained(abs_scale_model, abs_scale_batch_params[:, index], (), abs_scale_batch_constraints[index]) for
        index = 1:3
    ] atol=1e-8
    @test abs_scale_combined_gradient === abs_scale_gradient_cache.gradient_buffer
    @test abs_scale_combined_gradient ≈ abs_scale_gradient atol=1e-8
    @test power_scale_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                power_scale_model,
                power_scale_batch_params[:, index],
                (),
                power_scale_batch_constraints[index],
            ) for
            index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(power_scale_gradient_cache.backend_cache)
    @test isnothing(power_scale_gradient_cache.flat_cache)
    @test isempty(power_scale_gradient_cache.column_caches)
    @test min_scale_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                min_scale_model,
                min_scale_batch_params[:, index],
                (),
                min_scale_batch_constraints[index],
            ) for
            index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(min_scale_gradient_cache.backend_cache)
    @test isnothing(min_scale_gradient_cache.flat_cache)
    @test isempty(min_scale_gradient_cache.column_caches)
    min_scale_combined_values = fill(-1.0, 3)
    min_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        min_scale_combined_values,
        min_scale_gradient_cache,
        min_scale_batch_params,
    )[2]
    @test min_scale_combined_values ≈ [
        logjoint_unconstrained(min_scale_model, min_scale_batch_params[:, index], (), min_scale_batch_constraints[index]) for
        index = 1:3
    ] atol=1e-8
    @test min_scale_combined_gradient === min_scale_gradient_cache.gradient_buffer
    @test min_scale_combined_gradient ≈ min_scale_gradient atol=1e-8
    @test max_scale_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                max_scale_model,
                max_scale_batch_params[:, index],
                (),
                max_scale_batch_constraints[index],
            ) for
            index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(max_scale_gradient_cache.backend_cache)
    @test isnothing(max_scale_gradient_cache.flat_cache)
    @test isempty(max_scale_gradient_cache.column_caches)
    max_scale_combined_values = fill(-1.0, 3)
    max_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        max_scale_combined_values,
        max_scale_gradient_cache,
        max_scale_batch_params,
    )[2]
    @test max_scale_combined_values ≈ [
        logjoint_unconstrained(max_scale_model, max_scale_batch_params[:, index], (), max_scale_batch_constraints[index]) for
        index = 1:3
    ] atol=1e-8
    @test max_scale_combined_gradient === max_scale_gradient_cache.gradient_buffer
    @test max_scale_combined_gradient ≈ max_scale_gradient atol=1e-8
    @test mod_scale_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                mod_scale_model,
                mod_scale_batch_params[:, index],
                (),
                mod_scale_batch_constraints[index],
            ) for
            index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(mod_scale_gradient_cache.backend_cache)
    @test isnothing(mod_scale_gradient_cache.flat_cache)
    @test isempty(mod_scale_gradient_cache.column_caches)
    mod_scale_combined_values = fill(-1.0, 3)
    mod_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        mod_scale_combined_values,
        mod_scale_gradient_cache,
        mod_scale_batch_params,
    )[2]
    @test mod_scale_combined_values ≈ [
        logjoint_unconstrained(mod_scale_model, mod_scale_batch_params[:, index], (), mod_scale_batch_constraints[index]) for
        index = 1:3
    ] atol=1e-8
    @test mod_scale_combined_gradient === mod_scale_gradient_cache.gradient_buffer
    @test mod_scale_combined_gradient ≈ mod_scale_gradient atol=1e-8
    @test clamp_scale_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                clamp_scale_model,
                clamp_scale_batch_params[:, index],
                (),
                clamp_scale_batch_constraints[index],
            ) for
            index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(clamp_scale_gradient_cache.backend_cache)
    @test isnothing(clamp_scale_gradient_cache.flat_cache)
    @test isempty(clamp_scale_gradient_cache.column_caches)
    clamp_scale_combined_values = fill(-1.0, 3)
    clamp_scale_combined_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
        clamp_scale_combined_values,
        clamp_scale_gradient_cache,
        clamp_scale_batch_params,
    )[2]
    @test clamp_scale_combined_values ≈ [
        logjoint_unconstrained(clamp_scale_model, clamp_scale_batch_params[:, index], (), clamp_scale_batch_constraints[index]) for
        index = 1:3
    ] atol=1e-8
    @test clamp_scale_combined_gradient === clamp_scale_gradient_cache.gradient_buffer
    @test clamp_scale_combined_gradient ≈ clamp_scale_gradient atol=1e-8
    positive_workspace = UncertainTea.BatchedLogjointWorkspace(observed_positive_step, positive_batch_constraints)
    positive_workspace_values = UncertainTea._logjoint_unconstrained_batched_backend!(
        observed_positive_step,
        positive_workspace,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    )
    positive_workspace_constrained = positive_workspace.batched_constrained_buffer[]
    positive_workspace_logabsdet = positive_workspace.batched_logabsdet_buffer[]
    positive_workspace_observed = positive_workspace.batched_environment[].observed_values
    @test positive_workspace_values ≈ positive_batch_logjoint atol=1e-8
    @test UncertainTea._logjoint_unconstrained_batched_backend!(
        observed_positive_step,
        positive_workspace,
        positive_batch_unconstrained .+ 0.05,
        (),
        positive_batch_constraints,
    ) ≈ [
        logjoint_unconstrained(
            observed_positive_step,
            (positive_batch_unconstrained .+ 0.05)[:, index],
            (),
            positive_batch_constraints[index],
        ) for index = 1:3
    ] atol=1e-8
    @test positive_workspace.batched_constrained_buffer[] === positive_workspace_constrained
    @test positive_workspace.batched_logabsdet_buffer[] === positive_workspace_logabsdet
    @test positive_workspace.batched_environment[].observed_values === positive_workspace_observed
    positive_destination = fill(-1.0, 3)
    @test UncertainTea._batched_logjoint_unconstrained_with_workspace!(
        positive_destination,
        observed_positive_step,
        positive_workspace,
        positive_batch_unconstrained,
        (),
        positive_batch_constraints,
    ) === positive_destination
    @test positive_destination ≈ positive_batch_logjoint atol=1e-8
    positive_reconstrained = similar(positive_step_unconstrained)
    @test UncertainTea._transform_to_constrained!(
        positive_reconstrained,
        observed_positive_step,
        positive_step_unconstrained,
    ) === positive_reconstrained
    @test positive_reconstrained ≈ transform_to_constrained(
        observed_positive_step,
        positive_step_unconstrained,
    ) atol=1e-8
end

@testset "exception_free_batched_scoring" begin
    # Issue #98: a divergent trajectory that drives a log-transformed scale
    # latent to exp(u) == 0 must invalidate only its own column (scoring
    # non-finite, like the device path), not abort the whole batched run.
    @tea static function efs_hierarchical(n::Int)
        mu ~ normal(0.0, 3.0)
        s ~ gamma(2.0, 2.0)
        for i = 1:n
            {(:y, i)} ~ normal(mu, s)
        end
        return mu
    end
    efs_cm = choicemap((((:y, i), 0.1 * i) for i = 1:6)...)

    # The underflow column scores non-finite instead of throwing; this matches
    # the device reference (device_batched_logjoint returns [NaN] for the same
    # input) and what the leapfrog isfinite masking already assumes.
    efs_bad = batched_logjoint_unconstrained(efs_hierarchical, reshape([0.0, -800.0], 2, 1), (6,), efs_cm)
    @test length(efs_bad) == 1
    @test !isfinite(efs_bad[1])
    @test efs_bad ≈ UncertainTea.device_batched_logjoint(
        efs_hierarchical, reshape([0.0, -800.0], 2, 1), (6,), efs_cm,
    ) nans = true

    # A valid column sharing the batch with a bad one is scored correctly: it
    # matches the same column computed on its own (and the per-column scalar
    # reference), while the bad column stays non-finite.
    efs_mixed = batched_logjoint_unconstrained(
        efs_hierarchical, [0.0 0.0; 0.5 -800.0], (6,), efs_cm,
    )
    efs_good_alone = batched_logjoint_unconstrained(
        efs_hierarchical, reshape([0.0, 0.5], 2, 1), (6,), efs_cm,
    )
    @test isfinite(efs_mixed[1])
    @test efs_mixed[1] ≈ efs_good_alone[1] atol = 1e-10
    @test efs_mixed[1] ≈
          logjoint_unconstrained(efs_hierarchical, [0.0, 0.5], (6,), efs_cm) atol = 1e-10
    @test !isfinite(efs_mixed[2])

    # The analytic batched gradient degrades the same way: the bad column's
    # gradient is non-finite while the good column matches its own gradient.
    efs_grad = batched_logjoint_gradient_unconstrained(
        efs_hierarchical, [0.0 0.0; 0.5 -800.0], (6,), efs_cm,
    )
    efs_good_grad = batched_logjoint_gradient_unconstrained(
        efs_hierarchical, reshape([0.0, 0.5], 2, 1), (6,), efs_cm,
    )
    @test all(isfinite, view(efs_grad, :, 1))
    @test view(efs_grad, :, 1) ≈ view(efs_good_grad, :, 1) atol = 1e-10
    @test !all(isfinite, view(efs_grad, :, 2))

    # Fixed-step batched HMC that previously crashed on every step/seed combo
    # (a divergent chain reaching the underflow boundary) now completes, with
    # divergent chains handled per column.
    for seed = 1:5, step in (0.6, 0.9, 1.2)
        chains = batched_hmc(
            efs_hierarchical, (6,), efs_cm;
            num_chains=8, num_samples=60, num_warmup=0, step_size=step,
            num_leapfrog_steps=10, adapt_step_size=false,
            adapt_mass_matrix=false, find_reasonable_step_size=false,
            initial_params=fill(0.1, 2), rng=MersenneTwister(seed),
        )
        @test length(chains.chains) == 8
        @test all(chain -> all(isfinite, chain.constrained_samples), chains.chains)
    end
end

@testset "batched_logjoint_integer_matrix" begin
    # Issue #92: an integer-typed constrained matrix must be promoted to float
    # up front so the backend totals are float; otherwise scoring hits an
    # InexactError. The promoted result matches the float-matrix result.
    @tea static function blim_model()
        mu ~ normal(0.0, 1.0)
        {:y} ~ normal(mu, 1.0)
    end
    blim_constraints = [choicemap((:y, 0.0)), choicemap((:y, 1.0))]
    blim_int = batched_logjoint(blim_model, reshape([0, 1], 1, 2), (), blim_constraints)
    blim_float = batched_logjoint(blim_model, reshape([0.0, 1.0], 1, 2), (), blim_constraints)
    @test eltype(blim_int) <: AbstractFloat
    @test blim_int ≈ blim_float atol = 1e-12
end
