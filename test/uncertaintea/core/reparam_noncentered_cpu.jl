# CPU semantics of reparam=:noncentered (PR-3 of docs/noncentered-reparam.md):
# the slot holds the standardized z, the trace/constrained space keeps theta,
# and the plan-walk transform materializes theta = loc + scale * z with the
# log|scale| change-of-variables term.

# weak observation (sigma = 10): the classic setting where the centered
# funnel defeats NUTS while the non-centered form mixes cleanly
@tea static function ncc_funnel_centered()
    v ~ normal(0.0, 3.0)
    x ~ normal(0.0, exp(v / 2))
    {:y} ~ normal(x, 10.0)
    return v
end

@tea static function ncc_funnel_flagged()
    v ~ normal(0.0, 3.0)
    x ~ normal(0.0, exp(v / 2); reparam=:noncentered)
    {:y} ~ normal(x, 10.0)
    return v
end

@tea static function ncc_funnel_manual()
    v ~ normal(0.0, 3.0)
    z ~ normal(0.0, 1.0)
    x = exp(v / 2) * z
    {:y} ~ normal(x, 10.0)
    return v
end

@tea static function ncc_arg_model(location_offset)
    tau ~ lognormal(0.0, 0.5)
    theta ~ normal(location_offset, tau; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

# a bound discrete latent has no parameter slot, so the walk cannot know its
# value: a location depending on it must be rejected
@tea static function ncc_slotless_dependent_model()
    k ~ bernoulli(0.5)
    theta ~ normal(k * 2.0, 1.0; reparam=:noncentered)
    {:y} ~ normal(theta, 1.0)
    return theta
end

# the codex-review case: the slotless dependency flows through a comparison
@tea static function ncc_slotless_branch_model()
    k ~ bernoulli(0.5)
    theta ~ normal(ifelse(k > 0, 2.0, 0.0), 1.0; reparam=:noncentered)
    {:y} ~ normal(theta, 1.0)
    return theta
end

@tea static function ncc_studentt_model()
    mu ~ normal(0.0, 1.0)
    theta ~ studentt(4.0, mu, 2.0; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

# well-identified hierarchical normal for SBC (the weak funnel can push v so
# far negative during step-size exploration that exp(v/2) underflows to zero,
# a pre-existing robustness limit of scale expressions)
@tea static function ncc_sbc_model()
    mu ~ normal(0.0, 1.0)
    tau ~ lognormal(0.0, 0.5)
    theta ~ normal(mu, tau; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

@tea static function ncc_laplace_model()
    mu ~ normal(0.0, 1.0)
    theta ~ laplace(mu, 1.5; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

@testset "reparam_noncentered_cpu" begin
    ncc_constraints = choicemap((:y, 0.5))

    @testset "ncc_transform_round_trip_and_density" begin
        z = [0.7, -1.2]
        constrained = transform_to_constrained(ncc_funnel_flagged, z)
        @test constrained[1] == z[1]
        @test constrained[2] ≈ exp(z[1] / 2) * z[2] atol = 1e-12
        @test transform_to_unconstrained(ncc_funnel_flagged, constrained) ≈ z atol = 1e-12

        # z-space density: N(v; 0, 3) + N(z; 0, 1) + likelihood at theta
        x_value = exp(z[1] / 2) * z[2]
        manual =
            UncertainTea.logpdf(normal(0.0, 3.0), z[1]) +
            UncertainTea.logpdf(normal(0.0, 1.0), z[2]) +
            UncertainTea.logpdf(normal(x_value, 10.0), 0.5)
        @test logjoint_unconstrained(ncc_funnel_flagged, z, (), ncc_constraints) ≈ manual atol = 1e-10

        gradient = logjoint_gradient_unconstrained(ncc_funnel_flagged, z, (), ncc_constraints)
        for i in eachindex(z)
            h = cbrt(eps(Float64))
            up = copy(z)
            up[i] += h
            down = copy(z)
            down[i] -= h
            fd =
                (
                    logjoint_unconstrained(ncc_funnel_flagged, up, (), ncc_constraints) -
                    logjoint_unconstrained(ncc_funnel_flagged, down, (), ncc_constraints)
                ) / (2h)
            @test gradient[i] ≈ fd atol = 5e-6
        end

        # trace round trip: traces keep theta on the constrained scale
        trace, _ = generate(ncc_funnel_flagged, (), ncc_constraints; rng=MersenneTwister(3))
        unconstrained = transform_to_unconstrained(trace)
        @test transform_to_constrained(ncc_funnel_flagged, unconstrained) ≈ parameter_vector(trace) atol = 1e-12
    end

    @testset "ncc_argument_dependent_location" begin
        z = [0.3, 0.9]
        constrained = transform_to_constrained(ncc_arg_model, z, (2.0,))
        @test constrained[2] ≈ 2.0 + exp(z[1]) * z[2] atol = 1e-12
        @test transform_to_unconstrained(ncc_arg_model, constrained, (2.0,)) ≈ z atol = 1e-12
        # the walk needs the model arguments
        @test_throws DimensionMismatch transform_to_constrained(ncc_arg_model, z)
        chain = nuts(
            ncc_arg_model,
            (2.0,),
            ncc_constraints;
            num_samples=30,
            num_warmup=30,
            rng=MersenneTwister(4),
        )
        @test all(isfinite, chain.constrained_samples)
    end

    @testset "ncc_slotless_dependent_location_rejected" begin
        @test_throws ArgumentError transform_to_constrained(ncc_slotless_dependent_model, [0.5])
        # comparisons and branching must not swallow the poison (NaN > 0 would
        # silently be false); the sentinel throws instead
        @test_throws ArgumentError transform_to_constrained(ncc_slotless_branch_model, [0.5])
    end

    @testset "ncc_studentt_laplace_density" begin
        z = [0.4, -0.8]
        observation = choicemap((:y, 0.4))
        # noncentered studentt: N(z2; 0, 1)-analog is the standardized studentt
        studentt_theta = z[1] + 2.0 * z[2]
        studentt_manual =
            UncertainTea.logpdf(normal(0.0, 1.0), z[1]) +
            UncertainTea.logpdf(studentt(4.0, 0.0, 1.0), z[2]) +
            UncertainTea.logpdf(normal(studentt_theta, 0.5), 0.4)
        @test logjoint_unconstrained(ncc_studentt_model, z, (), observation) ≈ studentt_manual atol = 1e-10

        laplace_theta = z[1] + 1.5 * z[2]
        laplace_manual =
            UncertainTea.logpdf(normal(0.0, 1.0), z[1]) +
            UncertainTea.logpdf(laplace(0.0, 1.0), z[2]) +
            UncertainTea.logpdf(normal(laplace_theta, 0.5), 0.4)
        @test logjoint_unconstrained(ncc_laplace_model, z, (), observation) ≈ laplace_manual atol = 1e-10
    end

    @testset "ncc_funnel_acceptance" begin
        total_centered = 0.0
        total_flagged = 0.0
        flagged_vs = Float64[]
        manual_vs = Float64[]
        for chain_seed = 1:6
            centered = nuts(
                ncc_funnel_centered,
                (),
                ncc_constraints;
                num_samples=300,
                num_warmup=200,
                rng=MersenneTwister(600 + chain_seed),
            )
            flagged = nuts(
                ncc_funnel_flagged,
                (),
                ncc_constraints;
                num_samples=300,
                num_warmup=200,
                rng=MersenneTwister(600 + chain_seed),
            )
            manual = nuts(
                ncc_funnel_manual,
                (),
                ncc_constraints;
                num_samples=300,
                num_warmup=200,
                rng=MersenneTwister(600 + chain_seed),
            )
            total_centered += divergencerate(centered)
            total_flagged += divergencerate(flagged)
            append!(flagged_vs, flagged.constrained_samples[1, :])
            append!(manual_vs, manual.constrained_samples[1, :])
        end
        # calibrated on both Julia versions: centered ~0.4 total divergence and
        # mean ~-2.2 (stuck in the neck); flagged ~0.02 and unbiased
        @test total_flagged < total_centered / 4
        @test total_flagged / 6 < 0.02

        # (the centered chains are also badly biased into the neck on most
        # streams -- mean ~ -2.2 vs ~ -0.2 -- but with 6 chains that signal is
        # stream-dependent, so only the divergence and agreement margins are
        # asserted)
        flagged_mean = sum(flagged_vs) / length(flagged_vs)
        manual_mean = sum(manual_vs) / length(manual_vs)
        flagged_sd = sqrt(sum(abs2, flagged_vs .- flagged_mean) / (length(flagged_vs) - 1))
        manual_sd = sqrt(sum(abs2, manual_vs .- manual_mean) / (length(manual_vs) - 1))
        @test abs(flagged_mean - manual_mean) < 0.5
        @test 0.8 < flagged_sd / manual_sd < 1.3
    end

    @testset "ncc_sbc" begin
        result = sbc(
            ncc_sbc_model;
            num_simulations=40,
            num_posterior_draws=16,
            num_warmup=60,
            rng=MersenneTwister(9),
        )
        @test all(result.pvalues .> 0.005)
        @test !has_warnings(result)
    end
end
