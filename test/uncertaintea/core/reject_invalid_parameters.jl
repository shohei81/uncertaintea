# Issue #157: Stan-style reject semantics for sampling-context evaluation.
#
# A distribution constructor throwing on invalid parameters (e.g. `normal`
# with an exp-underflowed sigma == 0 reached mid-trajectory) used to propagate
# out of the batched gradient and kill a whole many-chain run. In reject mode
# (`reject_invalid_parameters=true`, enabled by the sampler-owned evaluation
# caches/workspaces) such parameters score a non-finite log-density for that
# chain/lane only, which the leapfrog guards turn into a per-chain divergence.
# The public logjoint/gradient APIs keep their throwing contract.

@testset "reject_invalid_parameters" begin
    @tea static function rip_scale_model()
        x ~ normal(0.0, 1.0)
        {:y} ~ normal(0.0, exp(x))
        return x
    end

    rip_cm = choicemap((:y, 0.3))
    # x = -800 makes exp(x) underflow to exactly 0.0: invalid sigma.
    rip_bad = [-800.0]
    rip_good = [0.2]

    @testset "rip_public_apis_still_throw" begin
        @test_throws ArgumentError logjoint_unconstrained(rip_scale_model, rip_bad, (), rip_cm)
        @test_throws ArgumentError logjoint(rip_scale_model, rip_bad, (), rip_cm)
    end

    @testset "rip_reject_mode_scores_minus_inf" begin
        @test logjoint_unconstrained(
            rip_scale_model, rip_bad, (), rip_cm; reject_invalid_parameters=true,
        ) == -Inf
        # valid positions are untouched by reject mode (bit-identical value)
        @test logjoint_unconstrained(
            rip_scale_model, rip_good, (), rip_cm; reject_invalid_parameters=true,
        ) == logjoint_unconstrained(rip_scale_model, rip_good, (), rip_cm)
    end

    @testset "rip_batched_gradient_poisons_only_the_bad_lane" begin
        rip_params = [0.2 -800.0]
        rip_cache = UncertainTea.BatchedLogjointGradientCache(
            rip_scale_model, rip_params, (), rip_cm; reject_invalid_parameters=true,
        )
        rip_values = Vector{Float64}(undef, 2)
        rip_values, rip_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
            rip_values, rip_cache, rip_params,
        )
        # the invalid lane is non-finite (rejected by the integrator guards);
        # the healthy lane keeps its exact value and a finite gradient
        @test !isfinite(rip_values[2])
        @test rip_values[1] == logjoint_unconstrained(rip_scale_model, rip_good, (), rip_cm)
        @test all(isfinite, rip_gradient[:, 1])
    end

    # Eight-schools-noncentered shape (issue #157 repro): one lane's
    # unconstrained tau under exp-underflow feeds `iid(normal(mu, 0.0), ...)`.
    @tea static function rip_eight_schools_nc(sigma)
        mu ~ normal(0.0, 5.0)
        tau ~ truncatedstudentt(1.0, 0.0, 5.0, 0.0, Inf)
        theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
        for i = 1:8
            {:y => i} ~ normal(theta[i], sigma[i])
        end
        return mu
    end

    rip_sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    rip_y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    rip_nc_cm = choicemap(((:y => i, rip_y[i]) for i = 1:8)...)

    @testset "rip_noncentered_tau_underflow_unit" begin
        rip_nc_bad = [0.5, -800.0, 0.1, 0.2, -0.1, 0.3, 0.0, -0.2, 0.4, 0.1]
        @test_throws ArgumentError logjoint_unconstrained(
            rip_eight_schools_nc, rip_nc_bad, (rip_sigma,), rip_nc_cm,
        )
        @test logjoint_unconstrained(
            rip_eight_schools_nc, rip_nc_bad, (rip_sigma,), rip_nc_cm;
            reject_invalid_parameters=true,
        ) == -Inf

        rip_nc_good = [0.5, 0.0, 0.1, 0.2, -0.1, 0.3, 0.0, -0.2, 0.4, 0.1]
        rip_nc_params = hcat(rip_nc_good, rip_nc_bad)
        rip_nc_cache = UncertainTea.BatchedLogjointGradientCache(
            rip_eight_schools_nc, rip_nc_params, (rip_sigma,), rip_nc_cm;
            reject_invalid_parameters=true,
        )
        rip_nc_values = Vector{Float64}(undef, 2)
        rip_nc_values, rip_nc_gradient = UncertainTea._batched_logjoint_and_gradient_unconstrained!(
            rip_nc_values, rip_nc_cache, rip_nc_params,
        )
        @test !isfinite(rip_nc_values[2])
        @test rip_nc_values[1] ≈ logjoint_unconstrained(
            rip_eight_schools_nc, rip_nc_good, (rip_sigma,), rip_nc_cm,
        ) atol = 1e-10
        @test all(isfinite, rip_nc_gradient[:, 1])

        # the throwing default is preserved for the public cache constructor:
        # without reject mode the invalid lane's constructor error propagates
        # (the exact #157 crash mechanism)
        rip_nc_throwing_cache = UncertainTea.BatchedLogjointGradientCache(
            rip_eight_schools_nc, rip_nc_params, (rip_sigma,), rip_nc_cm,
        )
        @test_throws ArgumentError UncertainTea._batched_logjoint_and_gradient_unconstrained!(
            Vector{Float64}(undef, 2), rip_nc_throwing_cache, rip_nc_params,
        )
    end

    # Regression for the #157 crash: on the pre-fix code these exact
    # (strategy, seed) combinations abort with `ArgumentError: normal requires
    # sigma > 0` out of the full-width batched gradient during warmup. Shared
    # adaptation is pinned because it is the crashing configuration (issue
    # #137 made per-chain the host default, which reduces exposure).
    @testset "rip_batched_nuts_completes_at_64_chains" begin
        for (rip_strategy, rip_seed) in ((:masked, 7), (:hybrid, 7))
            rip_chains = batched_nuts(
                rip_eight_schools_nc, (rip_sigma,), rip_nc_cm;
                num_chains=64, num_warmup=200, num_samples=10,
                tree_strategy=rip_strategy, per_chain_adaptation=false,
                rng=MersenneTwister(rip_seed),
            )
            @test nchains(rip_chains) == 64
            @test numsamples(rip_chains) == 10
            # the lanes that used to crash the run now register divergences
            @test sum(sum(rip_chain.divergent) for rip_chain in rip_chains) > 0
            @test all(all(isfinite, rip_chain.logjoint_values) for rip_chain in rip_chains)
        end
    end
end
