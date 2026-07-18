# MH-within-Gibbs (docs/mh-within-gibbs.md, issue #13 track 2): symmetric
# single-site MH on discrete latents alternating with NUTS on the continuous
# block. Track 1's enumeration provides the mixed-model oracle; pure-discrete
# posteriors are checked against direct truncated enumeration.

# Local mean helper (Statistics is not imported by the test harness).
gibbs_mean(x) = sum(x) / length(x)

# pure discrete: no continuous slots at all
@tea static function gibbs_pure_poisson_model()
    z ~ poisson(3.0)
    {:y} ~ normal(1.0 * z, 0.5)
    return z
end

# the indicator mixture in its unmarginalized (Gibbs) spelling ...
@tea static function gibbs_indicator_model()
    m1 ~ normal(-2.0, 1.0)
    m2 ~ normal(2.0, 1.0)
    z ~ bernoulli(0.3)
    {:y} ~ normal(z * m1 + (1 - z) * m2, 0.5)
    return m1
end

# ... and its track-1 enumerated spelling: the acceptance oracle
@tea static function gibbs_indicator_enum_model()
    m1 ~ normal(-2.0, 1.0)
    m2 ~ normal(2.0, 1.0)
    z ~ bernoulli(0.3; marginalize=:enumerate)
    {:y} ~ normal(z * m1 + (1 - z) * m2, 0.5)
    return m1
end

# loop-scoped bernoulli sites sharing one template spec
@tea static function gibbs_loop_model(n)
    p ~ beta(2.0, 2.0)
    for i = 1:n
        {:z => i} ~ bernoulli(p)
    end
    {:y} ~ normal(10.0 * p, 1.0)
    return p
end

# a binomial site whose walk proposals can leave the dynamic support: the
# suffix indexes ps[z + 1], so z = 3 throws and must reject, not crash
@tea static function gibbs_dynamic_bound_model(ps)
    q ~ beta(2.0, 2.0)
    z ~ binomial(2, q)
    {:y} ~ normal(ps[z+1], 1.0)
    return q
end

# a marginalized site must stay OUT of the Gibbs site set
@tea static function gibbs_mixed_marginalized_model()
    mu ~ normal(0.0, 1.0)
    m ~ bernoulli(0.4; marginalize=:enumerate)
    z ~ poisson(2.0)
    {:y} ~ normal(mu + m + z, 0.8)
    return mu
end

# a slotless CONTINUOUS latent has no sampler and must error clearly
@tea static function gibbs_continuous_slotless_model(n)
    for i = 1:n
        {:x => i} ~ normal(0.0, 1.0)
    end
    {:y} ~ normal(1.0, 1.0)
    return n
end

# a discrete site inside a SUBMODEL: the trace address is prefixed
# ((:sub, :z)) and only the inlined execution plan carries its spec
@tea static function gibbs_count_child()
    z ~ poisson(2.0)
    return z
end

@tea static function gibbs_nested_model()
    mu ~ normal(0.0, 1.0)
    z = ({:sub} ~ gibbs_count_child())
    {:y} ~ normal(mu + z, 0.8)
    return mu
end

# a marginalized site with a DYNAMIC (non-loop) address: the exclusion must
# match by template, not by forcing a static address
@tea static function gibbs_dynamic_marginalized_model(k)
    mu ~ normal(0.0, 1.0)
    {:m => k} ~ bernoulli(0.4; marginalize=:enumerate)
    z ~ poisson(2.0)
    {:y} ~ normal(mu + z, 0.8)
    return mu
end

@testset "mh_within_gibbs" begin
    @testset "gibbs_pure_discrete_vs_enumeration" begin
        # posterior over z by direct truncated enumeration of the support
        gibbs_pure_lp = [
            UncertainTea.logpdf(poisson(3.0), k) + UncertainTea.logpdf(normal(1.0 * k, 0.5), 4.2) for
            k = 0:60
        ]
        gibbs_pure_weights = exp.(gibbs_pure_lp .- maximum(gibbs_pure_lp))
        gibbs_pure_weights ./= sum(gibbs_pure_weights)
        gibbs_pure_oracle = sum((0:60) .* gibbs_pure_weights)

        gibbs_pure_draws = Float64[]
        for gibbs_pure_seed = 1:2
            gibbs_pure_chain = gibbs(
                gibbs_pure_poisson_model,
                (),
                choicemap((:y, 4.2));
                num_samples=3000,
                num_warmup=500,
                rng=MersenneTwister(40 + gibbs_pure_seed),
            )
            @test isempty(gibbs_pure_chain.mass_matrix)
            @test size(gibbs_pure_chain.constrained_samples, 1) == 0
            append!(gibbs_pure_draws, gibbs_pure_chain.discrete_samples[1, :])
        end
        @test gibbs_mean(gibbs_pure_draws) ≈ gibbs_pure_oracle atol = 0.15
    end

    @testset "gibbs_vs_enumerated_nuts" begin
        gibbs_ind_constraints = choicemap((:y, 0.8))
        gibbs_ind_m1 = Float64[]
        gibbs_ind_m2 = Float64[]
        gibbs_enum_m1 = Float64[]
        gibbs_enum_m2 = Float64[]
        gibbs_ind_z = Float64[]
        for gibbs_ind_seed = 1:3
            gibbs_ind_chain = gibbs(
                gibbs_indicator_model,
                (),
                gibbs_ind_constraints;
                num_samples=600,
                num_warmup=300,
                rng=MersenneTwister(100 + gibbs_ind_seed),
            )
            gibbs_enum_chain = nuts(
                gibbs_indicator_enum_model,
                (),
                gibbs_ind_constraints;
                num_samples=600,
                num_warmup=300,
                rng=MersenneTwister(200 + gibbs_ind_seed),
            )
            @test all(isfinite, gibbs_ind_chain.constrained_samples)
            @test gibbs_ind_chain.discrete_addresses == Any[(:z,)]
            append!(gibbs_ind_m1, gibbs_ind_chain.constrained_samples[1, :])
            append!(gibbs_ind_m2, gibbs_ind_chain.constrained_samples[2, :])
            append!(gibbs_enum_m1, gibbs_enum_chain.constrained_samples[1, :])
            append!(gibbs_enum_m2, gibbs_enum_chain.constrained_samples[2, :])
            append!(gibbs_ind_z, gibbs_ind_chain.discrete_samples[1, :])
        end
        @test gibbs_mean(gibbs_ind_m1) ≈ gibbs_mean(gibbs_enum_m1) atol = 0.2
        @test gibbs_mean(gibbs_ind_m2) ≈ gibbs_mean(gibbs_enum_m2) atol = 0.2
        @test 0.0 < gibbs_mean(gibbs_ind_z) < 1.0
    end

    @testset "gibbs_loop_sites_and_determinism" begin
        gibbs_loop_chain = gibbs(
            gibbs_loop_model,
            (3,),
            choicemap((:y, 7.0));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(5),
        )
        @test gibbs_loop_chain.discrete_addresses == Any[(:z, 1), (:z, 2), (:z, 3)]
        @test all(isfinite, gibbs_loop_chain.constrained_samples)
        @test all(value -> value in (0, 1), gibbs_loop_chain.discrete_samples)

        gibbs_loop_replay = gibbs(
            gibbs_loop_model,
            (3,),
            choicemap((:y, 7.0));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(5),
        )
        @test gibbs_loop_replay.constrained_samples == gibbs_loop_chain.constrained_samples
        @test gibbs_loop_replay.discrete_samples == gibbs_loop_chain.discrete_samples
    end

    @testset "gibbs_dynamic_bound_rejection" begin
        gibbs_bound_chain = gibbs(
            gibbs_dynamic_bound_model,
            ([0.0, 1.0, 2.0],),
            choicemap((:y, 1.4));
            num_samples=800,
            num_warmup=200,
            rng=MersenneTwister(7),
        )
        @test all(isfinite, gibbs_bound_chain.logjoint_values)
        @test minimum(gibbs_bound_chain.discrete_samples) >= 0
        @test maximum(gibbs_bound_chain.discrete_samples) <= 2
    end

    @testset "gibbs_site_discovery" begin
        # marginalized sites stay marginalized; only the poisson site remains
        gibbs_mixed_chain = gibbs(
            gibbs_mixed_marginalized_model,
            (),
            choicemap((:y, 2.4));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(9),
        )
        @test gibbs_mixed_chain.discrete_addresses == Any[(:z,)]
        @test all(isfinite, gibbs_mixed_chain.logjoint_values)

        # no discrete sites: degrades to plain NUTS semantics
        gibbs_none_chain = gibbs(
            gibbs_indicator_enum_model,
            (),
            choicemap((:y, 0.8));
            num_samples=100,
            num_warmup=100,
            rng=MersenneTwister(13),
        )
        @test isempty(gibbs_none_chain.discrete_addresses)
        @test all(isfinite, gibbs_none_chain.constrained_samples)

        # slotless continuous latents error with a clear message
        @test_throws ArgumentError gibbs(
            gibbs_continuous_slotless_model,
            (2,),
            choicemap((:y, 0.5));
            num_samples=10,
            rng=MersenneTwister(3),
        )

        # a submodel latent matches through the inlined plan's prefixed address
        gibbs_nested_chain = gibbs(
            gibbs_nested_model,
            (),
            choicemap((:y, 3.1));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(17),
        )
        @test gibbs_nested_chain.discrete_addresses == Any[(:sub, :z)]
        @test all(isfinite, gibbs_nested_chain.logjoint_values)

        # a marginalized site with a dynamic address stays excluded by
        # template matching (no static-address requirement)
        gibbs_dynamic_marg_chain = gibbs(
            gibbs_dynamic_marginalized_model,
            (7,),
            choicemap((:y, 2.4));
            num_samples=100,
            num_warmup=100,
            rng=MersenneTwister(19),
        )
        @test gibbs_dynamic_marg_chain.discrete_addresses == Any[(:z,)]
        @test all(isfinite, gibbs_dynamic_marg_chain.logjoint_values)
    end
end
