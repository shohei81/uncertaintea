# Issue #73: distribution builders normalize integer (and other non-float real)
# parameters to floats, so integer literals sample correctly everywhere instead
# of hitting MethodError/InexactError or corrupting the categorical sampler.

@testset "dist_integer_params_sampling" begin
    @test rand(MersenneTwister(1), normal(0, 1)) ==
          rand(MersenneTwister(1), normal(0.0, 1.0))
    @test rand(MersenneTwister(1), lognormal(0, 1)) ==
          rand(MersenneTwister(1), lognormal(0.0, 1.0))

    int_dirichlet_draw = rand(MersenneTwister(1), dirichlet(1, 1))
    @test int_dirichlet_draw isa Vector{Float64}
    @test sum(int_dirichlet_draw) ≈ 1.0 atol = 1e-12

    # categorical(0, 1) puts all mass on category 2; the integer-typed sampler
    # used to draw random Int64 thresholds and return ~50/50 categories.
    int_cat_rng = MersenneTwister(1)
    @test all(rand(int_cat_rng, categorical(0, 1)) == 2 for _ = 1:1_000)

    # generate on a model with integer literal parameters works end to end
    @tea static function int_param_model()
        x ~ normal(0, 1)
        return x
    end
    int_param_trace, _ = generate(int_param_model; rng=MersenneTwister(1))
    @test int_param_trace[:x] isa Float64
    @test int_param_trace[:x] == rand(MersenneTwister(1), normal(0.0, 1.0))
end

@testset "dist_integer_params_float_storage" begin
    @test normal(0, 1).sigma === 1.0
    @test lognormal(0, 1).sigma === 1.0
    @test laplace(0, 1).scale === 1.0
    @test exponential(2).rate === 2.0
    @test gamma(2, 3).rate === 3.0
    @test inversegamma(2, 3).scale === 3.0
    @test weibull(1, 2).scale === 2.0
    @test beta(2, 2).alpha === 2.0
    @test studentt(3, 0, 1).sigma === 1.0
    @test bernoulli(1).p === 1.0
    @test geometric(1).p === 1.0
    @test UncertainTea.binomial(4, 1).p === 1.0
    @test negativebinomial(2, 1).successes === 2.0
    @test poisson(3).lambda === 3.0
    @test categorical(0, 1).probabilities == [0.0, 1.0]
    @test eltype(categorical(0, 1).probabilities) === Float64
    @test dirichlet(1, 2).alpha == [1.0, 2.0]
    @test eltype(dirichlet(1, 2).alpha) === Float64
end
