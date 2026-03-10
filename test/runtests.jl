using Test
using Random
using UncertainTea

@testset "UncertainTea" begin
    @tea static function gaussian_mean()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 1.0f0)
        return mu
    end

    constraints = choicemap((:y, 0.3f0))
    trace, logw = generate(gaussian_mean, (), constraints; rng=MersenneTwister(1))
    spec = modelspec(gaussian_mean)

    @test trace.model === gaussian_mean
    @test haskey(trace.choices, :mu)
    @test trace[:y] == 0.3f0
    @test logw == trace.log_weight
    @test isfinite(logw)
    @test trace.retval == trace[:mu]
    @test spec.name == :gaussian_mean
    @test spec.mode == :static
    @test spec.arguments == Symbol[]
    @test length(spec.choices) == 2
    @test spec.choices[1].binding == :mu
    @test isstaticaddress(spec.choices[1].address)
    @test isstaticaddress(spec.choices[2].address)

    @tea static function iid_model(n)
        mu ~ normal(0.0f0, 1.0f0)
        for i in 1:n
            {:y => i} ~ normal(mu, 1.0f0)
        end
        return mu
    end

    ys = Float32[0.1f0, -0.2f0, 0.4f0]
    repeated = choicemap((:y => i, ys[i]) for i in eachindex(ys))
    trace2, logw2 = generate(iid_model, (length(ys),), repeated; rng=MersenneTwister(2))
    spec2 = modelspec(iid_model)

    @test trace2[:y => 1] == ys[1]
    @test trace2[:y => 3] == ys[3]
    @test isfinite(logw2)
    @test spec2.arguments == [:n]
    @test length(spec2.choices) == 2
    @test spec2.shape_specialized
    @test isaddresstemplate(spec2.choices[2].address)

    @tea static function step(prev)
        z ~ normal(prev, 1.0f0)
        return z
    end

    @tea static function chain_model(T)
        z = ({:z => 1} ~ step(0.0f0))
        for t in 2:T
            z = ({:z => t} ~ step(z))
        end
        return z
    end

    trace3, _ = generate(chain_model, (3,), choicemap(); rng=MersenneTwister(3))
    spec3 = modelspec(chain_model)

    @test haskey(trace3.choices, :z => 1 => :z)
    @test haskey(trace3.choices, :z => 3 => :z)
    @test length(spec3.choices) == 2
    @test isstaticaddress(spec3.choices[1].address)
    @test isaddresstemplate(spec3.choices[2].address)
end
