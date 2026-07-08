using Test
using Random
using UncertainTea

@testset "UncertainTea" begin
    include("uncertaintea/fixtures.jl")
    include("uncertaintea/core.jl")
    include("uncertaintea/sampling.jl")
end
