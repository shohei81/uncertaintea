using Test
using Random
using UncertainTea

@testset "UncertainTea" begin
    include("uncertaintea/core.jl")
    if get(ENV, "UNCERTAINTEA_TEST_GROUP", "all") in ("all", "sampling")
        include("uncertaintea/sampling.jl")
    end
end
