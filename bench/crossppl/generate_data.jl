# Regenerates the synthetic shared datasets (logistic.json, linreg.json).
# The JSON files are checked in; this script only documents their provenance.
# Run from bench/crossppl: julia --project=julia generate_data.jl
using Random
using JSON3

const SEED = 20260723

function write_json(path, obj)
    open(path, "w") do io
        JSON3.pretty(io, obj)
        println(io)
    end
    println("wrote ", path)
end

function logistic_data(rng)
    n, d = 500, 8
    X = randn(rng, d, n)
    alpha = 0.3
    beta = [0.8, -1.2, 0.5, 0.0, 1.5, -0.4, 0.9, -0.7]
    eta = alpha .+ X' * beta
    y = Int.(rand(rng, n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
    return Dict(
        "model" => "logistic",
        "seed" => SEED,
        "n" => n,
        "d" => d,
        "true_alpha" => alpha,
        "true_beta" => beta,
        "X" => [X[:, i] for i = 1:n],
        "y" => y,
    )
end

function gauss_data(rng)
    n = 1000
    mu, s = 0.5, 1.2
    ys = mu .+ s .* randn(rng, n)
    return Dict(
        "model" => "gauss",
        "seed" => SEED,
        "n" => n,
        "true_mu" => mu,
        "true_s" => s,
        "y" => ys,
    )
end

rng = MersenneTwister(SEED)
dir = joinpath(@__DIR__, "data")
write_json(joinpath(dir, "logistic.json"), logistic_data(rng))
write_json(joinpath(dir, "gauss.json"), gauss_data(rng))
