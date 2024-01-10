using Test

# TODO Find decent property-based testing framework for Julia

include("ga.jl")
include("minmaxscaler.jl")
include("MLJInterface.jl")
include("intervals.jl")

using RSLModels.Models
using Random

dim = 3
model = draw_model(dim)

N = 100
K = length(model.conditions)
X = rand(N, dim)

@test size(output(model, X)) == (N,)
@test size(output_mean(model, X)) == (N,)

M = Models.match(model, X)
@test size(output(model, X; matching_matrix=M)) == (N,)
@test size(output_mean(model, X; matching_matrix=M)) == (N,)

@test all(output_variance(model, X) .>= 0)

# Check determinism.

seed = 1123
y1 = output(Random.Xoshiro(seed), model, X)
y2 = output(Random.Xoshiro(seed), model, X)
@test all(y1 .== y2)
