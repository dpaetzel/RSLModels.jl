using Distributions
using LinearAlgebra
using Random
using StatsBase
using NPZ
using Serialization

# include("Intervals.jl")
# include("LocalModels.jl")
# using .Intervals
# using .LocalModels
include("Tasks.jl")
using .Tasks

# We want to track global assignments as well when doing
# `includet(thisfile.jl)`.
__revise_mode__ = :evalassign

task = generate(5, 10, 1000; seed=1337)

# TODO Extract plot_mapping and related stuff from Intervals to Score module
# TODO Include Score module here and add training data points to plots
# TODO Compute min/max variants of an interval given input data X
# TODO Compute min/max variants of many intervals given input data X
# TODO Compute min/max variants of many intervals given input data X

# TODO Consider to enforce that each interval is sampled k times
