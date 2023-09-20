# We want to track global assignments as well when doing
# `includet(thisfile.jl)`.
__revise_mode__ = :evalassign

using Infiltrator
using LinearAlgebra
using PyPlot
using RSLModels.Intervals
using RSLModels.Scores
using RSLModels.Tasks
using Random
using Statistics

dims = 2
spread_min = 0.05
volume_min = 0.1

function plot_em(simfers; n=300, equidistant=false)
    rows = 3
    cols = 6

    interval1 = draw_interval(dims, spread_min, volume_min)

    intervals = [
        tuple(interval1, draw_interval(dims, spread_min, volume_min)) for
        _ in 1:(rows * cols)
    ]
    if !equidistant
        X = rand(n, 2)
    else
        r = range(
            Intervals.X_MIN + 0.01,
            Intervals.X_MAX - 0.01,
            Int(ceil(sqrt(n))),
        )
        X = vcat([[x y] for x in r, y in r]...)
    end

    for (label, simfer) in simfers
        fig, ax = subplots(rows, cols; layout="constrained")

        simf = simfer(X)
        scores = [simf(i1, i2) for (i1, i2) in intervals]
        perm = sortperm(scores; order=Base.Sort.Reverse)
        scores = scores[perm]
        intervals = intervals[perm]

        for k in eachindex(ax)
            i1, i2 = intervals[k]

            plot_traversal_count(i1, i2, X; ax=ax[k])
            title = """
                score = $(round.(scores[k]; digits=2))
                tc = $(traversal_count(i1, i2, X)),
                V1 = $(round(volume(i1); digits=1)), V2 = $(round(volume(i2); digits=1)),
                mc1 = $(sum(elemof(X, i1))), mc2 = $(sum(elemof(X, i2)))"""
            ax[k].set_title(title)
        end

        fig.suptitle(label)
    end
    return X
end

plot_em(Dict("raw" => simf_traversal_count_root); n=100, equidistant=true)
# plot_em(Dict("raw" => simf_traversal_count_root); n=100, equidistant=false)
# plot_em(Dict("raw" => simf_traversal_count_root); n=300, equidistant=false)
