using AlgebraOfGraphics
using Base.Filesystem
using CairoMakie
using CSV
using DataFrames
using Dates
using KittyTerminalImages
using LaTeXStrings
using Random
using Serialization
using Statistics
using StatsBase

using RSLModels.LocalModels
using RSLModels.Tasks
using RSLModels.Utils
using RSLModels.Plots

# We aim at having this many learning tasks per combination of d, K and coverage bin.
n_per_d_k_coverage = 4

fname_paramselect = "2023-11-09-18-35-32-kdata.paramselect.csv"
println("Reading parameter selection from $fname_paramselect …")
df_selparams = DataFrame(CSV.File(fname_paramselect))

dname_stats = "../2023-11-10-data-merged"
println("Reading data set statistics from $dname_stats …")
df_stats = readstats(; prefix_fname=dname_stats)

df_stats_K2 = df_stats[df_stats[:, "stats.K"] .== 2, :]
if all(df_stats_K2[:, "stats.overlap_pairs_mean_per_rule"] .== 0)
    println(
        "$mjgood All $(nrow(df_stats_K2)) sets with K=2 have an overlap of 0, as expected.",
    )
end

bins = sort(unique(df_selparams.rate_coverage_min))

# Filter for `coverage >= 0.7` (may not be fulfilled by a margin due to coverage
# computation being sample-based but we have enough data to just discard these
# cases).
# TODO Not 100% sure whether we want to do this
gt = df_stats[!, "stats.coverage"] .>= bins[1]
println(
    "Discarding $(size(df_stats, 1) - count(gt)) of $(size(df_stats, 1)) rows where " *
    "coverage is smaller than lowest bin …",
)
df_stats = df_stats[gt, :]

# Add a column denoting the respective coverage bin.
df_stats[!, "stats.coverage_bin"] = map(
    coverage -> bins[searchsortedlast(bins, coverage)],
    df_stats[:, "stats.coverage"],
)

n_DXs = length(unique(df_stats[:, "params.d"]))
n_coverage = length(bins)
Ks = sort(unique(df_stats[:, "params.K_target"]))
n_Ks = length(Ks)
n_sel = n_DXs * n_coverage * n_Ks * n_per_d_k_coverage
println(
    "Overall, we want to select (n_DXs=$n_DXs) * (n_coverage=$n_coverage) " *
    "* (n_Ks=$n_Ks) * (n_per_d_k_coverage=$n_per_d_k_coverage) = $n_sel of " *
    "the remaining $(size(df_stats, 1)) learning tasks.",
)

mapping_grid = mapping(;
    row="stats.coverage_bin" => nonnumeric,
    col="params.d" => nonnumeric,
)

df = df_stats

println("This is the distribution of the K's.")
display(draw(data(df) * frequency() * mapping("stats.K")))
println()

println("These are the frequencies of the combinations.")
println(
    "Note that we will try in the next step to draw from each of " *
    "these bins for each K in $Ks $n_per_d_k_coverage learning tasks.",
)
display(
    draw(
        data(df) *
        frequency() *
        mapping(
            "params.d" => nonnumeric,
            "stats.coverage_bin" => nonnumeric;
            # TODO have to bin this first
            # layout="stats.K" => nonnumeric,
        ),
    ),
)
println()

display(
    draw(
        data(df) *
        mapping(
            "stats.K";
            row="params.K_target" => nonnumeric,
            col="params.d" => nonnumeric,
            color="stats.coverage_bin" => nonnumeric,
        ) *
        histogram(; bins=30),
    ),
)
println()

df_sel = DataFrame(;
    DX=Int[],
    rate_coverage_min=Float64[],
    K=Int[],
    candidates=DataFrame[],
)
for row in eachrow(df_selparams)
    push!(
        df_sel,
        (
            DX=row.DX,
            rate_coverage_min=row.rate_coverage_min,
            K=row.K,
            candidates=subset(
                df,
                "params.d" => dx -> dx .== row.DX,
                "stats.K" => K -> K .== row.K,
                "stats.coverage_bin" => rcm -> rcm .== row.rate_coverage_min,
            ),
        ),
    )
end

df_sel[!, :count] = nrow.(df_sel.candidates)

display(
    draw(
        data(df_sel) *
        mapping(
            :K,
            :count;
            row=:rate_coverage_min => nonnumeric,
            col=:DX => nonnumeric,
        ) *
        visual(ScatterLines),
    ),
)
println()

function sampleupto(collection, n)
    if size(collection, 1) < n
        return collection
    else
        return sample(collection, n; replace=false)
    end
end

df_sel[!, :selected] =
    DataFrame.(sampleupto.(eachrow.(df_sel.candidates), n_per_d_k_coverage))
df_sel[!, :count_selected] = nrow.(df_sel.selected)

display(
    draw(
        data(df_sel) *
        mapping(
            :K,
            :count_selected;
            row=:rate_coverage_min => nonnumeric,
            col=:DX => nonnumeric,
        ) *
        visual(ScatterLines),
    ),
)
println()

display(
    draw(
        data(df_sel) *
        mapping(
            :K,
            :count_selected;
            row=:coverage_bin => nonnumeric,
            col=:DX => nonnumeric,
        ) *
        visual(ScatterLines),
    ),
)
println()

df_sel[!, :fnames] = getproperty.(df_sel.selected, "fname")

function write_fishscript(df_sel)
    fnames_selected = reduce(vcat, df_sel.fnames)
    # We replace colons with dashes so scp'ing is easier.
    thetime = replace(string(now()), ":" => "-")
    folder_sel = "$thetime-task-selection"
    open("$thetime-copy_selection.fish", "w") do f
        write(f, "#!/usr/bin/env fish\n\n")
        write(f, "# CAREFUL! THIS FILE WAS AUTOGENERATED!\n\n")
        write(f, "mkdir $folder_sel\n")
        for fname in fnames_selected
            fname_glob = replace(fname, "stats.jls" => "*")
            write(f, "cp -vu $(fname_glob) $folder_sel\n")
        end
        return write(
            f,
            "echo\n",
            "echo There should be $(length(fnames_selected)) tasks and there are (math (ls $folder_sel | wc -l) / 3).\n",
        )
    end

    return nothing
end

write_fishscript(df_sel)
