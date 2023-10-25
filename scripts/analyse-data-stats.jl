using Base.Filesystem
using CairoMakie
using Comonicon
using DataFrames
using Dates
using KittyTerminalImages
using LaTeXStrings
using MLJ
using ProgressBars
using Serialization
using Statistics
using StatsBase

using RSLModels.LocalModels
using RSLModels.Tasks
using RSLModels.Utils
using RSLModels.Plots

function readstats(;
    prefix_fname::String="remote-genstats-genall",
    suffix=".stats.jls",
)
    fnames = map(
        fname -> prefix_fname * "/" * fname,
        filter(fname -> endswith(fname, suffix), readdir(prefix_fname)),
    )

    data = deserialize.(fnames)
    df = DataFrame(
        map(
            dict -> merge(
                Dict("params." * string(k) => v for (k, v) in dict[:params]),
                Dict("stats." * string(k) => v for (k, v) in dict[:stats]),
            ),
            data,
        ),
    )

    df[!, "fname"] .= fnames

    df[!, "params.d"] .= Int.(df[:, "params.d"])
    df[!, "params.N"] .= Int.(df[:, "params.N"])
    df[!, "params.nif"] .= Int.(df[:, "params.nif"])
    df[!, "params.seed"] .= Int.(df[:, "params.seed"])
    return df
end

function densities_K(df; linkaxes=false)
    grid(df; xlabel=L"K", ylabel=L"Density", linkaxes=linkaxes) do ax, df_sel
        hist!(ax, df_sel[:, "stats.K"]; normalization=:pdf)
        return density!(ax, df_sel[:, "stats.K"])
    end
    return current_figure()
end

function densities_overlap(df; linkaxes=false)
    grid(
        df;
        xlabel=L"Overlap",
        ylabel=L"Density",
        linkaxes=linkaxes,
    ) do ax, df_sel
        hist!(
            ax,
            df_sel[:, "stats.overlap_pairs_mean_per_rule"];
            normalization=:pdf,
        )
        return density!(ax, df_sel[:, "stats.overlap_pairs_mean_per_rule"])
    end
    return current_figure()
end

function overlap_vs_coverage(df; linkaxes=false)
    grid(
        df;
        xlabel=L"Overlap/rule, color is number of rules $K$",
        ylabel=L"Coverage [% of input space]$$",
        linkaxes=linkaxes,
    ) do ax, df_sel
        scatter!(
            ax,
            df_sel[:, "stats.overlap_pairs_mean_per_rule"],
            df_sel[:, "stats.coverage"];
            color=df_sel[:, "stats.K"],
            marker='o',
        )
        K_min = min(df_sel[:, "stats.K"]...)
        K_max = max(df_sel[:, "stats.K"]...)
        ax.title = L"%$(ax.title.val), $K \in [%$K_min, %$K_max]$"

        return nothing
    end
end

function K_vs_coverage(df; linkaxes=false)
    grid(
        df;
        xlabel=L"Number of rules $K$",
        ylabel=L"Coverage [% of input space]$$",
        linkaxes=linkaxes,
    ) do ax, df_sel
        scatter!(
            ax,
            df_sel[:, "stats.K"],
            df_sel[:, "stats.coverage"];
            # color=df_sel[:, "stats.K"],
            marker='o',
        )
        K_min = min(df_sel[:, "stats.K"]...)
        K_max = max(df_sel[:, "stats.K"]...)
        ax.title = L"%$(ax.title.val), $K \in [%$K_min, %$K_max]$"

        return nothing
    end
end

function K_vs_overlap(df; linkaxes=false)
    grid(
        df;
        xlabel=L"Number of rules $K$",
        ylabel=L"Overlap$$",
        linkaxes=linkaxes,
    ) do ax, df_sel
        scatter!(
            ax,
            df_sel[:, "stats.K"],
            df_sel[:, "stats.overlap_pairs_mean_per_rule"];
            # color=df_sel[:, "stats.K"],
            marker='o',
        )
        K_min = min(df_sel[:, "stats.K"]...)
        K_max = max(df_sel[:, "stats.K"]...)
        ax.title = L"%$(ax.title.val), $K \in [%$K_min, %$K_max]$"

        return nothing
    end
end

function nif_vs_K(df)
    fig = Figure()
    ax_nif_K = Axis(fig[1, 1])
    boxplot!(ax_nif_K, df[:, "params.nif"], df[:, "stats.K"])
    ax_nif_K.xlabel = "nif"
    ax_nif_K.ylabel = "K"
    return display(current_figure())
end

function overlap(df; linkaxes=false)
    line(df; xlabel=L"Overlap$$", linkaxes=linkaxes) do ax, df_sel
        density!(ax, df_sel[:, "stats.overlap_pairs_mean_per_rule"];)

        return nothing
    end
end

function K(df; linkaxes=false)
    line(df; xlabel=L"K$$", linkaxes=linkaxes) do ax, df_sel
        density!(ax, df_sel[:, "stats.K"];)

        return nothing
    end
end

function nontrivial(df)
    return df[df[:, "stats.K"] .!== 2, :]
end

function plotall(df)
    folder_plots = "$(folder)-plots"

    mkpath(folder_plots)

    densities_K(df)
    CairoMakie.save(
        "$folder_plots/grid-DX-nif-density-K.pdf",
        current_figure(),
    )

    densities_overlap(df)
    CairoMakie.save(
        "$folder_plots/grid-DX-nif-density-overlap.pdf",
        current_figure(),
    )

    overlap_vs_coverage(df)
    CairoMakie.save(
        "$folder_plots/grid-DX-nif-scatter-overlap-coverage.pdf",
        current_figure(),
    )

    K_vs_coverage(df)
    CairoMakie.save(
        "$folder_plots/grid-DX-nif-scatter-K-coverage.pdf",
        current_figure(),
    )

    K_vs_overlap(df)
    CairoMakie.save(
        "$folder_plots/grid-DX-nif-scatter-K-overlap.pdf",
        current_figure(),
    )

    nif_vs_K(df)
    CairoMakie.save("$folder_plots/box-nif-K.pdf", current_figure())

    overlap(df)
    CairoMakie.save(
        "$folder_plots/line-DX-density-overlap.pdf",
        current_figure(),
    )

    K(df)
    return CairoMakie.save(
        "$folder_plots/line-DX-density-K.pdf",
        current_figure(),
    )
end

folder = "2023-10-18-386605-data-complete"
df = readstats(; prefix_fname=folder)

if all(
    df[df[:, "stats.K"] .== 2, :][:, "stats.overlap_pairs_mean_per_rule"] .==
    0,
)
    println("All sets with K=2 have an overlap of 0, as expected.")
end

# Check whether all is dandy.
display(combine(groupby(df, ["params.d", "params.nif"]), df_ -> size(df_, 1)))

function draw_foreach_k(df; n_per_k=4, ks=[3, 5, 7, 10, 15, 20, 30])
    tasks = Dict(k => DataFrame() for k in ks)

    # Go once over the DataFrame, shuffling its rows beforehand.
    for row in eachrow(df[shuffle(1:size(df, 1)), :])
        local K = row["stats.K"]

        # Get the vector of tasks that we already selected for the `K` of the
        # current row and if we have not already collected enough tasks for it, add
        # the task to it.
        if in(K, ks) && size(tasks[K], 1) < n_per_k
            push!(tasks[K], row)
        end
    end

    return tasks
end

bins = [0.7, 0.8, 0.9, 0.95]

# Filter for `coverage >= 0.7` (may not be fulfilled by a margin due to coverage
# computation being sample based but we have enough data to just discard these
# cases).
gt = df[!, "stats.coverage"] .>= bins[1]
println(
    "Discarding $(size(df, 1) - count(gt)) rows where coverage is " *
    "smaller than lowest bin …",
)
df = df[gt, :]

df[!, "stats.coverage_bin"] = map(
    coverage -> bins[searchsortedlast(bins, coverage)],
    df[:, "stats.coverage"],
)

# Select learning tasks for each combination of `DX` and `coverage_bin`.
df_sel =
    combine(groupby(df, ["params.d", "stats.coverage_bin"]), draw_foreach_k)
df_sel = flatten(df_sel, "x1")
# Extract `K` to its own column.
df_sel[!, "K"] = map(first, df_sel.x1)
# Extract the file names from the inner `DataFrame`s. The ternary is required
# to catch cases where the DataFrame is empty (since then indexing for "fname"
# fails).
df_sel[!, "fnames"] =
    map(pair -> size(pair[2], 1) > 0 ? pair[2][!, "fname"] : [], df_sel.x1)
# Flatten wrt to the file names (which are vectors up until now).
df_sel = flatten(df_sel, "fnames")

# Compute how many learning tasks we were able to select per combination of `DX`,
# `coverage` and `K`.
df_sizes = combine(
    groupby(df_sel, ["params.d", "stats.coverage_bin", "K"]),
    df -> size(df, 1),
)
# Print any combinations for which we have not yet created enough learning
# tasks.
display(df_sizes[df_sizes.x1 .!= 4, :])

function write_fishscript(df_sel)
    length_df_sel = size(df_sel, 1)

    fnames_selected = df_sel.fnames
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
            "echo There should be $length_df_sel tasks and there are (math (ls $folder_sel | wc -l) / 3).\n",
        )
    end

    return nothing
end

write_fishscript(df_sel)
