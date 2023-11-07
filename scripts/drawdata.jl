using AlgebraOfGraphics
using Base.Filesystem
using CairoMakie
using DataFrames
using Dates
using KittyTerminalImages
using LaTeXStrings
using ProgressBars
using Serialization
using Statistics
using StatsBase

using RSLModels.LocalModels
using RSLModels.Tasks
using RSLModels.Utils
using RSLModels.Plots

# bins = [0.7, 0.8, 0.9, 0.95]
bins = [0.7, 0.8, 0.9]

# We aim at having this many learning tasks per combination of d, K and coverage bin.
n_per_d_k_coverage = 4
Ks = [3, 5, 7, 10, 15, 20, 30]

mapping_grid = mapping(;
    row="stats.coverage_bin" => nonnumeric,
    col="params.d" => nonnumeric,
)

function analyse(;
    folder="2023-10-27-data-full-12d",
    bins=bins,
    n_per_d_k_coverage=n_per_d_k_coverage,
    Ks=Ks,
)
    df = readstats(; prefix_fname=folder)

    println("$mjgood Read $(size(df, 1)) data set statistics from $folder.")

    if all(
        df[df[:, "stats.K"] .== 2, :][
            :,
            "stats.overlap_pairs_mean_per_rule",
        ] .== 0,
    )
        println("$mjgood All sets with K=2 have an overlap of 0, as expected.")
    end

    # Check whether all is dandy.
    println("Look at this for any weirdnesses:")
    display(
        combine(
            groupby(df, ["params.d", "params.nif"]),
            # Note that we select any column here (`fnames`) and count how many
            # values it has in order to be able to rename the output column
            # within the same expression.
            "fname" => (df_ -> size(df_, 1)) => "n_datasets",
        ),
    )

    # Filter for `coverage >= 0.7` (may not be fulfilled by a margin due to coverage
    # computation being sample-based but we have enough data to just discard these
    # cases).
    gt = df[!, "stats.coverage"] .>= bins[1]
    println(
        "\n" *
        "Discarding $(size(df, 1) - count(gt)) of $(size(df, 1)) rows where " *
        "coverage is smaller than lowest bin …",
    )
    df = df[gt, :]

    # Add a column denoting the respective coverage bin.
    df[!, "stats.coverage_bin"] = map(
        coverage -> bins[searchsortedlast(bins, coverage)],
        df[:, "stats.coverage"],
    )

    n_dims = length(unique(df[:, "params.d"]))
    n_overlap = length(bins)
    n_Ks = length(Ks)
    n_sel = n_dims * n_overlap * n_Ks * n_per_d_k_coverage
    println(
        "\n" *
        "Overall, we want to select (n_dims=$n_dims) * (n_overlap=$n_overlap) " *
        "* (n_Ks=$n_Ks) * (n_per_d_k_coverage=$n_per_d_k_coverage) = $n_sel of " *
        "the remaining $(size(df, 1)) learning tasks.",
    )

    println("This is the distribution of the K's.")
    display(draw(data(df) * frequency() * mapping("stats.K")))

    println("\n" * "These are the frequencies of the combinations.")
    println(
        "\n" *
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
    println(
        "\n" * "If I only keep `\"stats.K\" <= 30` and bin K to resolution 5:",
    )
    display(
        draw(
            data(df[df[:, "stats.K"] .<= 30, :]) *
            frequency() *
            mapping(
                "params.d" => nonnumeric,
                "stats.coverage_bin" => nonnumeric;
                # TODO have to bin this first
                layout="stats.K" => nonnumeric ∘ (x -> round(x / 5) * 5),
            ),
        ),
    )

    function draw_foreach_k(df; n_per_k=n_per_d_k_coverage, ks=Ks)
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

    # Select learning tasks for each combination of `DX` and `coverage_bin`.
    df_sel = combine(
        groupby(df, ["params.d", "stats.coverage_bin"]),
        draw_foreach_k,
    )
    df_sel = flatten(df_sel, "x1")
    # Extract `K` to its own column.
    df_sel[!, "K"] = map(first, df_sel.x1)
    # Extract the file names from the inner `DataFrame`s. The ternary is required
    # to catch cases where the DataFrame is empty (since then indexing for "fname"
    # fails) which can happen if not all data has been generated yet or if we
    # discarded (see above) all of the tasks within one of the groups.
    df_sel[!, "fname"] =
        map(pair -> size(pair[2], 1) > 0 ? pair[2][!, "fname"] : [], df_sel.x1)
    df_sel[!, "nif"] = map(
        pair -> size(pair[2], 1) > 0 ? pair[2][!, "params.nif"] : [],
        df_sel.x1,
    )
    # Flatten wrt to the file names and nif (which are vectors up until now).
    df_sel = flatten(df_sel, "fname")
    df_sel = flatten(df_sel, "nif")

    # # Next make sure that all the combinations are in the DataFrame (we only
    # # draw from existing combinations above).
    # for dx in unique(df[:, "params.d"])
    #     for coverage in bins
    #         for K in Ks
    #             df[:, "params.d"] ==
    #         end
    #     end
    # end

    # Print any combinations for which we have not yet created enough learning
    # tasks.
    println(
        "\n" *
        "Look at the following plot for configurations where we " *
        "did not yet create enough learning tasks (not-yellow areas in the plot).",
    )
    display(
        draw(
            data(df_sel) *
            frequency() *
            mapping(
                "params.d" => nonnumeric,
                "stats.coverage_bin" => nonnumeric;
                layout="K" => nonnumeric,
            ),
        ),
    )

    println("\n")
    draw(
        data(df_sel) *
        frequency() *
        mapping(
            "nif" => nonnumeric;
            row="stats.coverage_bin" => nonnumeric,
            col="params.d" => nonnumeric,
        ),
    ) |> display

    println("\n")
    draw(data(df) * mapping_grid * mapping("stats.K") * frequency()) |> display

    draw(
        data(df) *
        mapping_grid *
        mapping("stats.K", "params.nif") *
        histogram(),
    ) |> display

    function write_fishscript(df_sel)
        length_df_sel = size(df_sel, 1)

        fnames_selected = df_sel.fname
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

    # return write_fishscript(df_sel)

    return df
end
