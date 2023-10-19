using Base.Filesystem
using CairoMakie
using Comonicon
using DataFrames
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

    df[!, "params.d"] .= Int.(df[:, "params.d"])
    df[!, "params.N"] .= Int.(df[:, "params.N"])
    df[!, "params.nif"] .= Int.(df[:, "params.nif"])
    df[!, "params.seed"] .= Int.(df[:, "params.seed"])
    return df
end

L"""
Given a function (use do notation, though!) and a data frame, apply that
function to all the groups defined by the two given labels, creating a grid of
plots.

By default, grid over `params.d` and `params.nif` (typeset as
$\mathcal{D}_\mathcal{X}$ and $\nu$, respectivelty).
"""
function grid(
    f,
    df;
    linkaxes=false,
    xlabel=nothing,
    ylabel=nothing,
    grid_xlabel="params.d",
    grid_ylabel="params.nif",
    grid_xlabel_short=L"\mathcal{D}_\mathcal{X}",
    grid_ylabel_short=L"\nu",
)
    xs = sort(unique(df[:, grid_xlabel]))
    ys = sort(unique(df[:, grid_ylabel]))

    # TODO Add labels to columns and rows with label and value of x/y instead of
    # axis titles
    fig = Figure(; resolution=(230 * length(xs), 230 * length(ys)))
    axs = Array{Axis}(undef, length(xs), length(ys))
    for (idx_x, idx_y) in Iterators.product(eachindex(xs), eachindex(ys))
        x = xs[idx_x]
        y = ys[idx_y]

        local df_sel = @view df[
            (df[:, grid_xlabel] .== x) .&& (df[:, grid_ylabel] .== y),
            :,
        ]

        local title =
            L"%$grid_xlabel_short=%$x, %$grid_ylabel_short=%$y, n=%$(size(df_sel, 1))"
        local ax = Axis(fig[idx_x, idx_y]; title=title, titlefont=:regular)
        axs[idx_x, idx_y] = ax

        f(ax, df_sel)
    end

    if linkaxes
        linkaxes!(axs...)
    end

    if xlabel != nothing
        axs[length(xs), 1].xlabel = xlabel
    end
    if ylabel != nothing
        axs[length(xs), 1].ylabel = ylabel
    end

    return current_figure()
end

"""
Like grid but only go over one variable.
"""
function line(
    f,
    df;
    linkaxes=false,
    xlabel=nothing,
    grid_xlabel="params.d",
    grid_xlabel_short=L"\mathcal{D}_\mathcal{X}",
)
    xs = sort(unique(df[:, grid_xlabel]))

    # TODO Add labels to columns and rows with label and value of x/y instead of
    # axis titles
    fig = Figure(; resolution=(230 * length(xs), 300))
    axs = Array{Axis}(undef, length(xs))
    for (idx_x) in eachindex(xs)
        x = xs[idx_x]

        local df_sel = @view df[(df[:, grid_xlabel] .== x), :]

        local title = L"%$grid_xlabel_short=%$x, n=%$(size(df_sel, 1))"
        local ax = Axis(fig[1, idx_x]; title=title, titlefont=:regular)
        axs[idx_x] = ax

        f(ax, df_sel)
    end

    if linkaxes
        linkaxes!(axs...)
    end

    if xlabel != nothing
        axs[length(xs), 1].xlabel = xlabel
    end

    return current_figure()
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

folder = "data-stats/2023-10-18-386605-data-complete"
df = readstats(; prefix_fname=folder)

if all(
    df[df[:, "stats.K"] .== 2, :][:, "stats.overlap_pairs_mean_per_rule"] .==
    0,
)
    println("All sets with K=2 have an overlap of 0, as expected.")
end

df = nontrivial(df)

folder_plots = "$(folder)-plots"

mkpath(folder_plots)

densities_K(df)
CairoMakie.save("$folder_plots/grid-DX-nif-density-K.pdf", current_figure())

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
CairoMakie.save("$folder_plots/line-DX-density-overlap.pdf", current_figure())

K(df)
CairoMakie.save("$folder_plots/line-DX-density-K.pdf", current_figure())
