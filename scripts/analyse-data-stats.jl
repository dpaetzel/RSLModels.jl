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
