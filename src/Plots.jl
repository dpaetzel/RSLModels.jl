module Plots

using CairoMakie
using DataFrames
using LaTeXStrings

export grid, line

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

end
