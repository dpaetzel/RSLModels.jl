using Base.Filesystem
using CairoMakie
using Comonicon
using DataFrames
using KittyTerminalImages
using ProgressBars
using Serialization
using Statistics

using RSLModels.LocalModels
using RSLModels.Tasks
using RSLModels.Utils

function readstats(prefix_fname::String="data/genstats/genall")
    fnames = map(fname -> prefix_fname * "/" * fname, readdir(prefix_fname))

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

    return df
end

function densities(df, metric)
    DXs = unique(df."params.d")
    Ks = unique(df."params.k")

    fig = Figure()
    axs = Array{Axis}(undef, length(DXs), length(Ks))
    for (idx_DX, idx_K) in Iterators.product(eachindex(DXs), eachindex(Ks))
        DX = Int(DXs[idx_DX])
        K = Int(Ks[idx_K])

        local ax =
            Axis(fig[idx_DX, idx_K]; title="DX=$DX, K=$K", titlefont=:regular)
        axs[idx_DX, idx_K] = ax

        # Let's not copy but view (but be careful not to alter stuff).
        local df_sel =
            @view df[(df."params.d" .== DX) .&& (df."params.k" .== K), :]

        hist!(ax, df_sel[:, "stats.$metric"])
        density!(ax, df_sel[:, "stats.$metric"])
    end
    axs[length(DXs), 1].xlabel = "$metric"
    axs[length(DXs), 1].ylabel = "density"

    return display(current_figure())
end

df = readstats()
densities(df, "overlap")
