using AlgebraOfGraphics
using CSV
using CairoMakie
using ColorSchemes
using Comonicon
using DataFrames
using Distributions
using KittyTerminalImages
using LaTeXStrings
using ScientificTypes
using Statistics
using StatsBase
using Tables

Ks_default = [2, 4, 8, 12, 18, 24, 32]

function readdata(fname; dropcensored=true, collapsemedian=false)
    # TODO Lock this header with genkdata.jl
    csv_header =
        [:DX, :rate_coverage_min, :spread_min, :a, :b, :rate_coverage, :K]

    df = if isfile(fname)
        # Read the file into a DataFrame
        out = DataFrame(CSV.File(fname; header=csv_header))
        @info "Loaded CSV file $fname."
        out
    elseif isdir(fname)
        combined_df = DataFrame()

        fnames = readdir(fname)
        for f in fnames
            fpath = joinpath(fname, f)
            if isfile(fpath) && endswith(fpath, ".csv")
                df = DataFrame(CSV.File(fpath; header=csv_header))
                combined_df = vcat(combined_df, df)
            end
        end

        @info "Loaded $(length(fnames)) CSV files from directory $fname."
        combined_df
    else
        error("Provided path is neither a file nor a directory")
    end

    # When generating the data for this, we cancel trials at some number of
    # rules to not waste time with configurations that we're not interested in.
    # We can filter these based on whether the coverage condition is fulfilled
    # (if we cancel early, the coverage condition is not yet fulfilled).
    # Filtering these out may of course result in some of the configurations
    # having less trials than others. This could be done more elegantly but
    # fully serves our purpose for now.
    if dropcensored
        df = df[df.rate_coverage .>= df.rate_coverage_min, :]
    end

    if collapsemedian
        # Collapse the distributions of the `K`s to their median right away.
        df = combine(
            groupby(df, [:DX, :rate_coverage_min, :spread_min, :a, :b]),
            :K => median => :K_median,
            :rate_coverage => median => :rate_coverage_median,
        )
    end

    # # Set scientific types.
    df[!, :DX] = coerce(df.DX, Count)
    df[!, collapsemedian ? :K_median : :K] =
        coerce(df[:, collapsemedian ? :K_median : :K], Count)
    return df
end

function fname_sel_default(fname)
    return if isfile(fname)
        replace(fname, r".csv$" => ".paramselect.csv")
    elseif isdir(fname)
        replace(fname * ".paramselect.csv")
    else
        "paramselect.csv"
    end
end

function selectparams(fname, Ks; fname_sel=fname_sel_default(fname))
    set_theme!()
    update_theme!(theme_latexfonts())
    update_theme!(;
        resolution=(1100, 700),
        # palette=(; colors=:seaborn_colorblind),
        palette=(; color=reverse(ColorSchemes.seaborn_colorblind.colors)),
    )

    df = readdata(fname)

    # For each combination of `DX` and `rate_coverage_min` in the sample, show
    # the distribution over Ks.
    display(
        draw(
            data(df) *
            mapping(;
                col=:DX => nonnumeric,
                row=:rate_coverage_min => nonnumeric,
            ) *
            mapping(:K) *
            histogram(; bins=20);
            facet=(; linkyaxes=:none),
        ),
    )
    println()

    # Select the observations from the sample that are of interest (i.e. the
    # ones that fulfill our `K` condition).
    df_sel = subset(df, :K => K -> K .∈ Ref(Ks))

    # Show for each combination of `DX` and `rate_coverage_min`, how often we
    # observed each `K`.
    display(
        draw(
            data(df_sel) *
            mapping(;
                col=:DX => nonnumeric,
                row=:rate_coverage_min => nonnumeric,
            ) *
            mapping(:K => nonnumeric) *
            frequency();
            facet=(; linkyaxes=:none),
        ),
    )
    println()

    df_sel_mean = combine(
        groupby(df_sel, [:DX, :rate_coverage_min, :K]),
        nrow => :count,
        [:spread_min, :a, :b] =>
            ((s, a, b) -> histmode((; spread_min=s, a=a, b=b))) => AsTable,
    )
    df_sel_mean = sort(df_sel_mean)

    # NEXT compute beta distribution stats like above (mean etc.)

    df_sel_mean[!, :Beta] = Beta.(df_sel_mean.a, df_sel_mean.b)

    df_sel_mean[!, :Beta_mean] = mean.(df_sel_mean.Beta)

    df_sel_mean[!, :Beta_var] = var.(df_sel_mean.Beta)

    df_sel_mean[!, :Beta_std] = std.(df_sel_mean.Beta)

    # This is simply the formula from `Intervals.draw_spread`.
    df_sel_mean[!, :spread_mean] =
        df_sel_mean.spread_min .+
        mean.(df_sel_mean.Beta) .* (0.5 .- df_sel_mean.spread_min)

    display(
        draw(
            data(df_sel_mean) *
            mapping(;
                layout=:DX => nonnumeric,
                # row=:K => nonnumeric,
                color=:rate_coverage_min => nonnumeric,
            ) *
            (
                mapping(:K, :spread_mean) * visual(ScatterLines) +
                mapping(:K, :spread_mean, :Beta_std) * visual(Errorbars)
            ),
        ),
    )
    println()

    CSV.write(fname_sel, df_sel_mean)

    return nothing
end

function histmode(data::NamedTuple; nbins=Int(ceil(length(data[1])^(1 / 3))))
    matrix = hcat(data...)
    out, n_maxbin = histmode(matrix; nbins=nbins)
    return NamedTuple(vcat(keys(data) .=> out, :n_maxbin => n_maxbin))
end

# TODO Number of bins is probably suboptimal
function histmode(
    matrix::AbstractMatrix{Float64};
    nbins=Int(ceil(size(matrix, 1)^(1 / 3))),
)
    @info "Using $nbins bins per dimension."

    # TODO Consider whether Matrix/Tuple data transformations can be done better
    hst = fit(Histogram, Tuple([col for col in eachcol(matrix)]); nbins=nbins)

    # For sanity checking.
    n_maxbin = maximum(hst.weights)

    # Get the `CartesianIndex` of the bin with the most data points.
    idx_maxbin = argmax(hst.weights)

    # TODO Consider whether Matrix/Tuple data transformations can be done better
    # Get bin index for each data point. Transform matrix rows to tuples (since
    # `binindex` only accepts tuples).
    idxs =
        CartesianIndex.(
            StatsBase.binindex.(
                Ref(hst),
                [Tuple(row) for row in eachrow(matrix)],
            )
        )

    # Check which data points are in the highest density bin.
    idxs_max = idxs .== Ref(idx_maxbin)

    # Sanity check.
    @assert count(idxs_max) == n_maxbin

    # Extract data points using Bool indexing on the matrixified form of the
    # data.
    dfmax = matrix[idxs_max, :]

    @info "Estimating mode from mean of max bin containing " *
          "$n_maxbin of $(size(matrix, 1)) data points."

    # Compute mean.
    return mean.(eachcol(dfmax)), n_maxbin
end

"""
Analyse a sample (a CSV file) generated by `genkdata.jl`, derive parameter
values for `gendata.jl` and write them to another CSV file (with the same name
as the original file but extension `.selection.csv`).

# Args

- `fname`: Name of a CSV file generated by `genkdata.jl` containing samples of
  statistics of the condition set–generating process.
- `Ks`: Numbers of conditions to derive parameter values for.
"""
@main function main(fname, Ks...)
    fname_sel = fname_sel_default(fname)

    println("Analysing $fname and writing selected parameters to $fname_sel …")

    if isempty(Ks)
        println("No Ks given, using default $Ks_default …")
        Ks = Ks_default
    end

    selectparams(fname, Ks; fname_sel=fname_sel)

    return nothing
end
