using AlgebraOfGraphics
using CSV
using CairoMakie
using Comonicon
using DataFrames
using Distributions
using KittyTerminalImages
using LaTeXStrings
using MLJScientificTypes
using Statistics
using Tables

function readdata(fname; dropcensored=true, collapsemedian=false)
    df = DataFrame(
        CSV.File(
            fname;
            # TODO Set header in genkdata.jl
            header=[
                :DX,
                :rate_coverage_min,
                :spread_min,
                :a,
                :b,
                :rate_coverage,
                :K,
            ],
        ),
    )

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

    # Set scientific types.
    df[!, :DX] = coerce(df.DX, Count)
    df[!, collapsemedian ? :K_median : :K] =
        coerce(df[:, collapsemedian ? :K_median : :K], Count)
    return df
end

fname_sel_default(fname) = replace(fname, r".csv$" => ".paramselect.csv")

function selectparams(fname; fname_sel=fname_sel_default(fname))
    df = readdata(fname)

    draw(
        data(df) *
        mapping(;
            col=:DX => nonnumeric,
            row=:rate_coverage_min => nonnumeric,
        ) *
        mapping(:K) *
        histogram(; bins=20);
        facet=(; linkyaxes=:none),
    ) |> display
    println()

    DXs = [2, 3, 5, 8, 10, 13]
    Ks = [2, 4, 8, 10, 14, 18, 25, 32]
    df_sel = subset(df, :K => K -> K .∈ Ref(Ks), :DX => DX -> DX .∈ Ref(DXs))

    draw(
        data(df_sel) *
        mapping(;
            col=:DX => nonnumeric,
            row=:rate_coverage_min => nonnumeric,
        ) *
        mapping(:K => nonnumeric) *
        frequency();
        facet=(; linkyaxes=:none),
    ) |> display
    println()

    # The following is for the smallest `rate_coverage_min` only.
    plt_hist =
        data(
            df_sel[
                df_sel.rate_coverage_min .== first(sort(df.rate_coverage_min)),
                :,
            ],
        ) *
        mapping(; col=:DX => nonnumeric, row=:K => nonnumeric) *
        histogram(; bins=20)
    draw(plt_hist * mapping(:a); facet=(; linkyaxes=:none)) |> display
    println()
    draw(plt_hist * mapping(:b); facet=(; linkyaxes=:none)) |> display
    println()
    draw(plt_hist * mapping(:spread_min); facet=(; linkyaxes=:none)) |> display
    println()
    draw(plt_hist * mapping(:a, :b); facet=(; linkyaxes=:none)) |> display
    println()

    df_sel_mean = combine(
        groupby(df_sel, [:DX, :rate_coverage_min, :K]),
        nrow => :count,
        [:spread_min, :a, :b] =>
            (
                (s, a, b) -> (
                    spread_min_mean=mean(s),
                    a_mean=mean(a),
                    b_mean=mean(b),
                )
            ) => AsTable,
    )
    df_sel_mean = sort(df_sel_mean)

    df_sel_mean[!, :Beta] = Beta.(df_sel_mean.a_mean, df_sel_mean.b_mean)

    df_sel_mean[!, :Beta_mean] = mean.(df_sel_mean.Beta)

    df_sel_mean[!, :Beta_var] = var.(df_sel_mean.Beta)

    df_sel_mean[!, :Beta_std] = std.(df_sel_mean.Beta)

    draw(
        data(df_sel_mean) *
        mapping(;
            layout=:DX => nonnumeric,
            # row=:K => nonnumeric,
            color=:rate_coverage_min => nonnumeric,
        ) *
        (
            mapping(:K, :Beta_mean) * visual(ScatterLines) +
            mapping(:K, :Beta_mean, :Beta_std) * visual(Errorbars)
        ),
    ) |> display
    println()

    df_sel_mean = rename(
        df_sel_mean,
        :a_mean => :a,
        :b_mean => :b,
        :spread_min_mean => :spread_min,
    )

    CSV.write(fname_sel, df_sel_mean)

    return nothing
end

# fname = "2023-11-09-14-39-29-kdata.csv"
@main function main(fname)
    fname_sel = fname_sel_default(fname)

    println("Analysing $fname and writing selected parameters to $fname_sel …")

    selectparams(fname; fname_sel=fname_sel)

    return nothing
end
