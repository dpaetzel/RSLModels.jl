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
using Tables

Ks_default = [2, 4, 8, 10, 14, 18, 25, 32]

function readdata(fname; dropcensored=true, collapsemedian=false)
    df = DataFrame(
        CSV.File(
            fname;
            # TODO Lock this header with genkdata.jl
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

    # # Set scientific types.
    df[!, :DX] = coerce(df.DX, Count)
    df[!, collapsemedian ? :K_median : :K] =
        coerce(df[:, collapsemedian ? :K_median : :K], Count)
    return df
end

fname_sel_default(fname) = replace(fname, r".csv$" => ".paramselect.csv")

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

    # For the smallest `rate_coverage_min`, plot distributions over `a`, `b` and
    # `spread_min`.
    plt_hist =
        data(
            df_sel,
            # df_sel[
            #     df_sel.rate_coverage_min .== first(sort(df.rate_coverage_min)),
            #     :,
            # ],
        ) *
        mapping(;
            col=:DX => nonnumeric,
            row=:K => nonnumeric,
            color=:rate_coverage_min => nonnumeric,
        ) *
        AlgebraOfGraphics.density()
    display(
        draw(
            plt_hist * mapping(:a);
            facet=(; linkyaxes=:none),
            figure=(; resolution=(1100, 1000)),
        ),
    )
    println()
    display(
        draw(
            plt_hist * mapping(:b);
            facet=(; linkyaxes=:none),
            figure=(; resolution=(1100, 1000)),
        ),
    )
    println()
    display(
        draw(
            plt_hist * mapping(:spread_min);
            facet=(; linkyaxes=:none),
            figure=(; resolution=(1100, 1000)),
        ),
    )
    println()

    error("This disregards the fact that the parameters are not independent")
    # We need to estimate their joint distribution!
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

    display(
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
        ),
    )
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
