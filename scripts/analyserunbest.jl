using AlgebraOfGraphics
using CairoMakie
using ColorSchemes
using DataFrames
using KittyTerminalImages
using LaTeXStrings
using Latexify
using MLFlowClient
using MLJ
using RSLModels
using Serialization
using Statistics
RidgeRegressor = @load RidgeRegressor pkg = MLJLinearModels

# using MathTeXEngine
# textheme = Theme(;
#     fonts=(;
#         regular=texfont(:text),
#         bold=texfont(:bold),
#         italic=texfont(:italic),
#         bold_italic=texfont(:bolditalic),
#     ),
# )

# set_aog_theme!()
set_theme!()
update_theme!(theme_latexfonts())
update_theme!(;
    resolution=(1200, 800),
    # palette=(; colors=:seaborn_colorblind),
    palette=(; color=Reverse(ColorSchemes.seaborn_colorblind.colors)),
)

df___ = deserialize("../2024-evostar-results/2023-11-14T14-38-31.912-df__.jls")

function dx_pretty(dx)
    return L"\mathcal{D}_\mathcal{X} = %$dx"
end

function prep(df)
    alg_pretty = Dict(
        "DecisionTreeRegressor2-50" => "DT (2–50)",
        "DecisionTreeRegressor2-100" => "DT (2–100)",
        "DecisionTreeRegressor2-200" => "DT (2–200)",
        "RandomForestRegressor10-2-50" => "RF (10 à 2–50)",
        "RandomForestRegressor20-2-100" => "RF (20 à 2–100)",
        "RandomForestRegressor20-2-200" => "RF (20 à 2–200)",
        "XCSF50" => "XCSF (1–50)",
        "XCSF100" => "XCSF (1–100)",
        "XCSF200" => "XCSF (1–200)",
    )

    # Unselect columns that we do not require.
    df = select(
        df,
        Not([
            "start_time",
            "end_time",
            "params.data.sha256",
            "run_name",
            "lifecycle_stage",
            "experiment_id",
            "duration",
            "lbounds",
            "ubounds",
            "intervals",
            "run_id",
            "status",
        ]),
    )

    # Unify column names.
    df = rename(
        df,
        "params.data.fname" => "task.data.fname",
        "params.data.hash" => "task.hash",
        "params.data.N" => "task.data.N",
        "params.data.DX" => "task.DX",
        "duration_min" => "metrics.duration_min",
        "K" => "metrics.K",
        "task_fname" => "task.fname",
        "model" => "task.model",
        "intervals_gt" => "task.intervals_gt",
        "sim_tcr" => "metrics.computesimresult",
    )

    df[!, "task.K"] = length.(df[:, "task.intervals_gt"])

    # Extract similarity values.
    df[!, "metrics.sim"] =
        getproperty.(df[:, "metrics.computesimresult"], :value)

    # Get data stats file names.
    df[!, "task.stats.fname"] =
        replace.(
            df[:, "task.data.fname"],
            r"^../RSLModels.jl/2023-11-10T19-01-45.606-task-selection/" => "../2023-11-10-data-merged/",
            r".data.npz" => ".stats.jls",
        )
    df[!, "task.stats"] = deserialize.(df[:, "task.stats.fname"])
    # Make sure that these are the same learning tasks.
    @assert all(
        df[!, "task.hash"] .== get.(df[:, "task.stats"], :hash, missing),
    )
    df[!, "task.stats.stats"] = get.(df[:, "task.stats"], :stats, missing)
    df[!, "task.stats.params"] = get.(df[:, "task.stats"], :params, missing)
    df[!, "task.rate_coverage_min"] =
        get.(df[:, "task.stats.stats"], :rate_coverage_min, missing)
    df[!, "task.rate_coverage"] =
        get.(df[:, "task.stats.stats"], :coverage, missing)

    df[!, "params.algorithm"] =
        get.(Ref(alg_pretty), df[:, "params.algorithm"], missing)

    df[!, "params.algorithm.class"] =
        get.(split.(df[:, "params.algorithm"], " "), 1, missing)

    # We don't have enough learning tasks for these configurations.
    df = subset(df, "task.K" => (k -> k .<= 18))

    df[!, "metrics.sim"] = -df[:, "metrics.sim"]

    df[!, "task.rate_coverage_bin"] =
        ifelse.(
            df[:, "task.rate_coverage"] .< 0.85,
            L"\kappa < 0.85",
            L"\kappa \geq 0.85",
        )

    # Sort column names alphabetically.
    df = select(df, sort(names(df)))

    return df
end

function chk(df)
    @assert all(
        10 .==
        combine(groupby(df, ["params.algorithm", "task.hash"]), nrow).nrow,
    )
end

df = prep(df___)
chk(df)

coldx = mapping(; col="task.DX" => dx_pretty)

function graphs(df)
    set_theme!()
    update_theme!(theme_latexfonts())
    update_theme!(;
        resolution=(1100, 700),
        # palette=(; colors=:seaborn_colorblind),
        palette=(; color=reverse(ColorSchemes.seaborn_colorblind.colors)),
    )

    alg_sorter = sorter([
        "DT (2–50)",
        "DT (2–100)",
        "DT (2–200)",
        "RF (10 à 2–50)",
        "RF (20 à 2–100)",
        "RF (20 à 2–200)",
        "XCSF (1–50)",
        "XCSF (1–100)",
        "XCSF (1–200)",
    ])

    # k_sorter = sorter(latexstring.("K=", sort(unique(df[:, "task.K"]))))
    k_sorter_ = sorter("K=" .* string.(sort(unique(df[:, "task.K"]))))

    coldx = mapping(; col="task.DX" => dx_pretty)
    coloralg = mapping(; color="params.algorithm" => alg_sorter => "Algorithm")
    rowcover = mapping(; row="task.rate_coverage_bin")
    plt =
        data(df) *
        mapping(
            "metrics.sim" => L"d_X(\mathcal{M}, \mathcal{M}_0)";
            # row="task.K" => (k -> k_sorter(L"K=%$k")),
            row="task.K" => (k -> k_sorter_("K=$k")),
            # linestyle="params.algorithm",
            # linestyle="params.algorithm.class",
            # linestyle="params.algorithm.class" => alg_sorter => "Algorithm",
            # linestyle="params.algorithm.class" => alg_sorter => "Algorithm",
            # linestyle=:dash,
            # linestyle=[:solid, :dash, :dot],
        ) *
        # mapping(; linestyle="params.algorithm.class" => alg_sorter => "Algorithm")
        coloralg *
        coldx *
        visual(ECDFPlot; linewidth=1)
    fig = draw(
        plt;
        axis=(; ylabel="Density", xscale=log10),
        facet=(; linkxaxes=:none),
        figure=(; resolution=(1100, 1000)),
    )
    CairoMakie.save("plots/DXvsK.pdf", fig)
    display(fig)
    println()

    plt =
        data(df) *
        mapping(
            "metrics.mae.test" => "Test MAE";
            row="task.K" => (k -> k_sorter_("K=$k")),
            # linestyle="params.algorithm",
        ) *
        # rowcover *
        coloralg *
        coldx *
        visual(ECDFPlot; linewidth=1)
    fig = draw(
        plt;
        axis=(; ylabel="Density", xscale=log10),
        facet=(; linkxaxes=:none),
        figure=(; resolution=(1100, 1000)),
    )
    CairoMakie.save("plots/DXvsKMAE.pdf", fig)
    display(fig)
    println()

    plt =
        data(df) *
        mapping("metrics.sim" => L"d_X(\mathcal{M}, \mathcal{M}_0)";) *
        rowcover *
        coloralg *
        coldx *
        visual(ECDFPlot; linewidth=2, linestyle=:dash)
    fig = draw(plt; axis=(; ylabel="Density"), facet=(; linkxaxes=:none))
    CairoMakie.save("plots/DXvscoverage.pdf", fig)
    display(fig)
    println()

    fig = draw(
        data(df) *
        mapping("task.rate_coverage") *
        coldx *
        (
            histogram(; bins=50, normalization=:pdf) +
            AlgebraOfGraphics.density()
        ),
    )
    CairoMakie.save("plots/coveragehist.pdf", fig)
    display(fig)
    println()

    plt =
        data(df) *
        mapping(
            "metrics.mae.test" => "Test MAE";
            # linestyle="params.algorithm",
        ) *
        rowcover *
        coloralg *
        coldx *
        visual(ECDFPlot; linewidth=2, linestyle=:dash)
    fig = draw(plt; axis=(; ylabel="Density"), facet=(; linkxaxes=:none))
    CairoMakie.save("plots/DXvsMAE.pdf", fig)
    display(fig)
    println()

    return nothing
end

function computecost(df)
    # This is in seconds because the @timed macro returns seconds for its :time field.
    df[!, "metrics.duration_sim"] =
        getproperty.(df[:, "metrics.computesimresult"], :time)

    df[!, "metrics.K_bin"] = Int.(floor.(df[!, "metrics.K"] ./ 100)) * 100

    df[!, "metrics.pairings"] = df[:, "task.K"] .* (df[:, "metrics.K"])

    draw(
        data(df) *
        mapping(
            "metrics.duration_sim";
            col="metrics.K_bin" => nonnumeric,
            row="task.K" => nonnumeric,
        ) *
        AlgebraOfGraphics.density(),
    )

    for N in unique(df[:, "task.data.N"])
        dfsub = subset(df, "task.data.N" => (n -> n .== N))
        df_time = combine(
            # groupby(df, ["metrics.K_bin", "task.K"]),
            groupby(dfsub, ["metrics.pairings"]),
            # "metrics.pairings",
            "metrics.duration_sim" => mean,
            "metrics.duration_sim" => std,
            nrow,
        )

        display(df_time)

        X = (; npairings=dfsub[:, "metrics.pairings"])
        # Ugly but whatever.
        X = coerce(X, "npairings" => Continuous)
        y = dfsub[:, "metrics.duration_sim"]
        mach = machine(RidgeRegressor(; fit_intercept=false), X, y)
        fit!(mach)
        ypred = predict(mach, X)
        println("MAE: ", mae(ypred, y))
        println("R²: ", rsquared(ypred, y))

        fig = Figure()
        ax = Axis(fig[1, 1])

        draw!(
            ax,
            data(df_time) * (
                mapping(
                    "metrics.pairings",
                    "metrics.duration_sim_mean";
                    color="nrow" => log => "Log of number of observations",
                    # color="task.data.N",
                    # glowwidth="nrow" => log10,
                    # markersize="nrow",
                ) * visual(Scatter; marker='+')
                # mapping(
                #     "metrics.pairings",
                #     "metrics.duration_sim_mean",
                #     "metrics.duration_sim_std",
                # ) * visual(Errorbars; color=:green)
            ),
        )
        fparams = fitted_params(mach)
        ablines!(ax, 0, Dict(fparams.coefs)[:npairings])
        println("|X| = $N")
        display(current_figure())
        println(fparams)
    end
end
