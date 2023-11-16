using AlgebraOfGraphics
using CairoMakie
using DataFrames
using KittyTerminalImages
using LaTeXStrings
using Latexify
using MLFlowClient
using RSLModels
using Serialization
using Statistics

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
            # linestyle="params.algorithm.class",
            # linestyle="params.algorithm.class" => alg_sorter => "Algorithm",
            # linestyle=:dash,
        ) *
        # mapping(; linestyle="params.algorithm.class" => alg_sorter => "Algorithm")
        coloralg *
        coldx *
        visual(ECDFPlot; linewidth=2, linestyle=:dash)
    fig = draw(
        plt;
        axis=(; ylabel="Density"),
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
        visual(ECDFPlot; linewidth=2, linestyle=:dash)
    fig = draw(
        plt;
        axis=(; ylabel="Density"),
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

    fig =
        data(df) *
        mapping("task.rate_coverage") *
        coldx *
        (
            histogram(; bins=50, normalization=:pdf) +
            AlgebraOfGraphics.density()
        ) |> draw
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

function stats(df)
    grp = ["task.DX", "task.K", "task.rate_coverage_bin"]
    df_ = sort(
        transform(
            combine(
                groupby(
                    # Only keep one entry per task.
                    combine(groupby(df, ["task.hash", grp...]), nrow),
                    grp,
                ),
                nrow,
            ),
            "task.rate_coverage_bin" =>
                (
                    r ->
                        replace.(
                            r,
                            r"^.*<.*$" => L"<",
                            r"^.*geq.*$" => L"\geq",
                        )
                ) => "task.rate_coverage_bin",
        ),
    )

    df_summary = combine(
        groupby(df_, ["task.DX", "task.K"]),
        "task.DX" => (dx -> dx[1]),
        "task.K" => (k -> k[1]),
        "nrow" =>
            (
                nr -> length(nr) == 2 ? "$(nr[1])/$(nr[2])" : "$(nr[1])/0"
            ) => "Number of tasks",
        ;
        keepkeys=false,
    )

    println(tolatex(df_summary))

    mu = combine(df_, "nrow" => mean)
    println("Mean nrow: $mu")

    total = combine(df_, "nrow" => sum)
    println("Total: $total")

    println(
        tolatex(
            combine(
                groupby(df, "params.algorithm"),
                "metrics.K" =>
                    (k -> round(mean(k); digits=1)) => "metrics.K_mean",
                "metrics.K" =>
                    (k -> round(std(k); digits=1)) => "metrics.K_std",
            ),
        ),
    )

    println(
        tolatex(
            combine(
                groupby(
                    subset(df, "task.DX" => (dx -> dx .== 8)),
                    "params.algorithm",
                ),
                "metrics.K" =>
                    (k -> round(mean(k); digits=1)) => "metrics.K_mean",
                "metrics.K" =>
                    (k -> round(std(k); digits=1)) => "metrics.K_std",
            ),
        ),
    )

    return nothing
end

function tolatex(df)
    return join(join.(Vector.(eachrow(df)), "  &  "), "\\\\\n") * "\\\\"
end

function todo()
    data(df) * mapping("metrics.sim") * histogram(; bins=30) |> draw |> display
    data(df) * mapping("metrics.sim") * histogram(; bins=100) |>
    draw |>
    display
    data(df) *
    mapping(
        "metrics.sim" => L"d_X(a, b)";
        color="params.algorithm",
        layout="params.algorithm" => nonnumeric,
    ) *
    histogram(; bins=100) |>
    draw |>
    display

    data(df) *
    mapping(
        "metrics.sim" => L"d_X(a, b)";
        row="params.algorithm",
        color="params.algorithm",
    ) *
    coldx *
    histogram(; bins=70) |>
    draw |>
    display

    data(df) *
    mapping(
        "metrics.sim" => L"d_X(a, b)";
        row="task.rate_coverage_min" => nonnumeric,
        color="params.algorithm",
    ) *
    coldx *
    visual(ECDFPlot) |>
    draw |>
    display

    display(
        draw(
            data(subset(df, "task.rate_coverage_min" => (r -> r .>= 0.9))) *
            mapping(
                "metrics.sim" => L"d_X(a, b)";
                row="task.K" => nonnumeric,
                color="params.algorithm",
            ) *
            coldx *
            visual(ECDFPlot; linewidth=2);
        ),
    )

    display(
        draw(
            data(subset(df, "task.rate_coverage_min" => (r -> r .< 0.9))) *
            mapping(
                "metrics.sim" => L"d_X(a, b)";
                row="task.K" => nonnumeric,
                color="params.algorithm",
            ) *
            coldx *
            visual(ECDFPlot; linewidth=2);
            palettes=(; color=ColorSchemes.seaborn_colorblind.colors),
        ),
    )

    display(
        draw(
            data(df) *
            mapping(
                "metrics.sim" => L"d_X(a, b)";
                row="task.rate_coverage_min" => nonnumeric,
                color="params.algorithm",
            ) *
            coldx *
            visual(ECDFPlot; linewidth=2);
            palettes=(; color=ColorSchemes.seaborn_colorblind.colors),
        ),
    )

    display(
        draw(
            data(df) *
            mapping(
                "metrics.mae.test" => "MAE.test";
                row="task.rate_coverage_min" => nonnumeric,
                color="params.algorithm",
                linestyle="params.algorithm",
            ) *
            coldx *
            visual(ECDFPlot; linewidth=2);
            facet=(; linkxaxes=:none),
            palettes=(; color=ColorSchemes.seaborn_colorblind.colors),
        ),
    )

    # Add an index to each data set which counts up for each DX/K/rate_coverage_min
    # triple.
    df = combine(
        groupby(df, ["task.DX", "task.K", "task.rate_coverage_min"]),
        All(),
        "task.hash" =>
            (
                hashs ->
                    get.(
                        Ref(Dict(unique(hashs) .=> 1:length(unique(hashs)))),
                        hashs,
                        missing,
                    )
            ) => "idx_data",
    )

    data(df) *
    mapping(
        "task.K";
        # 9 algorithms, 10 repetitions.
        # "nrow" => (n -> n / 9 / 10);
        row="task.rate_coverage_min" => nonnumeric,
    ) *
    coldx *
    frequency() |>
    draw |>
    display
    # visual(ScatterLines) |> draw
    # visual(ScatterLines) |> draw

    df_summary = combine(
        groupby(
            df,
            [
                "task.DX",
                "task.K",
                "task.rate_coverage_min",
                "idx_data",
                "task.hash",
                "params.algorithm",
            ],
        ),
        # All(),
        # "params.data.hash" =>
        #     (hashs -> collect(1:length(hashs))) => "idx_data",
        # "idx_data",
        # "task.K",
        # "task.rate_coverage_min",
        "metrics.K" => mean,
        "metrics.K" => median,
        "metrics.K" => std,
        "metrics.sim" => mean,
        "metrics.sim" => median,
        "metrics.sim" => std,
    )

    plt =
        data(
            subset(df_summary, "task.rate_coverage_min" => (r -> r .== 0.9)),
        ) *
        mapping(; color="params.algorithm", row="task.K" => nonnumeric) *
        coldx *
        (
            mapping(
                "idx_data" => nonnumeric,
                "metrics.sim_mean" => abs => L"d_X(a, b)",
            ) * visual(ScatterLines)
        )
    # draw(plt; axis=(yscale=log10,), figure=(resolution=(1500, 1000),))
    draw(plt; axis=(), figure=(resolution=(1500, 1000),)) |> display

    return nothing
end
