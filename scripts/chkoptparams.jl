using AlgebraOfGraphics
using CairoMakie
using DataFrames
using Dates
using JSON
using KittyTerminalImages
using LaTeXStrings
using MLFlowClient
using Missings
using NPZ
using Serialization

using RSLModels.Intervals
using RSLModels.MLFlowUtils
using RSLModels.Models
using RSLModels.Scores
using RSLModels.Utils

# TODO Make this a proper script with a CLI arg
df = load_runs(
    "optparams",
    "results/2023-11-10T19:41:30.234918-evostar-optparams/mlruns",
)
check(df)

# Extract all parameters of all algorithms and add them to the DataFrame using
# `missing` as necessary (e.g. XCSF does not have the same parameters as DTs).
df[!, "best_params"] = JSON.parsefile.(df.artifact_uri .* "/best_params.json")
keys_all = unique(first.(vcat(flatten.(df.best_params, Ref([]))...)))
dicts = Dict.(flatten.(df.best_params, Ref([])))
dicts = add_missing_keys!(dicts)
dicts = map(
    dict -> Dict(join(key, ".") => val for (key, val) in pairs(dict)),
    dicts,
)
df_best_params = DataFrame(dicts)
df = hcat(df, df_best_params)

# Improve the prefix of the newly created colums.
prefix_old = "ttregressor__regressor__"
prefix = "params.algorithm."
names_new = [replace(name, prefix_old => prefix) for name in names(df)]
rename!(df, names_new)

df_stacked = stack(
    df,
    filter(
        name ->
            startswith(name, prefix) &&
                nonmissingtype(eltype(df[:, name])) == Float64,
        names(df),
    );
    variable_name="param",
)
df_stacked = dropmissing(df_stacked, ["value"])

df_minmax = combine(
    groupby(df_stacked, ["param", "params.data.DX"]),
    "value" => minimum => "minimum",
    "value" => maximum => "maximum",
)

mapping_grid = mapping(;
    row="param" => (str -> replace(str, r"^params.algorithm." => "")),
    col="params.data.DX" => nonnumeric,
)

plt =
    data(df_minmax) *
    mapping_grid *
    mapping(
        "minimum",
        "params.data.DX" => x -> 0,
        "maximum",
        "params.data.DX" => x -> 0;
    ) *
    visual(Bracket)
draw(
    plt;
    facet=(; linkxaxes=:none),
    axis=(; yticksvisible=false),
    figure=(; resolution=(800, 1200)),
) |> display

println("")

draw(
    data(df_stacked) *
    mapping_grid *
    mapping("value";) *
    AlgebraOfGraphics.density(; datalimits=extrema);
    facet=(; linkxaxes=:none),
    figure=(; resolution=(800, 1200)),
) |> display
