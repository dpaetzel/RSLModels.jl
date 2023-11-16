# Compute the similarity values. Takes quite some time.
# TODO Properly parallelize using processes and onall (instead of trusting transform threading)
using Infiltrator
using AlgebraOfGraphics
using CairoMakie
using DataFrames
using Dates
using Distributed
using KittyTerminalImages
using LaTeXStrings
using MLFlowClient
using Missings
using NPZ
using ProgressMeter
using RSLModels
using RSLModels.Intervals
using RSLModels.MLFlowUtils
using RSLModels.Models
using RSLModels.Plots
using RSLModels.Scores
using RSLModels.Utils
using Serialization
using Statistics

dname = "../results/2024-evostar/2023-11-12T15:11:21.393284-evostar-runbest-10rep/mlruns"

function fixpath(path)
    if Base.Libc.gethostname() == "sokrates"
        return replace(
            path,
            r".*-selection/" => "/home/david/mnt/2023-10-20T17-01-56.409-task-selection/",
        )
    elseif Base.Libc.gethostname() == "oc-compute03"
        return replace(
            path,
            r".*-selection/" => "/data/oc-compute03/hoffmada/RSLModels.jl/2023-10-20T17-01-56.409-task-selection/",
        )
    end
end

function intervals(lbounds, ubounds)
    return map(Interval, eachrow(lbounds), eachrow(ubounds))
end

function load_data(; dname=dname)
    fname_serial = dname * "/df.jls"

    df = try
        println("Trying to deserialize $fname_serial …")
        deserialize(fname_serial)
    catch e
        println("Deserialization failed, loading from mlflow …")
        df_ = load_runs("runbest", dname)
        serialize(fname_serial, df_)
        df_
    end

    bounds = tryread.(df[:, "artifact_uri"] .* "/bounds.npz")

    # Slightly ugly, I'd like to replace `[]` (which is never used because `get`
    # never fails here unless something is really broken) with `missing` but
    # then I only get `missing`s because one of the args is always `missing`.
    df[!, "lbounds"] .=
        [passmissing(get)(bound, "lowers", []) for bound in bounds]
    df[!, "ubounds"] .=
        [passmissing(get)(bound, "uppers", []) for bound in bounds]
    df[!, "intervals"] .= map(passmissing(intervals), df.lbounds, df.ubounds)

    df[!, "K"] .= passmissing(length).(df[!, "intervals"])

    df[!, "task_fname"] =
        replace.(df[:, "params.data.fname"], "data.npz" => "task.jls")

    for task_fname in df[:, "task_fname"]
        @assert isfile(task_fname) "Task file does not exist: $task_fname"
    end

    # Doing this vectorized may use prohibitively much RAM. Rather deserialize one
    # task at a time and then lookup the rows in `df` that link to that task and
    # then extract only the things we really need and then let the GC delete the
    # task again.
    # fnames_task = unique(fixpath.(df.task_fname))

    df = transform(
        groupby(df, "task_fname"),
        # Since we grouped by `"task_fname"` we only deserialize the first one
        # (since they're all equal in the group and we would otherwise
        # deserialize the same file many times).
        "task_fname" => (fnames -> deserialize(fnames[1])) => "task",
    )

    df = transform(
        df,
        "task" => (tasks -> getproperty.(tasks, :model)) => "model",
    )
    df = transform(
        df,
        "model" =>
            (models -> getproperty.(models, :conditions)) => "intervals_gt",
    )

    return df
end

function discard_data(df)
    nrows = nrow(df)
    df_ = subset(df, "params.data.DX" => (dx -> dx .<= (8)))
    println("Removed $(nrows - nrow(df_)) rows by DX≤8.")
    nrows = nrow(df_)

    df_ = subset(df_, "params.data.DX" => (dx -> dx .>= (3)))
    println("Removed $(nrows - nrow(df_)) rows by DX≥3.")
    nrows = nrow(df_)

    df_ = subset(df_, "params.algorithm" => (alg -> alg .!= "XCSF600"))
    println("Removed $(nrows - nrow(df_)) rows by alg≠XCSF600.")
    nrows = nrow(df_)

    return df_
end

# @everywhere using NPZ
# @everywhere using RSLModels.Scores
# @everywhere using DataFrames
# @everywhere using Serialization
function compute_sim_tcr(fname_npz, intervals, intervals_gt)
    X, time, rest... = @timed npzread(fname_npz)["X"]

    println(
        "Read NPZ file $fname_npz in $time s, computing sim_tcr for lengths " *
        "$(length(intervals)) and $(length(intervals_gt)) …",
    )

    # TODO Remove the / 2 after rebasing.
    @warn "Still have / 2 in compute_sim_tcr"
    @time "Sizes: $(length(intervals)) and $(length(intervals_gt)) with X shape $(size(X))" sim =
        similarity(
            intervals,
            intervals_gt;
            simf=simf_traversal_count_root(X),
        ) / 2

    return sim
end

df = load_data()
df_ = discard_data(df)

# idx = rand(1:nrow(df_), 1000)
# row = eachrow(df_[idx, :])[1]

# out = onall(Ref.(eachrow(df_))) do row
#     @timed compute_sim_tcr(
#         row["params.data.fname"],
#         row["intervals"],
#         row["intervals_gt"],
#     )
# end

df__ = transform(
    groupby(df_, "params.data.hash"),
    ["params.data.fname", "intervals", "intervals_gt"] =>
        (
            (fs, is, is_gt) -> map(
                ((f, i, i_gt) -> @timed compute_sim_tcr(f, i, i_gt)),
                fs,
                is,
                is_gt,
            )
            # Ref(@timed compute_sim_tcr.(fs, is, is_gt))
        ) => "sim_tcr",
)

thetime = replace(string(now()), ":" => "-")
serialize(dname * "/$thetime-df__.jls", df__)
