module MLFlowUtils

using DataFrames
using Dates
using MLFlowClient
using NPZ

using RSLModels.Intervals

export add_missing_keys!,
    algorithms_inv,
    check,
    has_bounds,
    load_runs,
    run_to_dict,
    runtimes,
    tryread

function killsubprocs(proc)
    # First, record PID of the process and its child processes.
    pid = getpid(proc)
    # https://superuser.com/questions/363169/#comment999457_363179
    pids = read(
        pipeline(
            `pstree -pn $pid`,
            `grep -o "([[:digit:]]*)"`,
            `grep -o "[[:digit:]]*"`,
        ),
        String,
    )
    pids = parse.(Int, split(chomp(pids), "\n"))

    # Try to end the process and its child processes nicely.
    kill(proc)

    # Send kill to all child process just in case.
    for pid in pids
        try
            run(`kill $pid`)
        catch e
            println(
                "Child process with PID $pid ended before we could kill it. 👍",
            )
        end
    end
end

# https://stackoverflow.com/a/68853482
function struct_to_dict(s, S)
    return Dict(key => getfield(s, key) for key in fieldnames(S))
end

function run_to_dict(r)
    infodict = struct_to_dict(r.info, MLFlowRunInfo)
    datadict = struct_to_dict(r.data, MLFlowRunData)
    paramsdict =
        Dict("params.$key" => val.value for (key, val) in datadict[:params])
    metricsdict =
        Dict("metrics.$key" => val.value for (key, val) in datadict[:metrics])
    # Transform infodict keys to String as well (params and metrics contain
    # dots which make it hard to use Symbols everywhere, but we want uniformity).
    infodict = Dict(string(key) => val for (key, val) in infodict)
    # Note that we ignore tags for now.
    return rundict = merge(infodict, paramsdict, metricsdict)
end

function has_bounds(label_algorithm)
    return !in(label_algorithm, ["Ridge", "KNeighborsRegressor"])
end

function tryread(fname_npz)
    try
        return npzread(fname_npz)
    catch SystemError
        return missing
    end
end

function intervals(lbounds, ubounds)
    return map(Interval, eachrow(lbounds), eachrow(ubounds))
end

"""
For each key `k` in each `dict` of the dictionaries `dicts`, add in-place an
entry `k => missing` if `!in(k, keys(dict))`. This allows us to pass the
resulting vector of dicts to `DataFrame`.
"""
function add_missing_keys!(dicts)
    keys_all = union(keys.(dicts)...)
    for key in keys_all
        for dict in dicts
            get!(dict, key, missing)
        end
    end
    return dicts
end

"""
If given an experiment name and a directory, start an `mlflow ui` process on
port 33333 using that directory as the backend store URI (i.e. specify the
`mlruns` folder!) and and load all runs of that experiment from there. Then, end
the process.

If given an experiment name and a URL, access the mlflow REST API at that URL
(i.e. make sure `mlflow ui` is running at the corresponding port—or simply use
the first form of this command so you don't need to care about running the
`mlflow ui` server by hand) and load all runs of that experiment from there.
"""
function load_runs end

function load_runs(exp_name, dir::String)
    port = 33333
    println("Starting mlflow ui server …")
    out = Pipe()
    err = Pipe()
    proc = run(
        pipeline(
            `mlflow ui --backend-store-uri "$dir" --default-artifact-root "$dir" --gunicorn-opts "--timeout 0" --port $port`;
            stdout=out,
            stderr=err,
        );
        wait=false,
    )
    println("Started mlflow ui server.")

    i_max = 15
    i = 1
    line = readline(err)
    while process_running(proc) &&
              !occursin("Booting worker with pid", line) &&
              i < i_max
        println(line)
        println("Waiting for mlflow REST interface to be responsive …")
        sleep(1.0)
        i += 1
        line = readline(err)
    end

    if !process_running(proc)
        error("Something went wrong, mlflow ui stopped by itself")
    elseif i >= i_max
        killsubprocs(proc)
        error("mlflow ui did not start or did not start fast enough")
    else
        println("mlflow ui up and running!")
    end

    df = load_runs(exp_name; url="http://localhost:$port")
    println("Fixing artifact URIs.")
    df[:, "artifact_uri"] = replace.(df.artifact_uri, r"^.*/mlruns" => "$dir")

    println("Shutting down mlflow ui server …")
    killsubprocs(proc)
    println("Shut down mlflow ui server.")

    return df
end

function load_runs(exp_name; url="http://localhost:5000")
    println("Loading experiments from $url …")
    mlf = MLFlow(url)
    exp = searchexperiments(mlf; filter="name = '$exp_name'")
    @assert (length(exp) == 1) "Experiment not found or ambiguous, did you serve the right directory?"
    expid = exp[1].experiment_id
    println("Loading runs for experiment $expid from $url …")
    runs = searchruns(mlf, expid; max_results=10000)
    println("Finished loading runs for experiment $expid from $url.")

    dicts = run_to_dict.(runs)
    add_missing_keys!(dicts)
    df = DataFrame(dicts)
    df[!, "params.data.DX"] .= parse.(Int, df[:, "params.data.DX"])
    df[!, "params.data.hash"] .= parse.(UInt, df[:, "params.data.hash"])

    # Add algorithm IDs for easier plotting.
    algorithms = sort(unique(df[:, "params.algorithm"]))
    algorithm_id = Dict(algorithms .=> 1:length(algorithms))
    df[!, "params.algorithm_id"] = [
        algorithm_id[algorithm_name] for
        algorithm_name in df[:, "params.algorithm"]
    ]

    df[!, "start_time"] .=
        Dates.unix2datetime.(round.(df.start_time / 1000)) .+
        Millisecond.(df.start_time .% 1000) .+
        # TODO Consider to use TimeZones.jl
        # Add my timezone.
        Hour(2)
    df[!, "end_time"] .=
        passmissing(
            Dates.unix2datetime,
        ).(passmissing(round).(df.end_time / 1000)) .+
        passmissing(Millisecond).(df.end_time .% 1000) .+
        # TODO Consider to use TimeZones.jl
        # Add my timezone.
        Hour(2)

    df[!, "duration"] = df.end_time .- df.start_time

    df[!, "duration_min"] =
        Missings.replace(df.end_time, now()) .- df.start_time
    return df
end

function algorithms_inv(df)
    return Dict(
        unique(df[:, "params.algorithm_id"] .=> df[:, "params.algorithm"]),
    )
end

function runtimes(df; unit=Minute, yscale=identity)
    n_algorithms = length(algorithms_inv(df))

    fig = Figure()
    ax = Axis(fig[1, 1]; yscale=yscale)
    scatter!(
        ax,
        df[:, "params.algorithm_id"],
        passmissing(Dates.value).(passmissing(ceil).(df.duration, [unit]));
        marker='+',
    )
    df_missing = df[ismissing.(df.duration), :]
    if size(df_missing, 1) > 0
        scatter!(
            ax,
            df_missing[:, "params.algorithm_id"],
            Dates.value.(ceil.(df_missing.duration_min, [unit]));
            marker='?',
        )
    end
    ax.xticks =
        (1:n_algorithms, [algorithms_inv(df)[id] for id in 1:n_algorithms])
    ax.xticklabelrotation = 0.5
    ax.ylabel = "Runtime [$(string(unit))]"
    return current_figure()
end

function check(df)
    df_n_runs = combine(groupby(df, "params.algorithm"), dfg -> size(dfg, 1))
    if all(df_n_runs[:, :x1] .== df_n_runs[1, :x1])
        println(
            "✅ All algorithms have been run the same number of times " *
            "(i.e. $(df_n_runs[1, :x1]) times).",
        )
    else
        println("❌ Algorithms have been run different number of times.")
        display(df_n_runs)
    end

    df_n_runs = combine(groupby(df, "params.data.hash"), dfg -> size(dfg, 1))
    if all(df_n_runs[:, :x1] .== df_n_runs[1, :x1])
        println(
            "✅ Each dataset was used the same number " *
            "of times (i.e. $(df_n_runs[1, :x1]) times).",
        )
    else
        println("❌ Datasets have been used different numbers of times.")
    end

    return nothing
end

end
