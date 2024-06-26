module MLFlowUtils

using Base64
using CSV
using DataFrames
using Dates
using MLFlowClient
using NPZ
using TOML

using RSLModels.Intervals

export add_missing_keys!,
    algorithms_inv,
    check,
    getmlf,
    has_bounds,
    loadruns,
    readcsvartifact,
    run_to_dict,
    runs_to_df,
    runtimes,
    tryread

"""
Get MLflow client using authentication.
"""
function getmlf(; url="http://localhost:5000")
    config = open("config.toml", "r") do file
        return TOML.parse(read(file, String))
    end

    username = config["database"]["user"]
    password = config["database"]["password"]
    encoded_credentials = base64encode("$username:$password")
    headers = Dict("Authorization" => "Basic $encoded_credentials")
    return MLFlow(url; headers=headers)
end

function killsubprocs(proc)
    # First, record PID of the process and its child processes.
    pid = getpid(proc)
    pids = []
    try
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
        # TODO Specify exception; this should actually only catch if grep exits with
        # exit code 1 because then there are no child processes any more
    catch e
        pids = []
    end

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

"""
    runs_to_df(runs::Vector{MLFlowRun})

Transforms the vector into a `DataFrame`.
"""
function runs_to_df(runs::Vector{MLFlowRun})
    dicts = run_to_dict.(runs)
    add_missing_keys!(dicts)
    return df = DataFrame(dicts)
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
    if !isempty(dicts)
        keys_all = union(keys.(dicts)...)
        for key in keys_all
            for dict in dicts
                get!(dict, key, missing)
            end
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
function loadruns end

function loadruns(exp_name, dir::String)
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

    df = loadruns(exp_name; url="http://localhost:$port")
    println("Fixing artifact URIs.")
    df[:, "artifact_uri"] = replace.(df.artifact_uri, r"^.*/mlruns" => "$dir")

    println("Shutting down mlflow ui server …")
    killsubprocs(proc)
    println("Shut down mlflow ui server.")

    return df
end

function todatetime(mlflowtime)
    return Dates.unix2datetime(round(mlflowtime / 1000)) +
           Millisecond(mlflowtime % 1000) +
           # TODO Consider to use TimeZones.jl
           # Add my timezone.
           Hour(2)
end

function loadruns(exp_name; url="http://localhost:5000", max_results=5000)
    # TODO Consider to serialize-cache this as well (see the `jid` variant of
    # `loadruns`)
    @info "Searching for experiment $exp_name at $url …"
    mlf = getmlf(; url=url)
    mlfexp = getexperiment(mlf, exp_name)

    @info "Loading runs for experiment \"$(mlfexp.name)\" from $url …"
    runs = searchruns(mlf, mlfexp; max_results=max_results)
    @info "Finished loading $(length(runs)) runs for experiment " *
          "\"$(mlfexp.name)\" from $url."

    @info "Converting mlflow data to Julia representations …"
    df = runs_to_df(runs)
    df[!, "start_time"] .= todatetime.(df.start_time)
    df[!, "end_time"] .= passmissing(todatetime).(df.end_time)

    @info "Adding helpful additional columns …"
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

function DataFrames.flatten(dict::Dict, key_hierarchy::AbstractVector)
    pairs_all = []
    for (key, val) in pairs(dict)
        if val isa Dict
            for pair in flatten(val, vcat(key_hierarchy, key))
                push!(pairs_all, pair)
            end
        else
            push!(pairs_all, vcat(key_hierarchy, key) => val)
        end
    end
    return pairs_all
end

"""
Given an artifact URI, tries to read the given CSV file from it using SSH and
convert it to a `DataFrame`.
"""
function readcsvartifact(artifact_uri, fname)
    # TODO Make SSH login configurable
    fpath = "$artifact_uri/$fname"

    # Check whether the file exist.
    process = run(
        `ssh c3d "test -f $fpath"`;
        # stdout=Base.DevNull,
        # stderr=Base.DevNull,
        wait=true,
    )

    # Check the exit code of the process
    if process.exitcode != 0
        println("File does not exist on the remote server.")
        return missing
    else
        io = open(`ssh c3d cat $fpath`)
        df = DataFrame(CSV.File(io))
        close(io)
        return df
    end
end

end
