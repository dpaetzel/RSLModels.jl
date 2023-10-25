module MLFlowUtils

using DataFrames
using Dates
using MLFlowClient
using NPZ

using RSLModels.Intervals

export add_missing_keys!, has_bounds, load_runs, run_to_dict, tryread

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

# NOTE Always gotta `serve-results` for now, then do this.
function load_runs(exp_name; url="http://localhost:5000")
    println("Loading experiments from $url …")
    mlf = MLFlow(url)
    exp = searchexperiments(mlf; filter="name = '$exp_name'")
    @assert (length(exp) == 1) "Experiment not found or ambiguous, did you serve the right directory?"
    expid = exp[1].experiment_id
    println("Loading runs for experiment $expid from $url …")
    runs = searchruns(mlf, expid; max_results=10000)

    dicts = run_to_dict.(runs)
    add_missing_keys!(dicts)
    df = DataFrame(dicts)
    df[!, "params.data.DX"] .= parse.(Int, df[:, "params.data.DX"])
    df[!, "params.data.hash"] .= parse.(UInt, df[:, "params.data.hash"])

    # Add algorithm IDs for easier plotting.
    algorithms = sort(unique(df[:, "params.algorithm"]))
    algorithm_id = Dict(algorithms .=> 1:length(algorithms))
    df[!, "params.algorithm.id"] = [
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

end
