using Base.Filesystem
using MLJ
using MLJLinearModels
using Serialization
using Statistics

using RSLModels.LocalModels
using RSLModels.Tasks
using RSLModels.Utils

function gentodisk(;
    d::Int=1,
    nif::Int=20,
    N::Int=200,
    seed::Int=0,
    rate_coverage_min::Float64=0.8,
    remove_final_fully_overlapped::Bool=false,
    full::Bool=false,
    prefix_fname::String="data/$d-$nif-$N-$seed-$rate_coverage_min-$remove_final_fully_overlapped",
)
    params = Dict(:d => d, :N => N, :nif => nif, :seed => seed)
    task = generate(
        d,
        nif,
        N;
        seed=seed,
        rate_coverage_min=rate_coverage_min,
        remove_final_fully_overlapped=remove_final_fully_overlapped,
    )
    match_X = task.match_X

    y_pred = output_mean(task.model, task.X)
    y_test_pred = output_mean(task.model, task.X_test)

    mach = machine(RidgeRegressor(), MLJ.table(task.X), task.y)
    fit!(mach; verbosity=0)
    y_test_pred = MLJ.predict(mach, table(task.X_test))
    mae_test_linear = mae(y_test_pred, task.y_test)

    stats = Dict(
        :K => length(task.model.conditions),
        :coverage => data_coverage(match_X),
        :overlap_pairs_mean_per_ruleset =>
            data_overlap_pairs_mean_per_ruleset(match_X),
        :overlap_pairs_mean_per_rule =>
            data_overlap_pairs_mean_per_rule(match_X),
        :mae_train => mean(abs.(task.y .- y_pred)),
        :mae_test => mean(abs.(task.y_test .- y_test_pred)),
        :mae_test_linear => mae_test_linear,
    )

    data = Dict(:hash => task.hash, :params => params, :stats => stats)

    # TODO Mlflow here

    mkpath(dirname(prefix_fname))
    serialize("$prefix_fname.stats.jls", data)

    if full
        save(prefix_fname, task)
    end

    return nothing
end
