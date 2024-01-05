using Base.Filesystem
using MLJ
using MLJLinearModels
using Mmap
using Serialization
using Statistics

using RSLModels.AbstractModels
using RSLModels.LocalModels
using RSLModels.Tasks
using RSLModels.Utils

function gentodisk(;
    d::Int=1,
    N::Int=200,
    spread_min::Float64=0.0,
    # Parameters for the Beta distribution used for drawing spread rates.
    params_spread::NamedTuple{(:a, :b),Tuple{Float64,Float64}}=(a=1.0, b=1.0),
    seed::Int=0,
    rate_coverage_min::Float64=0.8,
    # This is just a tag to mark the file(s); not used during task generation.
    K_target::Int=10,
    remove_final_fully_overlapped::Bool=false,
    full::Bool=true,
    usemmap::Bool=false,
    prefix_fname::String="data/$d-$K_target-$N-$seed-$rate_coverage_min-$remove_final_fully_overlapped",
)
    params = Dict(
        :d => d,
        :N => N,
        :spread_min => spread_min,
        :params_spread_a => params_spread.a,
        :params_spread_b => params_spread.b,
        :seed => seed,
        :K_target => K_target,
    )
    task = generate(
        d,
        N;
        spread_min=spread_min,
        params_spread=params_spread,
        seed=seed,
        rate_coverage_min=rate_coverage_min,
        remove_final_fully_overlapped=remove_final_fully_overlapped,
    )

    X, y, X_test, y_test, match_X = generate_data(task; usemmap=usemmap)

    (path_y_pred, io_y_pred) = mktemp(tempdir())
    y_pred = mmap(io_y_pred, Vector{Float64}, task.n_train)

    (path_y_test_pred, io_y_test_pred) = mktemp(tempdir())
    y_test_pred = mmap(io_y_test_pred, Vector{Float64}, task.n_test)

    y_pred[:] = output_mean(task.model, X)
    mae_train = mean(abs.(y .- y_pred))

    y_test_pred[:] = output_mean(task.model, X_test)
    mae_test = mae(y_test_pred, y_test)

    mach = machine(RidgeRegressor(), MLJ.table(X), y)
    fit!(mach; verbosity=0)
    y_test_pred[:] = MLJ.predict(mach, table(X_test))
    mae_test_linear = mae(y_test_pred, y_test)

    stats = Dict(
        :rate_coverage_min => rate_coverage_min,
        :K => length(task.model.conditions),
        :coverage => data_coverage(match_X),
        :overlap_pairs_mean_per_ruleset =>
            data_overlap_pairs_mean_per_ruleset(match_X),
        :overlap_pairs_mean_per_rule =>
            data_overlap_pairs_mean_per_rule(match_X),
        :mae_train => mae_train,
        :mae_test => mae_test,
        :mae_test_linear => mae_test_linear,
    )

    data = Dict(:hash => task.hash, :params => params, :stats => stats)

    # TODO Mlflow here

    mkpath(dirname(prefix_fname))
    serialize("$prefix_fname.stats.jls", data)

    if full
        write_npz(prefix_fname, task, X, y, X_test, y_test, match_X)
        serialize("$prefix_fname.task.jls", task)
    end

    close(io_y_pred)
    close(io_y_test_pred)
    rm(path_y_pred)
    rm(path_y_test_pred)

    # TODO Consider to use Bumper.jl to ensure that nothing stays open

    return nothing
end
