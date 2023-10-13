using Base.Filesystem
using Comonicon
using MLJ
using MLJLinearModels
using ProgressBars
using Serialization
using Statistics

using RSLModels.Intervals
using RSLModels.LocalModels
using RSLModels.Tasks
using RSLModels.Utils

RidgeRegressor = @load RidgeRegressor pkg = MLJLinearModels

"""
Generate a single learning task and compute learning task statistics for it.
Then store the statistics together with the parameters required to reconstruct
the learning tasks to disk.

# Options

- `--d, -d`:
- `--nif, -n`:
- `--N, -N`:
- `--seed`:
- `--spread-min-factor`:
- `--prefix-fname`:
"""
@cast function gen(;
    d::Int=1,
    nif::Int=20,
    N::Int=200,
    seed::Int=0,
    prefix_fname::String="data/$d-$N-$nif-$seed",
    pbar=missing,
)
    if ismissing(pbar)
        myprintln = println
    else
        myprintln = s -> println(pbar, s)
    end

    params = Dict(:d => d, :N => N, :nif => nif, :seed => seed)
    task = generate(d, nif, N; seed=seed)
    match_X = task.match_X

    y_pred = output_mean(task.model, task.X)
    y_test_pred = output_mean(task.model, task.X_test)

    mach = machine(RidgeRegressor(), MLJ.table(task.X), task.y)
    fit!(mach)
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

    data = Dict(:params => params, :stats => stats)

    # TODO Mlflow here
    myprintln("$d $nif $N $seed => $stats")

    mkpath(dirname(prefix_fname))
    serialize("$prefix_fname.jls", data)

    # TODO Consider to store model (but not generated data)

    return nothing
end

"""
Generate one task per seed in the given range, sample it, then write the task
and the sample to disk.

# Options

- `--d, -d`:
- `--nif, -n`:
- `--N, -N`:
- `--spread-min-factor`:
- `--startseed`:
- `--endseed`:
- `--prefix-fname`:
"""
@cast function genmany(;
    d::Int=1,
    nif::Int=3,
    N::Int=200,
    startseed::Int=0,
    endseed::Int=9,
    prefix_fname::String="data/$d-$nif-$N",
)
    for seed in startseed:endseed
        gen(; d=d, nif=nif, N=N, seed=seed, prefix_fname="$prefix_fname-$seed")
    end
end

"""
Generate All The Tasks per seed in the given range and sample them, then write
the task and the sample to disk.

# Options

- `--spread-min-factor`:
- `--startseed`:
- `--endseed`:
- `--prefix-fname`:
"""
@cast function genall(;
    startseed::Int=0,
    endseed::Int=9,
    prefix_fname::String="data/genstats/genall",
)
    iter = ProgressBar(
        Iterators.product(
            startseed:endseed,
            [1, 2, 3, 5, 10],
            [2, 4, 7, 10, 15],
        ),
    )
    for (seed, d, nif) in iter
        N = Int(round(200 * 10^(d / 5)))
        gen(;
            d=d,
            nif=nif,
            N=N,
            seed=seed,
            prefix_fname="$prefix_fname/$d-$nif-$N-$seed",
            pbar=iter,
        )
    end
end

@main
