using Base.Filesystem
using Comonicon
using ProgressBars
using Serialization
using Statistics

using RSLModels.LocalModels
using RSLModels.Tasks
using RSLModels.Utils

"""
Generate a single learning task and compute learning task statistics for it.
Then store the statistics together with the parameters required to reconstruct
the learning tasks to disk.

# Options

- `--d, -d`:
- `--k, -k`:
- `--n, -n`:
- `--seed`:
- `--prefix-fname`:
"""
@cast function gen(;
    d::Int=1,
    k::Int=3,
    n::Int=200,
    seed::Int=0,
    prefix_fname::String="data/$d-$k-$n-$seed",
    pbar=missing,
)
    if ismissing(pbar)
        myprintln = println
    else
        myprintln = s -> println(pbar, s)
    end

    params = Dict(:d => d, :k => k, :seed => seed)
    task = generate(d, k, n; seed=seed)
    match_X = task.match_X

    y_pred = output_mean(task.model, task.X)
    y_test_pred = output_mean(task.model, task.X_test)
    stats = Dict(
        # TODO Consider to also add linearity
        :coverage => data_coverage(match_X),
        :overlap => data_overlap_pairs_mean(match_X),
        :mae_train => mean(abs.(task.y .- y_pred)),
        :mae_test => mean(abs.(task.y_test .- y_test_pred)),
    )

    data = Dict(:params => params, :stats => stats)

    # TODO Mlflow here
    myprintln("$d $k $n $seed => $stats")

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
- `--k, -k`:
- `--n, -n`:
- `--startseed`:
- `--endseed`:
- `--prefix-fname`:
"""
@cast function genmany(;
    d::Int=1,
    k::Int=3,
    n::Int=200,
    startseed::Int=0,
    endseed::Int=9,
    prefix_fname::String="data/$d-$k-$n",
)
    for seed in startseed:endseed
        gen(; d=d, k=k, n=n, seed=seed, prefix_fname="$prefix_fname-$seed")
    end
end

"""
Generate All The Tasks per seed in the given range and sample them, then write
the task and the sample to disk.

# Options

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
            [1, 2, 3, 5, 10, 20],
            [2, 4, 7, 10, 15, 20],
        ),
    )
    for (seed, d, k) in iter
        n = Int(round(200 * 10^(d / 5)))
        gen(;
            d=d,
            k=k,
            n=n,
            seed=seed,
            prefix_fname="$prefix_fname/$d-$k-$n-$seed",
            pbar=iter,
        )
    end
end

@main
