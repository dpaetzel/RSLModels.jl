using Base.Filesystem
using Comonicon
using ProgressBars

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
)
    task = generate(d, k, n; seed=seed)
    match_X = task.match_X

    stats = Dict(
        :coverage => data_coverage(match_X),
        :overlap => data_overlap_pairs_mean(match_X),
    )

    # TODO Mlflow here
    println("$d $k $n $seed => $stats")

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
@cast function genall(; startseed::Int=0, endseed::Int=9)
    for (seed, d, k) in ProgressBar(
        Iterators.product(
            startseed:endseed,
            [1, 2, 3, 5, 10, 20],
            [2, 4, 7, 10, 15, 20],
        ),
    )
        n = Int(round(200 * 10^(d / 5)))
        gen(;
            d=d,
            k=k,
            n=n,
            seed=seed,
            prefix_fname="data/genall/$d-$k-$n-$seed",
        )
    end
end

@main
