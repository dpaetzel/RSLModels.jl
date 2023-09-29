using Base.Filesystem
using Comonicon

using RSLModels.Tasks

"""
Generate a single task and sample it, then write the task and the sample to
disk.

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

    mkpath(dirname(prefix_fname))
    save(prefix_fname, task)

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

@main
