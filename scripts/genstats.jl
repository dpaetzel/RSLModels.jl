using Base.Filesystem
using Comonicon
using Distributed
using MLJ
using MLJLinearModels
using ProgressMeter
using Serialization
using Statistics

using RSLModels.Intervals
using RSLModels.LocalModels
using RSLModels.Tasks
using RSLModels.Utils

include("../src/gentodisk.jl")

"""
Generate a single learning task and compute learning task statistics for it.
Then store the statistics together with the parameters required to reconstruct
the learning tasks to disk; or, if `--full` is given, store the full learning
task (i.e. train/test data etc.) to disk.

# Options

- `--d, -d`:
- `--nif, -n`:
- `--N, -N`:
- `--seed`:
- `--rate-coverage-min`:
- `--remove-final-fully-overlapped`:
- `--spread-min-factor`:
- `--full`: Whether to generate the full data (if not, only generate statistics;
            much faster due to much less IO esp. for many input space
            dimensions).
- `--prefix-fname`:
"""
@cast function gen(;
    d::Int=1,
    nif::Int=20,
    N::Int=200,
    seed::Int=0,
    rate_coverage_min::Float64=0.8,
    remove_final_fully_overlapped::Bool=false,
    full::Bool=false,
    prefix_fname::String="data/$d-$nif-$N-$seed-$rate_coverage_min-$remove_final_fully_overlapped",
)
    # TODO Consider to store model (but not generated data)

    gentodisk(;
        d=d,
        nif=nif,
        N=N,
        seed=seed,
        rate_coverage_min=rate_coverage_min,
        remove_final_fully_overlapped=remove_final_fully_overlapped,
        full=full,
        prefix_fname=prefix_fname,
    )

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
- `--full`: Whether to generate the full data (if not, only generate statistics;
            much faster due to much less IO esp. for many input space
            dimensions).
- `--prefix-fname`:
"""
@cast function genmany(;
    d::Int=1,
    nif::Int=3,
    N::Int=200,
    startseed::Int=0,
    endseed::Int=9,
    full::Bool=false,
    prefix_fname::String="data/$d-$nif-$N",
)
    println(
        "Warning: `genmany` may not be up to date " *
        "(look at code before using it!).",
    )
    for seed in startseed:endseed
        gen(;
            d=d,
            nif=nif,
            N=N,
            seed=seed,
            full=full,
            prefix_fname="$prefix_fname-$seed",
        )
    end
end

"""
Generate All The Tasks per seed in the given range.

# Options

- `--spread-min-factor`:
- `--startseed`:
- `--endseed`:
- `--full`: Whether to generate the full data (if not, only generate statistics;
            much faster due to much less IO esp. for many input space
            dimensions).
- `--prefix-fname`:
"""
@cast function genall(;
    startseed::Int=0,
    endseed::Int=9,
    full::Bool=false,
    prefix_fname::String="data/genstats/genall",
)
    # Start 1 additional workers.
    # addprocs(2; exeflags="--project")
    # Choose the number of workers via the `-p` parameter to Julia (probably `-p
    # auto` for as many workers as there are logical cores).
    println("Running on $(nworkers()) workers.")
    @everywhere include("../src/gentodisk.jl")

    remove_final_fully_overlapped = true

    # ProgressBar(
    # Note that we have to `collect` here since `@distributed for` seems not to
    # work with `Iterators.product` objects.
    iter = collect(
        enumerate(
            Iterators.product(
                startseed:endseed,
                [1, 2, 3, 5, 10],
                [2, 4, 7, 10, 15],
                [0.7, 0.8, 0.9, 0.95],
            ),
        ),
    )
    n_iter = length(iter)

    # Pattern from
    # https://github.com/timholy/ProgressMeter.jl/tree/master#tips-for-parallel-programming
    # (but fixed).
    prog = Progress(n_iter)
    channel = RemoteChannel(() -> Channel{Bool}())
    # Sync the two tasks at the very end.
    @sync begin
        # The first task updates the progress bar.
        @async while take!(channel)
            next!(prog)
        end

        # The second task does the computation.
        @async begin
            # Note that we have to add a `@sync` here since otherwise the
            # `false` is writen to the channel first.
            @sync @distributed for (i, (seed, d, nif, rate_coverage_min)) in
                                   iter
                N = Int(round(200 * 10^(d / 5)))
                gentodisk(;
                    d=d,
                    nif=nif,
                    N=N,
                    seed=seed,
                    rate_coverage_min=rate_coverage_min,
                    remove_final_fully_overlapped=remove_final_fully_overlapped,
                    full=full,
                    prefix_fname="$prefix_fname/$d-$nif-$N-$seed-$rate_coverage_min-$remove_final_fully_overlapped",
                )
                # Trigger a process bar update.
                put!(channel, true)
            end
            # Tell the progress bar task to finish.
            put!(channel, false)
        end
    end
end

@main
