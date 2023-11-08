using Comonicon
using Dates
using Distributed
using Distributions
using ProgressMeter

using RSLModels.Intervals
using RSLModels.Utils

DXs = [1, 2, 3, 5, 10, 15]
Ks = [5, 10, 15, 20, 25, 30, 35, 40]
rates_coverage_min = [0.7, 0.8, 0.9]

"""
Sample the condition set–drawing process many times and write statistics (most
importantly, number of rules and coverage) to a CSV file (name generated based
on current time) which can later be used to optimize the parameters of said
process.

# Args

- `n_iter`: Number of times to sample.

# Options

- `--n-samples`: Number of samples to draw each time (to later be able to
  estimate distribution statistics such as mean).
"""
@main function mysample(
    n_iter::Int;
    n_samples::Int=21,
    DXs=DXs,
    rates_coverage_min=rates_coverage_min,
)
    @everywhere include((@__DIR__) * "/_genkdata.jl")

    # Pattern from
    # https://github.com/timholy/ProgressMeter.jl/tree/master#tips-for-parallel-programming
    # (but fixed).
    prog = Progress(n_iter)
    channel = RemoteChannel(() -> Channel{Bool}())
    channel_out = RemoteChannel(() -> Channel{Tuple{Int64,Any}}())

    # Open a file in append mode.
    dtime = Dates.format(now(), "yyyy-mm-dd-HH-MM-SS")
    fhandle = open("$dtime-kdata.csv", "a")

    @sync begin
        # The first task updates the progress bar and collects the results.
        @async while take!(channel)
            (i, out) = take!(channel_out)

            for (rate_coverage, K) in zip(out[:rates_coverage], out[:Ks])
                line = join(
                    [
                        out[:DX],
                        out[:rate_coverage_min],
                        out[:spread_min],
                        out[:a],
                        out[:b],
                        rate_coverage,
                        K,
                    ],
                    ",",
                )
                println(fhandle, line)
            end

            # Flush the file every 5th time or so.
            if rand() < 0.2
                flush(fhandle)
            end

            next!(prog)
        end

        # The second task does the computation.
        @async begin
            # Note that we have to add a `@sync` here since otherwise the
            # `false` is written to the channel first.
            @sync @distributed for i in 1:n_iter
                out = mysample1(;
                    n_samples=n_samples,
                    DXs=DXs,
                    rates_coverage_min=rates_coverage_min,
                    usemmap=true,
                )

                # Trigger process bar update and result fetching.
                put!(channel, true)

                # Push result.
                put!(channel_out, (i, out))
            end
            # Tell the progress bar and result fetching task to finish.
            put!(channel, false)
        end
    end

    close(fhandle)

    return nothing
end
