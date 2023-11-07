using Dates
using Distributed
using Distributions
using ProgressMeter

using RSLModels.Intervals
using RSLModels.Utils

# `includet(thisfile.jl)`.
__revise_mode__ = :evalassign

DXs = [1, 2, 3, 5, 10, 15]
Ks = [5, 10, 15, 20, 25, 30, 35, 40]
rates_coverage_min = [0.7, 0.8, 0.9]

function mysample1(;
    n_samples=21,
    DXs=DXs,
    rates_coverage_min=rates_coverage_min,
)
    DX = rand(DXs)
    rate_coverage_min = rand(rates_coverage_min)

    # TODO Consider to use a slightly informed distribution for `spread_min`.
    # spread_min = hist(rand(Beta(2, 2), 10000) / 2; bins=30)
    spread_min = rand() / 2

    # TODO Consider to invest time in choosing sensible ranges here
    a = rand() * 200 + 1
    b = rand() * 200 + 1

    rates_coverage = Vector{Float64}(undef, n_samples)
    Ks = Vector{Int}(undef, n_samples)

    for i in 1:n_samples
        rates_coverage[i], intervals = draw_intervals(
            DX;
            rate_coverage_min=rate_coverage_min,
            params_spread=(a, b),
            spread_min=spread_min,
            n_intervals_max=100,
            return_coverage_rate=true,
            # verbose=10,
        )

        Ks[i] = length(intervals)
    end

    return Dict(
        :DX => DX,
        :rate_coverage_min => rate_coverage_min,
        :spread_min => spread_min,
        :a => a,
        :b => b,
        :rates_coverage => rates_coverage,
        :Ks => Ks,
    )
end

function mysample(n_iter; DXs=DXs, rates_coverage_min=rates_coverage_min)
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
                out =
                    mysample1(; DXs=DXs, rates_coverage_min=rates_coverage_min)

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
