using Comonicon
using Dates
using Distributed
using Distributions
using ProgressMeter

using RSLModels.Intervals
using RSLModels.Utils

DXs_default::Vector{Int} = [2, 3, 5, 8, 12]
rates_coverage_min::Vector{Float64} = [0.75, 0.9]

"""
Sample the condition set–drawing process many times and write statistics (most
importantly, number of rules and coverage) to a CSV file (name generated based
on current time) which can later be used to optimize the parameters of said
process.

# Args

- `DXs`: Input space dimensions to sample for.

# Options

- `--n-iter`: Number of times to sample.
- `--usemmap`: Whether to memory-map large arrays (X, y, matching matrices, …)
  to disk and save RAM that way.
"""
@main function mysample(
    DXs::Int...;
    n_iter::Int=100,
    usemmap::Bool=false,
    rates_coverage_min=rates_coverage_min,
)
    if isempty(DXs)
        println("No DXs given, using default $DXs_default …")
        DXs = DXs_default
    end

    @everywhere include((@__DIR__) * "/_genkdata.jl")

    iter = collect(Iterators.product(1:n_iter, DXs, rates_coverage_min))

    # Pattern from
    # https://github.com/timholy/ProgressMeter.jl/tree/master#tips-for-parallel-programming
    # (but fixed).
    prog = Progress(length(iter))
    channel = RemoteChannel(() -> Channel{Bool}())
    channel_out = RemoteChannel(() -> Channel{Tuple{Int64,Any}}())

    # Open a file in append mode, include a short random ID to keep the file
    # name readable while decreasing the possibility of collisions in the
    # filename (Julia 1.9.3 does not support a `prefix` parameter for `mktemp`
    # but only for `mktempdir` and I want a .csv file during running already).
    dtime = Dates.format(now(), "yyyy-mm-dd-HH-MM-SS-sss")
    idrand = join(rand(vec(['a':'z'; 'A':'Z'; '0':'9']), 4))
    fname_part = "$dtime-$idrand-kdata.csv.part"
    fname = "$dtime-$idrand-kdata.csv"
    fhandle = open(fname_part, "a")
    println("Opened $fname_part for writing …")

    @sync begin
        # The first task updates the progress bar and collects the results.
        @async while take!(channel)
            (i, out) = take!(channel_out)

            line = join(
                [
                    out[:DX],
                    out[:rate_coverage_min],
                    out[:spread_min],
                    out[:a],
                    out[:b],
                    out[:rate_coverage],
                    out[:K],
                ],
                ",",
            )
            println(fhandle, line)

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
            @sync @distributed for (i, DX, rate_coverage_min) in iter
                out = mysample1(
                    DX,
                    rate_coverage_min;
                    usemmap=usemmap,
                    n_intervals_max=100,
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
    try
        mv(fname_part, fname)
        println("Successfully moved $fname_part to $fname.")
    catch e
        if isa(e, ArgumentError)
            println(e)
            println(
                "Consider to simply rename the .part file to a not-yet-used name.",
            )
        else
            rethrow(e)
        end
    end

    return nothing
end
