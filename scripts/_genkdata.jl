# This file is sent to all workers by genkdata.jl (which is itself a Comonicon
# CLI app and cannot be sent to all workers straightforwardly).
using Distributions

using RSLModels.Intervals

DXs = [2, 3, 5, 8, 10, 13]
rates_coverage_min = [0.75, 0.9]

function mysample1(;
    n_samples=21,
    DXs=DXs,
    rates_coverage_min=rates_coverage_min,
    usemmap=false,
)
    DX = rand(DXs)
    rate_coverage_min = rand(rates_coverage_min)

    # TODO Consider to use a slightly informed distribution for `spread_min`.
    # spread_min = hist(rand(Beta(2, 2), 10000) / 2; bins=30)
    spread_min = rand() / 2

    # TODO Consider to invest time in choosing sensible ranges here
    a = rand() * 99 + 1
    b = rand() * 99 + 1

    rates_coverage = Vector{Float64}(undef, n_samples)
    Ks = Vector{Int}(undef, n_samples)

    for i in 1:n_samples
        rates_coverage[i], intervals = draw_intervals(
            DX;
            rate_coverage_min=rate_coverage_min,
            params_spread=(a, b),
            spread_min=spread_min,
            usemmap=usemmap,
            n_intervals_max=50,
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
