# This file is sent to all workers by genkdata.jl (which is itself a Comonicon
# CLI app and cannot be sent to all workers straightforwardly).
using Distributions

using RSLModels.Intervals

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
