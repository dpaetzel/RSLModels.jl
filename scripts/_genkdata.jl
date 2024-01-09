# This file is sent to all workers by genkdata.jl (which is itself a Comonicon
# CLI app and cannot be sent to all workers straightforwardly).
using Distributions
using Random

using RSLModels.Intervals

function mysample1(DX, rate_coverage_min; usemmap=false, n_intervals_max=100)
    return mysample1(
        Random.default_rng(),
        DX,
        rate_coverage_min;
        usemmap=usemmap,
        n_intervals_max=n_intervals_max,
    )
end

function mysample1(
    rng::AbstractRNG,
    DX,
    rate_coverage_min;
    usemmap=false,
    n_intervals_max=100,
)
    # TODO Consider to use a slightly informed distribution for `spread_min`.
    # spread_min = hist(rand(Beta(2, 2), 10000) / 2; bins=30)
    spread_min = rand(rng) / 2

    # TODO Consider to invest time in choosing sensible ranges here
    a = rand(rng) * 199 + 1
    b = rand(rng) * 199 + 1

    rate_coverage, intervals = draw_intervals(
        rng,
        DX;
        rate_coverage_min=rate_coverage_min,
        params_spread=(a=a, b=b),
        spread_min=spread_min,
        usemmap=usemmap,
        n_intervals_max=n_intervals_max,
        return_coverage_rate=true,
        # verbose=10,
    )

    K = length(intervals)

    return Dict(
        :DX => DX,
        :rate_coverage_min => rate_coverage_min,
        :spread_min => spread_min,
        :a => a,
        :b => b,
        :rate_coverage => rate_coverage,
        :K => K,
    )
end
