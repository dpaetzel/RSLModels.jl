"""
Create an initializer that will generate a single random solution by doing these
steps upon being called:

- Draw a random row from `params_dist`.
- Draw a solution from the random distribution specified by the row.

This can be used to approximate an initialization scheme where the user chooses
a set of solution lengths and then we repeatedly draw one of these lengths at
random and create an individual of that length until the configured population
size is reached. The only difference is that `init1_inverse` probably does not
fully hit the chosen lengths due to the probabilistic nature of generating a
random solution based on minimum coverage termination criterion.
"""
function mkinit1_inverse(
    x_min::Float64,
    x_max::Float64,
    params_dist::DataFrame,
)
    function _init1(rng, X)
        N, DX = size(X)
        params_dist = subset(params_dist, :DX => dx -> dx .== DX)
        row = sample(eachrow(params_dist))
        return draw_genotype(
            rng,
            # Draw intervals using the training data for checking the coverage
            # criterion. (An alternative would be to supply `DX` here which
            # would result in `draw_genotype` drawing a random sample from the
            # input space and trying to fulfil the coverage criterion on that.)
            X;
            spread_min=row.spread_min,
            params_spread=(a=row.a, b=row.b),
            rate_coverage_min=row.rate_coverage_min,
            remove_final_fully_overlapped=true,
            x_min=x_min,
            x_max=x_max,
        )
    end

    return _init1
end

"""
Create an initializer that will generate a single random solution by drawing it
from the distribution defined by the `spread_*`, `params_spread_*` and
`rate_coverage_min` parameters.

Note that it can be difficult to control solution length this way (see
`mkinit1_inverse` for a way to deal with this).
"""
function mkinit1_custom(
    x_min::Float64,
    x_max::Float64,
    spread_min::Float64,
    spread_max::Float64,
    params_spread_a::Float64,
    params_spread_b::Float64,
    rate_coverage_min::Float64,
)
    function _init1(rng, X)
        return draw_genotype(
            rng,
            # Draw intervals using the training data for checking the coverage
            # criterion. (An alternative would be to supply `DX` here which
            # would result in `draw_genotype` drawing a random sample from the
            # input space and trying to fulfil the coverage criterion on that.)
            X;
            spread_min=spread_min,
            spread_max=spread_max,
            params_spread=(a=params_spread_a, b=params_spread_b),
            rate_coverage_min=rate_coverage_min,
            # We always remove fully overlapped rules during initialization.
            remove_final_fully_overlapped=true,
            # TODO Consider to reduce this number for faster initialization
            # n_samples=Parameters.n(dims),
            x_min=x_min,
            x_max=x_max,
        )
    end
    return _init1
end

"""
Applies the `init1` function (should probably be an initializer such as the one
created by `mkinit1_*`) `size_pop` times to `rng` and `X` and collects the
results in a vector.
"""
function init(rng::AbstractRNG, X::XType, init1::Function, size_pop::Int)
    pop::Vector{Genotype} = [init1(rng, X) for _ in 1:(size_pop)]
    report = (;)
    return pop, report
end
