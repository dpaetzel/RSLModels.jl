module Intervals

using Distributions
using LinearAlgebra
using Random
using StatsBase

export Interval,
    dimensions,
    draw_interval,
    draw_intervals,
    elemof,
    hull,
    intersection,
    mutate,
    plot_interval,
    volume

# We want to track global assignments as well when doing
# `includet(thisfile.jl)`.
__revise_mode__ = :evalassign

const X_MIN::Float64 = 0.0
const X_MAX::Float64 = 1.0

# TODO Consider to move this global constant somewhere
const dist_spread = Beta(1.55, 2.74)

struct Interval
    lbound::AbstractVector{Float64}
    ubound::AbstractVector{Float64}
    lopen::AbstractVector{Bool}
    uopen::AbstractVector{Bool}
    function Interval(
        lbound::AbstractVector{Float64},
        ubound::AbstractVector{Float64};
        lopen::AbstractVector{Bool}=repeat([false], length(lbound)),
        uopen::AbstractVector{Bool}=repeat([false], length(ubound)),
    )
        # Note that open intervals are allowed to have lower and upper bounds be
        # the same (which results in an empty set).
        return any(lbound .> ubound) ? error("out of order") :
               new(lbound, ubound, lopen, uopen)
    end
end

function elemof(x::AbstractVector{Float64}, interval::Interval)
    # This has been made enormously ugly by the fact that I need to allow for
    # closed/open options at the bound level.
    for dim in 1:dimensions(interval)
        # If the lower bound is truly greater than the value then we're always
        # false (independent of whether the interval is open or not).
        if interval.lbound[dim] > x[dim]
            return false
        end
        # If the interval is open, further check for equality.
        if interval.lopen[dim] && interval.lbound[dim] == x[dim]
            return false
        end

        if interval.ubound[dim] < x[dim]
            return false
        end
        if interval.uopen[dim] && interval.ubound[dim] == x[dim]
            return false
        end
    end
    return true
end

function elemof(X::AbstractMatrix{Float64}, interval::Interval)
    result = Array{Bool}(undef, size(X)[1])
    for n in 1:size(X)[1]
        result[n] = elemof(view(X, n, :), interval)
    end
    return result
end

function elemof(
    X::AbstractMatrix{Float64},
    intervals::AbstractVector{Interval},
)
    matching_matrix = Matrix{Bool}(undef, size(X)[1], length(intervals))
    for i in 1:length(intervals)
        matching_matrix[:, i] = elemof(X, intervals[i])
    end
    return matching_matrix
end

# The volume of input space.
function volume(dims::Integer; x_min=X_MIN, x_max=X_MAX)
    return (x_max - x_min)^dims
end

# The volume of the given interval.
function volume(interval::Interval)
    return prod(interval.ubound .- interval.lbound)
end

function dimensions(interval::Interval)
    return size(interval.lbound, 1)::Int64
end

"""
Given centers and spreads, compute the volume of the corresponding interval.
"""
function volume(; centers, spreads)
    # NOTE We enforce keyword arguments here so that users do not confuse
    # arguments of this with the in some cases more natural `lbound`/`ubound`
    # representation.
    interval = Interval(centers - spreads, centers + spreads)
    return volume(interval)
end

# NOTE Dimension should actually be unsigned but I don't see how to achieve that
# cleanly right now (i.e. without having to do `unsigned(10)` or `0xa` or
# similar).
function draw_spreads(
    dims::Integer,
    spread_min;
    spread_max=Inf,
    x_min=X_MIN,
    x_max=X_MAX,
)
    return draw_spreads(
        Random.default_rng(),
        dims,
        spread_min;
        spread_max=spread_max,
        x_min=x_min,
        x_max=x_max,
    )
end

function draw_spreads(
    rng::AbstractRNG,
    dims::Integer,
    spread_min;
    spread_max=Inf,
    x_min=X_MIN,
    x_max=X_MAX,
)
    if spread_min > spread_max
        throw(
            ArgumentError(
                "the condition `spread_min <= spread_max` is not satisfied",
            ),
        )
    end
    rates_spread = rand(rng, dist_spread, dims)
    spread_max = min((x_max - x_min) / 2, spread_max)
    return spread_min .+ rates_spread .* (spread_max - spread_min)
end

function draw_centers(spread; x_min=X_MIN, x_max=X_MAX)
    return draw_centers(Random.default_rng(), spread; x_min=x_min, x_max=x_max)
end

function draw_centers(rng::AbstractRNG, spread::Real; x_min=X_MIN, x_max=X_MAX)
    lbound = x_min + spread
    ubound = x_max - spread
    if lbound == ubound
        lbound
    elseif lbound >= ubound
        throw(
            ArgumentError(
                "the condition `spread <= (x_min - x_max) / 2` is not satisfied",
            ),
        )
    else
        rand(rng, Uniform(x_min + spread, x_max - spread))
    end
end

# NOTE We cannot write `Vector{Real}` here because in Julia it does NOT follow
# from `T1 <: T2` that `T{T1} <: T{T2}`. Instead, we write `Vector` for which we
# have `Vector{T} <: Vector` for all `T`.
function draw_centers(
    rng::AbstractRNG,
    spread::AbstractVector{Float64};
    x_min=X_MIN,
    x_max=X_MAX,
)
    # Simply draw an independent spread for each dimension.
    return [draw_centers(rng, s; x_min=x_min, x_max=x_max) for s in spread]
end

function draw_interval(
    dims::Integer,
    spread_min,
    volume_min;
    spread_max=Inf,
    x_min=X_MIN,
    x_max=X_MAX,
)
    return draw_interval(
        Random.default_rng(),
        dims,
        spread_min,
        volume_min;
        spread_max=spread_max,
        x_min=x_min,
        x_max=x_max,
    )
end

function draw_interval(
    rng::AbstractRNG,
    dims::Integer,
    spread_min,
    volume_min;
    spread_max=Inf,
    x_min=X_MIN,
    x_max=X_MAX,
)
    spreads = draw_spreads(
        rng,
        dims - 1,
        spread_min;
        spread_max=spread_max,
        x_min=x_min,
        x_max=x_max,
    )
    centers = draw_centers(rng, spreads)

    width_min = max(
        spread_min * 2,
        volume_min / volume(; centers=centers, spreads=spreads),
    )

    # TODO Put a check here for spread_max not conflicting with our “fix the last dimension thing”

    width_max = x_max - x_min

    iter_max = 20
    iter = 0
    # For high dimensions, we often have `width_min > width_max`. We re-draw the
    # smallest spread until `width_min` is small enough.
    while width_min > width_max && iter < iter_max
        iter += 1
        println(
            "Rejecting due to min width greater max width " *
            "($width_min > $width_max).",
        )
        # Start by getting the smallest spread's index.
        i = argmin(spreads)
        # Draw a new spread (and based on that, a center) and replace the
        # smallest spread (and the corresponding center).
        # TODO Consider to overload draw_spread so we don't need to extract here
        spreads[i] = draw_spreads(
            rng,
            1,
            spread_min;
            spread_max=spread_max,
            x_min=x_min,
            x_max=x_max,
        )[1]
        l = x_min + spreads[i]
        u = x_max - spreads[i]
        centers[i] = draw_centers(rng, spreads[i])[1]

        width_min = volume_min / volume(; centers=centers, spreads=spreads)
    end

    if iter >= iter_max
        error("Had to reject too many generated intervals, aborting.")
    end

    # Finally, we may draw a random width for the last dimension.
    width_last = if width_min == width_max
        width_min
        # At this point we can be sure that width_min < width_max.
    else
        rand(rng, Uniform(width_min, width_max))
    end

    # Compute the spread of the last dimension.
    spread_last = width_last / 2

    # Draw the center for the last dimension.
    center_last = draw_centers(rng, spread_last)

    # Insert the last center and spread at a random index so that there's no
    # weird bias towards the last dimension.
    i_rand = rand(rng, 1:dims)
    insert!(spreads, i_rand, spread_last)
    insert!(centers, i_rand, center_last)

    return Interval(centers - spreads, centers + spreads)
end

function draw_intervals(
    dims::Integer,
    n_intervals;
    spread_min=spread_ideal_cubes(dims, n_intervals),
    spread_max=Inf,
    x_min=X_MIN,
    x_max=X_MAX,
    volume_min=Intervals.volume_min_factor(
        dims,
        n_intervals;
        x_min=x_min,
        x_max=x_max,
    ),
)
    return draw_intervals(
        Random.default_rng(),
        dims,
        n_intervals;
        spread_min=spread_min,
        spread_max=spread_max,
        x_min=x_min,
        x_max=x_max,
        volume_min=volume_min,
    )
end

"""
Parameters
----------
dims : int > 0
n_intervals : int > 0
volume_min : float > 0
random_state : np.random.RandomState

Returns
-------
array, list, list
    The intervals as an array of shape `(n_intervals, 2, dims)`, the
    set of pair-wise intersections between the intervals, the set of volumes
    of the non-empty ones of these pair-wise intersections.
"""
function draw_intervals(
    rng::AbstractRNG,
    dims::Integer,
    n_intervals;
    spread_min=spread_ideal_cubes(dims, n_intervals),
    spread_max=Inf,
    x_min=X_MIN,
    x_max=X_MAX,
    volume_min=Intervals.volume_min_factor(
        dims,
        n_intervals;
        x_min=x_min,
        x_max=x_max,
    ),
)
    return [
        draw_interval(
            rng,
            dims,
            spread_min,
            volume_min;
            spread_max=spread_max,
            x_min=x_min,
            x_max=x_max,
        ) for _ in 1:n_intervals
    ]
end

"""
Given a number of intervals and an input space dimensions, compute the spread
that a cube would have with a volume of `1/n_intervals` of input space volume.
"""
function spread_ideal_cubes(
    dims::Integer,
    n_intervals;
    x_min=X_MIN,
    x_max=X_MAX,
)
    volume_avg = (x_max - x_min)^dims / n_intervals
    return volume_avg^(1.0 / dims) / 2.0
end

# TODO Consider to factor in the number of training data points here
"""
Given a number of intervals and input space dimensions, compute the minimum
volume as `factor` of `1/n_intervals` of input space volume.
"""
function volume_min_factor(
    dims::Integer,
    n_intervals,
    factor=0.1;
    x_min=X_MIN,
    x_max=X_MAX,
)
    volume_input_space = (x_max - x_min)^dims
    return factor * volume_input_space / n_intervals
end

"""
Computes the intersection of the two intervals. If the intersection is empty,
return `nothing`.
"""
function intersection(interval1::Interval, interval2::Interval)
    l = max.(interval1.lbound, interval2.lbound)
    u = min.(interval1.ubound, interval2.ubound)

    if any(u .< l)
        return nothing
    else
        return Interval(l, u)
    end
end

"""
The convex hull of the two intervals.
"""
function hull(interval1, interval2)
    lbounds = [interval1.lbound interval2.lbound]
    ubounds = [interval1.ubound interval2.ubound]
    lopens = [interval1.lopen interval2.lopen]
    uopens = [interval1.uopen interval2.uopen]

    idx_lbound = argmin(lbounds; dims=2)
    idx_ubound = argmax(ubounds; dims=2)

    lbound = reshape(lbounds[idx_lbound], 2)
    ubound = reshape(ubounds[idx_ubound], 2)
    lopen = reshape(lopens[idx_lbound], 2)
    uopen = reshape(uopens[idx_ubound], 2)

    return Interval(lbound, ubound; lopen=lopen, uopen=uopen)
end

"""
Uses the default RNG to mutate the lower and upper bounds of the given interval
at random (see code).
"""
function mutate(interval::Interval; factor=0.1, x_min=X_MIN, x_max=X_MAX)
    lbound, ubound = interval.lbound, interval.ubound
    lbound_new =
        lbound .+
        (rand(size(lbound, 1)) .* factor .- factor) .* (X_MAX - X_MIN)
    ubound_new =
        ubound .+
        (rand(size(ubound, 1)) .* factor .- factor) .* (X_MAX - X_MIN)
    lbound_new = clamp.(lbound_new, X_MIN, X_MAX)
    ubound_new = clamp.(ubound_new, X_MIN, X_MAX)
    return Interval(
        lbound_new,
        ubound_new;
        lopen=interval.lopen,
        uopen=interval.uopen,
    )
end

"""
Uses the default RNG to mutate the lower and upper bounds of `n` (randomly
drawn) of the given intervals at random (see code).
"""
function mutate(
    intervals::AbstractVector{Interval},
    n::Integer;
    factor=0.1,
    x_min=X_MIN,
    x_max=X_MAX,
)
    intervals_changed = deepcopy(intervals)
    idx = sample(1:size(intervals_changed, 1), n; replace=false)
    intervals_changed[idx] .= mutate.(intervals_changed[idx]; factor=factor)
    return intervals_changed
end

end
