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
    maxgeneral,
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

function Base.clamp(x::Interval, lo::Float64, hi::Float64)
    return Interval(
        Base.clamp.(x.lbound, lo, hi),
        Base.clamp.(x.ubound, lo, hi),
    )
end

function elemof(x, interval::Nothing)
    return false
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

function volume(nothing)
    return 0
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

"""
The maximally general interval for the given dimensions.
"""
function maxgeneral(dims::Integer; x_min=X_MIN, x_max=X_MAX)
    return Interval(repeat([x_min], dims), repeat([x_max], dims))
end

# NOTE Dimension should actually be unsigned but I don't see how to achieve that
# cleanly right now (i.e. without having to do `unsigned(10)` or `0xa` or
# similar).
function draw_spread(
    dims::Integer,
    spread_min;
    spread_max=Inf,
    x_min=X_MIN,
    x_max=X_MAX,
    uniform=false,
)
    return draw_spread(
        Random.default_rng(),
        dims,
        spread_min;
        spread_max=spread_max,
        x_min=x_min,
        x_max=x_max,
        uniform=uniform,
    )
end

function draw_spread(
    rng::AbstractRNG,
    dims::Integer,
    spread_min;
    spread_max=Inf,
    x_min=X_MIN,
    x_max=X_MAX,
    uniform=false,
)
    if spread_min > spread_max
        throw(
            ArgumentError(
                "the condition `spread_min <= spread_max` is not satisfied",
            ),
        )
    end
    if uniform
        rates_spread = rand(rng, dims)
    else
        rates_spread = rand(rng, dist_spread, dims)
    end
    spread_max = min((x_max - x_min) / 2, spread_max)
    return spread_min .+ rates_spread .* (spread_max - spread_min)
end

"""
Given a data point `x` and a spread (and space boundaries), the lower bound of
the range of viable interval centers such that the interval [center - spread,
center + spread] does not violate the space constraints and contains `x`.
"""
function lbound_center(x, spread; x_min=Intervals.X_MIN, x_max=Intervals.X_MAX)
    return max(x - spread, x_min + spread)
end

"""
Given a data point `x` and a spread (and space boundaries), the upper bound of
the range of viable interval centers such that the interval [center - spread,
center + spread] does not violate the space constraints and contains `x`.
"""
function ubound_center(x, spread; x_min=Intervals.X_MIN, x_max=Intervals.X_MAX)
    return min(x + spread, x_max - spread)
end

function draw_center(x, spread; x_min=X_MIN, x_max=X_MAX)
    return draw_center(
        Random.default_rng(),
        x,
        spread;
        x_min=x_min,
        x_max=x_max,
    )
end

function draw_center(
    rng::AbstractRNG,
    x::Float64,
    spread::Float64;
    x_min=X_MIN,
    x_max=X_MAX,
)
    # The spread defines an interval around `x` from which we can draw a center
    # such that `x` is still matched.
    lb_center = lbound_center(x, spread; x_min=x_min, x_max=x_max)
    ub_center = ubound_center(x, spread; x_min=x_min, x_max=x_max)

    # We next draw a rate.
    rate_center = rand(rng)

    # Then we convert this rate into a center.
    return lb_center + rate_center * (ub_center - lb_center)
end

# NOTE We cannot write `Vector{Real}` here because in Julia it does NOT follow
# from `T1 <: T2` that `T{T1} <: T{T2}`. Instead, we write `Vector` for which we
# have `Vector{T} <: Vector` for all `T`.
function draw_center(
    rng::AbstractRNG,
    x::AbstractVector{Float64},
    spread::AbstractVector{Float64};
    x_min=X_MIN,
    x_max=X_MAX,
)
    # Simply draw an independent spread for each dimension.
    return draw_center.(rng, x, spread; x_min=x_min, x_max=x_max)
end

function draw_interval(
    x::Vector{Float64},
    spread_min;
    spread_max=Inf,
    uniform_spread=true,
    x_min=X_MIN,
    x_max=X_MAX,
)
    return draw_interval(
        Random.default_rng(),
        x,
        spread_min;
        spread_max=spread_max,
        uniform_spread=uniform_spread,
        x_min=x_min,
        x_max=x_max,
    )
end

function draw_interval(
    rng::AbstractRNG,
    x::Vector{Float64},
    spread_min;
    spread_max=Inf,
    uniform_spread=true,
    x_min=X_MIN,
    x_max=X_MAX,
)
    dims = size(x, 1)
    spread = draw_spread(
        rng,
        dims,
        spread_min;
        spread_max=spread_max,
        uniform=uniform_spread,
        x_min=x_min,
        x_max=x_max,
    )
    center = draw_center(rng, x, spread; x_min=x_min, x_max=x_max)

    return Interval(center - spread, center + spread)
end

function draw_intervals(
    dims::Integer;
    n_intervals_fantasy=20,
    spread_min::Float64=Intervals.spread_ideal_cubes(
        dims,
        n_intervals_fantasy,
    ),
    spread_max::Float64=Inf,
    uniform_spread::Bool=true,
    rate_coverage_min::Float64=0.8,
    n::Int=Int(round(200 * 10^(dims / 5))),
    x_min=Intervals.X_MIN,
    x_max=Intervals.X_MAX,
)
    return draw_intervals(
        Random.default_rng(),
        dims;
        n_intervals_fantasy=n_intervals_fantasy,
        spread_min=spread_min,
        spread_max=spread_max,
        uniform_spread=uniform_spread,
        rate_coverage_min=rate_coverage_min,
        n=n,
        x_min=x_min,
        x_max=x_max,
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
    dims::Int;
    n_intervals_fantasy=20,
    spread_min::Float64=Intervals.spread_ideal_cubes(
        dims,
        n_intervals_fantasy,
    ),
    spread_max::Float64=Inf,
    uniform_spread::Bool=true,
    rate_coverage_min::Float64=0.8,
    n::Int=Int(round(200 * 10^(dims / 5))),
    x_min=Intervals.X_MIN,
    x_max=Intervals.X_MAX,
)
    X = rand(rng, n, dims) .* (x_max - x_min)
    matched = fill(false, n)
    intervals::Vector{Interval} = []

    while count(matched) / n < rate_coverage_min
        idx = rand(rng, 1:n)
        x = X[idx, :]

        interval = draw_interval(
            rng,
            x,
            spread_min;
            spread_max=spread_max,
            uniform_spread=uniform_spread,
            x_min=x_min,
            x_max=x_max,
        )

        push!(intervals, interval)

        # Should be faster (due to short-circuiting) than calling `elemof` on
        # `X`.
        for i in 1:n
            matched[i] = matched[i] || elemof(X[i, :], interval)
        end
    end

    return intervals
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

    dims = dimensions(interval1)
    lbound = reshape(lbounds[idx_lbound], dims)
    ubound = reshape(ubounds[idx_ubound], dims)
    lopen = reshape(lopens[idx_lbound], dims)
    uopen = reshape(uopens[idx_ubound], dims)

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
    lbound_new = min.(lbound_new, ubound_new)
    ubound_new = max.(lbound_new, ubound_new)
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

"""
Mutate the lower and upper bounds of the intervals at the given indices at
random (see code).
"""
function mutate(
    intervals::AbstractVector{Interval},
    indices::AbstractVector;
    factor=0.1,
    x_min=X_MIN,
    x_max=X_MAX,
)
    intervals_changed = deepcopy(intervals)
    idx = indices
    intervals_changed[idx] .= mutate.(intervals_changed[idx]; factor=factor)
    return intervals_changed
end

end
