module Intervals

using AutoHashEquals
using Distributions
using LinearAlgebra
using Mmap
using Random
using StatsBase

using RSLModels.Parameters

export Interval,
    dimensions,
    draw_interval,
    draw_intervals,
    elemof,
    hull,
    intersection,
    maxgeneral,
    mutate,
    remove_fully_overlapped,
    volume

# We want to track global assignments as well when doing
# `includet(thisfile.jl)`.
__revise_mode__ = :evalassign

const X_MIN::Float64 = 0.0
const X_MAX::Float64 = 1.0

# TODO Consider to move this global constant somewhere
const dist_spread = Beta(1.55, 2.74)

@auto_hash_equals struct Interval
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

function elemof(X::AbstractMatrix{Float64}, interval::Interval; usemmap=false)
    N = size(X, 1)
    if usemmap
        (path_out, io_out) = mktemp(tempdir())
        out = mmap(io_out, Vector{Bool}, N)
    else
        out = Array{Bool}(undef, N)
    end
    for n in 1:N
        out[n] = elemof(view(X, n, :), interval)
    end
    return out
end

function elemof(
    X::AbstractMatrix{Float64},
    intervals::AbstractVector{Interval};
    usemmap=false,
)
    N = size(X, 1)
    K = length(intervals)

    (path_out, io_out) = mktemp(tempdir())
    out = mmap(io_out, Matrix{Bool}, (N, K))

    # Outer loop should go over columns and inner loop over rows. See
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-column-major
    for k in 1:K
        for n in 1:N
            out[n, k] = elemof(view(X, n, :), intervals[k])
        end
    end
    return out
end

function volume(nothing)
    return 0.0
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
    x::AbstractVector{Float64},
    spread_min::Float64;
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
    x::AbstractVector{Float64},
    spread_min::Float64;
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
    nif=20,
    spread_min::Float64=Intervals.spread_ideal_cubes(dims, nif),
    spread_max::Float64=Inf,
    uniform_spread::Bool=true,
    rate_coverage_min::Float64=0.8,
    n_samples::Int=Parameters.n(dims),
    remove_final_fully_overlapped::Bool=true,
    x_min=Intervals.X_MIN,
    x_max=Intervals.X_MAX,
)
    return draw_intervals(
        Random.default_rng(),
        dims;
        nif=nif,
        spread_min=spread_min,
        spread_max=spread_max,
        uniform_spread=uniform_spread,
        rate_coverage_min=rate_coverage_min,
        n_samples=n_samples,
        remove_final_fully_overlapped=remove_final_fully_overlapped,
        x_min=x_min,
        x_max=x_max,
    )
end

"""
    draw_intervals([rng,] dims; <keyword arguments>)

Generate random intervals for `dim` dimensions.

 If no `rng` is provided, use the
`Random.default_rng()`.

# Arguments
- `rng::AbstractRNG`:
- `dims::Int`:
- `nif`: Stands for `Number of Intervals in my Fantasy`. Reminds me of Nifflers,
  which are cute.
- `spread_min::Float64`:
- `spread_max::Float64`:
- `uniform_spread::Bool`:
- `rate_coverage_min::Float64`: [0, 1].
- `n_samples`:
- `remove_final_fully_overlapped`:
- `x_min::Float64`:
- `x_max::Float64`:
"""
function draw_intervals(
    rng::AbstractRNG,
    dims::Int;
    nif=20,
    spread_min::Float64=Intervals.spread_ideal_cubes(dims, nif),
    spread_max::Float64=Inf,
    uniform_spread::Bool=true,
    rate_coverage_min::Float64=0.8,
    n_samples::Int=Parameters.n(dims),
    remove_final_fully_overlapped::Bool=true,
    x_min=Intervals.X_MIN,
    x_max=Intervals.X_MAX,
)
    X = rand(rng, n_samples, dims) .* (x_max - x_min)
    M = []
    matched = fill(false, n_samples)
    intervals::Vector{Interval} = []

    while count(matched) / n_samples < rate_coverage_min
        idx = rand(rng, 1:n_samples)
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

        m = elemof(X, interval)

        # If the just-created interval is fully covered by the other intervals
        # (at least as measured by looking at `X`), then do retry.
        #
        # Note that the predicate corresponds to `!any(m .&& .!matched)` but we
        # try not to compute stuff twice.
        matched_new = m .|| matched
        if all(matched_new .== matched)
            continue
        else
            push!(intervals, interval)
            push!(M, m)
            matched = matched_new
        end
    end

    if remove_final_fully_overlapped
        # Note that we have to `hcat` `M` because it is a vector of vectors.
        return remove_fully_overlapped(
            intervals,
            X;
            matching_matrix=hcat(M...),
        )
    else
        return intervals
    end
end

"""
Remove intervals that only match data points from `X` that the remaining
intervals match as well.
"""
function remove_fully_overlapped(
    intervals::AbstractVector{Interval},
    X::AbstractMatrix{Float64};
    matching_matrix::Union{Missing,AbstractMatrix{Bool}}=missing,
)
    if ismissing(matching_matrix)
        matching_matrix = elemof(X, intervals)
    end

    # Avoid copying the intervals multiple times by creating and manipulating a
    # view.
    view_intervals = view(intervals, :)

    k = 1
    while k < length(view_intervals)
        idxs = (1:length(view_intervals) .!= k)
        # Compute the bit vector of what the other not-yet-marked-for-removal rules
        # match.
        m_others = any(view(matching_matrix, :, idxs); dims=2)
        # Check whether the current rule matches anything that the others do not
        # match. If not, mark it for removal.
        if !any(view(matching_matrix, :, k) .&& .!m_others)
            matching_matrix = view(matching_matrix, :, idxs)
            view_intervals = view(view_intervals, idxs)
        else
            k += 1
        end
    end

    # Get the intervals pointed to by the view and return copies of them.
    return intervals[parentindices(view_intervals)...]
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
