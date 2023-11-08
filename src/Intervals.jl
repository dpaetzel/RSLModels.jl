module Intervals

using AutoHashEquals
using Distributions
using LinearAlgebra
using Memoize
using Mmap
using Optim
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
    params=(1.0, 1.0),
    x_min=X_MIN,
    x_max=X_MAX,
    uniform=false,
)
    return draw_spread(
        Random.default_rng(),
        dims,
        spread_min;
        spread_max=spread_max,
        params=params,
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
    # TODO Rename to params_spread_rate since unit scale
    params=(1.0, 1.0),
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

    rates_spread = rand(rng, Beta(params...), dims)

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
    params_spread::Tuple{Float64,Float64}=(1.0, 1.0),
    x_min=X_MIN,
    x_max=X_MAX,
)
    return draw_interval(
        Random.default_rng(),
        x,
        spread_min;
        spread_max=spread_max,
        params_spread=params_spread,
        x_min=x_min,
        x_max=x_max,
    )
end

function draw_interval(
    rng::AbstractRNG,
    x::AbstractVector{Float64},
    spread_min::Float64;
    spread_max=Inf,
    params_spread::Tuple{Float64,Float64}=(1.0, 1.0),
    x_min=X_MIN,
    x_max=X_MAX,
)
    dims = size(x, 1)
    spread = draw_spread(
        rng,
        dims,
        spread_min;
        spread_max=spread_max,
        params=params_spread,
        x_min=x_min,
        x_max=x_max,
    )
    center = draw_center(rng, x, spread; x_min=x_min, x_max=x_max)

    return Interval(center - spread, center + spread)
end

@memoize function my_params_spread(dims, nif)
    # TODO This is problematic: spread_ideal_cubes allows for setting
    # x_min/x_max but here we assume a unit cube
    s_ideal = Intervals.spread_ideal_cubes(dims, nif)

    function delta(fit, actual)
        return sum((fit .- actual) .^ 2)
    end

    function objective(theta, x, prob)
        fit = cdf(Beta(theta[1], theta[2]), x)
        return delta(fit, prob)
    end

    x = [s_ideal, s_ideal - 0.5 * s_ideal, s_ideal + 0.5 * s_ideal]
    cdf_x = [0.5, 0.25, 0.75]

    start = [1.0, 1.0]
    sol = optimize(
        theta -> objective(theta, x, cdf_x),
        # Lower bound for the parameters.
        [0.0, 0.0],
        # Upper bound for the parameters.
        [Inf, Inf],
        start,
    )

    a, b = Optim.minimizer(sol)

    return a, b
end

function draw_intervals(
    dims::Integer;
    nif=20,
    spread_max::Float64=Inf,
    params_spread::Tuple{Float64,Float64}=my_params_spread(dims, nif),
    rate_coverage_min::Float64=0.8,
    n_samples::Int=Parameters.n(dims),
    remove_final_fully_overlapped::Bool=true,
    x_min::Float64=Intervals.X_MIN,
    x_max::Float64=Intervals.X_MAX,
    spread_min::Float64=(x_max - x_min) / nif / 2,
    usemmap::Bool=false,
    n_intervals_max::Union{Int,Nothing}=nothing,
    return_coverage_rate::Bool=false,
    verbose::Int=0,
)
    return draw_intervals(
        Random.default_rng(),
        dims;
        nif=nif,
        spread_max=spread_max,
        params_spread=params_spread,
        rate_coverage_min=rate_coverage_min,
        n_samples=n_samples,
        remove_final_fully_overlapped=remove_final_fully_overlapped,
        x_min=x_min,
        x_max=x_max,
        spread_min=spread_min,
        usemmap=usemmap,
        n_intervals_max=n_intervals_max,
        return_coverage_rate=return_coverage_rate,
        verbose=verbose,
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
- `params_spread::Tuple{Float64, Float64}`: a and b parameters for the beta distribution.
- `rate_coverage_min::Float64`: [0, 1].
- `n_samples`:
- `remove_final_fully_overlapped`:
- `x_min::Float64`:
- `x_max::Float64`:
"""
function draw_intervals(
    rng::AbstractRNG,
    dims::Int;
    nif::Int=20,
    spread_max::Float64=Inf,
    params_spread::Tuple{Float64,Float64}=my_params_spread(dims, nif),
    rate_coverage_min::Float64=0.8,
    n_samples::Int=Parameters.n(dims),
    remove_final_fully_overlapped::Bool=true,
    x_min::Float64=Intervals.X_MIN,
    x_max::Float64=Intervals.X_MAX,
    spread_min::Float64=(x_max - x_min) / nif / 2,
    usemmap::Bool=false,
    n_intervals_max::Union{Int,Nothing}=nothing,
    return_coverage_rate::Bool=false,
    verbose::Int=0,
)
    if usemmap
        (path_X, io_X) = mktemp(tempdir())
        X = mmap(io_X, Matrix{Float64}, (n_samples, dims))
        for i in eachindex(X)
            X[i] = rand(rng) * (x_max - x_min)
        end
    else
        X = rand(rng, n_samples, dims) .* (x_max - x_min)
    end

    # TODO Use mmap for M as well? But then I have to preallocate or grow?
    M = []
    matched = fill(false, n_samples)
    intervals::AbstractVector{Interval} = []

    rate_coverage = 0.0

    while rate_coverage < rate_coverage_min &&
        (n_intervals_max == nothing || length(intervals) < n_intervals_max)
        if verbose >= 10
            println(
                "Current coverage: $(round(rate_coverage; digits=2)) " *
                "of required $rate_coverage_min",
            )
        end

        idx = rand(rng, 1:n_samples)
        # Note that this may use the same `x` for multiple intervals which is
        # probably not wanted. However, filtering for unmatched `x` is probably
        # not that much cheaper than just trying any `x` and possibly discarding
        # the interval.
        x = X[idx, :]

        interval = draw_interval(
            rng,
            x,
            Intervals.spread_ideal_cubes(dims, nif; factor=0.1);
            spread_max=spread_max,
            params_spread=params_spread,
            x_min=x_min,
            x_max=x_max,
        )

        m = elemof(X, interval; usemmap=usemmap)

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
            rate_coverage = count(matched) / n_samples
        end
    end

    intervals_final = if remove_final_fully_overlapped
        if verbose >= 10
            println("Removing fully overlapped intervals …")
        end
        # Note that we have to `hcat` `M` because it is a vector of vectors.
        remove_fully_overlapped(intervals, X; matching_matrix=reduce(hcat, M))
    else
        intervals
    end

    if return_coverage_rate
        return rate_coverage, intervals
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

# TODO Rename this to …unit and replace x_min/x_max with 0/1
"""
Given a number of intervals and an input space dimensions, compute the spread
that a cube would have with a volume of `1/n_intervals` of input space volume.

`factor` is a factor of the *volume* of what we call an ideal cube. I.e. we
compute the spread that a cube would have with a volume of `1/n_intervals *
factor` of input space volume.
"""
function spread_ideal_cubes(
    dims::Integer,
    n_intervals;
    factor=1.0,
    x_min=X_MIN,
    x_max=X_MAX,
)
    volume_avg = factor * (x_max - x_min)^dims / n_intervals
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
