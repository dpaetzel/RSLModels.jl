using Distributions
using LinearAlgebra
using PyPlot
using Random
using StatsBase


# We want to track global assignments as well when doing
# `includet(thisfile.jl)`.
__revise_mode__ = :evalassign


X_MIN = 0.0
X_MAX = 1.0


dist_spread = Beta(1.55, 2.74)


# TODO Consider to represent empty arrays properly instead of `nothing`
struct Interval
    lbound::Vector{Float64}
    ubound::Vector{Float64}
    Interval(lbound::Vector{Float64}, ubound::Vector{Float64}) =
        any(lbound .> ubound) ? error("out of order") : new(lbound, ubound)
end


function match(interval, x)
    interval.lbound <= x <= interval.ubound
end


function volume(interval::Interval)
    prod(interval.ubound .- interval.lbound)
end


function dimensions(interval::Interval)
    length(interval.lbound)
end


"""
Given centers and spreads, compute the volume of the corresponding interval.
"""
function volume(;centers, spreads)
    # NOTE We enforce keyword arguments here so that users do not confuse
    # arguments of this with the in some cases more natural `lbound`/`ubound`
    # representation.
    interval = Interval(centers - spreads, centers + spreads)
    volume(interval)
end


# NOTE Dimension should actually be unsigned but I don't see how to achieve that
# cleanly right now (i.e. without having to do `unsigned(10)` or `0xa` or
# similar).
function draw_spreads(dimensions::Integer,
                      spread_min;
                      spread_max=Inf,
                      x_min=X_MIN,
                      x_max=X_MAX)
    draw_spreads(Random.default_rng(),
                 dimensions,
                 spread_min;
                 spread_max=spread_max,
                 x_min=x_min,
                 x_max=x_max)
end


function draw_spreads(rng::AbstractRNG,
                      dimensions::Integer,
                      spread_min;
                      spread_max=Inf,
                      x_min=X_MIN,
                      x_max=X_MAX)
    if spread_min > spread_max
        throw(ArgumentError(
            "the condition `spread_min <= spread_max` is not satisfied"))
    end
    rates_spread = rand(rng, dist_spread, dimensions)
    spread_max = min((x_max - x_min) / 2, spread_max)
    spread_min .+ rates_spread .* (spread_max - spread_min)
end


function draw_centers(spread; x_min=X_MIN, x_max=X_MAX)
    draw_centers(Random.default_rng(), spread; x_min=x_min, x_max=x_max)
end


function draw_centers(rng::AbstractRNG, spread::Real; x_min=X_MIN, x_max=X_MAX)
    lbound = x_min + spread
    ubound = x_max - spread
    if lbound == ubound
        lbound
    elseif lbound >= ubound
        throw(ArgumentError(
            "the condition `spread <= (x_min - x_max) / 2` is not satisfied"))
    else
        rand(rng, Uniform(x_min + spread, x_max - spread))
    end
end


# NOTE We cannot write `Vector{Real}` here because in Julia it does NOT follow
# from `T1 <: T2` that `T{T1} <: T{T2}`. Instead, we write `Vector` for which we
# have `Vector{T} <: Vector` for all `T`.
function draw_centers(rng::AbstractRNG,
                      spread::Vector;
                      x_min=X_MIN,
                      x_max=X_MAX)
    # Simply draw an independent spread for each dimension.
    [draw_centers(rng, s; x_min=x_min, x_max=x_max) for s in spread]
end


function draw_interval(dimensions::Integer,
                       spread_min,
                       volume_min;
                       spread_max=Inf,
                       x_min=X_MIN,
                       x_max=X_MAX)
    draw_interval(Random.default_rng(),
                  dimensions,
                  spread_min,
                  volume_min;
                  spread_max=spread_max,
                  x_min=x_min,
                  x_max=x_max)
end


function draw_interval(rng::AbstractRNG,
                       dimensions::Integer,
                       spread_min,
                       volume_min;
                       spread_max=Inf,
                       x_min=X_MIN,
                       x_max=X_MAX)
    spreads = draw_spreads(rng,
                           dimensions - 1,
                           spread_min;
                           spread_max=spread_max,
                           x_min=x_min,
                           x_max=x_max)
    centers = draw_centers(rng, spreads)

    width_min = max(spread_min * 2,
                    volume_min / volume(centers=centers, spreads=spreads))

    # TODO Put a check here for spread_max not conflicting with our “fix the last dimension thing”

    width_max = x_max - x_min

    iter_max = 20
    iter = 0
    # For high dimensions, we often have `width_min > width_max`. We re-draw the
    # smallest spread until `width_min` is small enough.
    while width_min > width_max && iter < iter_max
        iter += 1
        println("Rejecting due to min width greater max width " *
            "($width_min > $width_max).")
        # Start by getting the smallest spread's index.
        i = argmin(spreads)
        # Draw a new spread (and based on that, a center) and replace the
        # smallest spread (and the corresponding center).
        # TODO Consider to overload draw_spread so we don't need to extract here
        spreads[i] = draw_spreads(rng,
                                  1,
                                  spread_min;
                                  spread_max=spread_max,
                                  x_min=x_min,
                                  x_max=x_max)[1]
        l = x_min + spreads[i]
        u = x_max - spreads[i]
        centers[i] = draw_centers(rng, spreads[i])[1]

        width_min = volume_min / volume(centers=centers, spreads=spreads)
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
    i_rand = rand(rng, 1:dimensions)
    insert!(spreads, i_rand, spread_last)
    insert!(centers, i_rand, center_last)

    Interval(centers - spreads, centers + spreads)
end


function draw_intervals(dimensions::Integer,
                        n_intervals,
                        spread_min=spread_ideal_cubes(dimensions, n_intervals),
                        volume_min=volume_min_factor(dimensions, n_intervals);
                        spread_max=Inf,
                        x_min=X_MIN,
                        x_max=X_MAX)
    draw_intervals(
        Random.default_rng(),
        dimensions,
        n_intervals,
        spread_min,
        volume_min;
        spread_max=spread_max,
        x_min=x_min,
        x_max=x_max)
end


"""
Parameters
----------
dimensions : int > 0
n_intervals : int > 0
volume_min : float > 0
random_state : np.random.RandomState

Returns
-------
array, list, list
    The intervals as an array of shape `(n_intervals, 2, dimensions)`, the
    set of pair-wise intersections between the intervals, the set of volumes
    of the non-empty ones of these pair-wise intersections.
"""
function draw_intervals(rng::AbstractRNG,
                        dimensions::Integer,
                        n_intervals,
                        spread_min=spread_ideal_cubes(dimensions, n_intervals),
                        volume_min=volume_min_factor(dimensions, n_intervals);
                        spread_max=Inf,
                        x_min=X_MIN,
                        x_max=X_MAX)

    [ draw_interval(rng,
                    dimensions,
                    spread_min,
                    volume_min;
                    spread_max=spread_max,
                    x_min=x_min,
                    x_max=x_max)
      for _ in 1:n_intervals
    ]
end


"""
Given a number of intervals and an input space dimensions, compute the spread
that a cube would have with a volume of `1/n_intervals` of input space volume.
"""
function spread_ideal_cubes(dimensions::Integer, n_intervals; x_min=X_MIN, x_max=X_MAX)
    volume_avg = (x_max - x_min)^dimensions / n_intervals
    volume_avg^(1.0 / dimensions) / 2.0
end


# TODO Consider to factor in the number of training data points here
"""
Given a number of intervals and an input space dimensions, compute the minimum
volume as `factor` of `1/n_intervals` of input space volume.
"""
function volume_min_factor(dimensions::Integer,
                           n_intervals,
                           factor=0.1;
                           x_min=X_MIN,
                           x_max=X_MAX)
    volume_input_space = (x_max - x_min)^dimensions
    factor * volume_input_space / n_intervals
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
Computes the subsethood degree of the first interval with respect to the second
interval, i.e. the rate of the first interval's volume that is contained in the
second interval.

See huidobro2022.
"""
function subsethood(interval1::Interval, interval2::Interval)
    intersect = intersection(interval1, interval2)

    if intersect === nothing
        0.0
    else
        v = volume(interval1)
        if v == 0.0
            1.0
        else
            volume(intersect) / v
        end
    end
end


"""
Computes the mean of the subsethood degrees of the two intervals with respect to
each other.
"""
function subsethood_mean(interval1, interval2)
    ssh1 = subsethood(interval1, interval2)
    ssh2 = subsethood(interval2, interval1)
    return (ssh1 + ssh2) / 2.0
end


"""
For a certain similarity function, compute for two sets of intervals the
pairwise similarity values.

Returns an array where rows correspond to intervals from the first set and
columns correspond to intervals from the second set. I.e. the value at
index `2,3` corresponds `simf(intervals1[2], intervals2[3])`.
"""
function similarities_pairwise(intervals1, intervals2; simf=subsethood_mean)
    K1 = length(intervals1)
    K2 = length(intervals2)

    similarity = fill(-1.0, (K1, K2))
    for i1 in 1:K1
        for i2 in 1:K2
            similarity[i1, i2] = simf(intervals1[i1], intervals2[i2])
        end
    end
    similarity
end


# function plot_interval(ax, interval; x_min=X_MIN, x_max=X_MAX, edge_color="black", fill_color="none")
function plot_interval(ax, interval; x_min=X_MIN, x_max=X_MAX, score=nothing, kwargs...)
    if dimensions(interval) != 2
        error("Plotting only supported for 2D intervals")
    end
    x1_start, x2_start = interval.lbound
    x1_end, x2_end = interval.ubound
    rectangle = plt.Rectangle((x1_start, x2_start),
                              x1_end - x1_start,
                              x2_end - x2_start;
                              kwargs...)
    ax.add_patch(rectangle)
    # Add the score (if given) to the top right corner.
    if score != nothing
        padding = (x_max - x_min) / 100.0
        textcolor = get(kwargs, :edgecolor, "black")
        ax.text(x1_end,
                x2_end - padding,
                string(round(score,
                digits=2)),
                verticalalignment="top",
                horizontalalignment="right",
                color=textcolor)
    end
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax
end


function plot_mapping(intervals1, intervals2)
    # Make It So™ that intervals1 is the smaller set of intervals.
    if length(intervals1) > length(intervals2)
        tmp = intervals1
        intervals1 = intervals2
        intervals2 = tmp
    end


    # Maximize over the second dimension of the pairwise similarities to get for
    # each interval in `intervals1` the closest interval in `intervals2`. Do the
    # same for the transposed array to get for each interval in `intervals2` the
    # closest interval in `intervals1`.
    sims_pairwise = similarities_pairwise(intervals1, intervals2)
    idx1to2 = argmax(sims_pairwise, dims=2)
    idx2to1 = argmax(sims_pairwise', dims=2)


    # Transform to dictionaries.
    idx1to2 = Dict(Tuple.(idx1to2))
    idx2to1 = Dict(Tuple.(idx2to1))


    # The number of colors used corresponds to the larger number of intervals.
    n_colours = maximum(length.([intervals1, intervals2]))
    cmap = get_cmap("rainbow")
    colnorm = plt.Normalize(1, n_colours)
    rule_colour = idx -> cmap(colnorm(idx))


    # Initialize the figure.
    fig, ax = plt.subplots(2, 2, layout="constrained", figsize=(10,5))
    ax[1,1].set_title("Intervals in first set")
    ax[1,2].set_title("Most similar intervals in second set")
    ax[2,1].set_title("Intervals in second set")
    ax[2,2].set_title("Most similar intervals in first set")


    # Compute which interval (indices) aren't mapped to.
    idx_not_mapped_to_from_1_to_2 = setdiff(Set(1:length(intervals2)), Set(values(idx1to2)))
    idx_not_mapped_to_from_2_to_1 = setdiff(Set(1:length(intervals1)), Set(values(idx2to1)))

    # Plot intervals that are not mapped to.
    for idx in idx_not_mapped_to_from_1_to_2
        plot_interval(ax[1,2], intervals2[idx]; edgecolor="black", linestyle="dashed", facecolor="none")
    end

    for idx in idx_not_mapped_to_from_2_to_1
        plot_interval(ax[2,2], intervals1[idx]; edgecolor="black", linestyle="dashed", facecolor="none")
    end

    # Always loop over the smaller set (otherwise we will definitely paint over
    # already coloured stuff).
    for idx1 in 1:length(intervals1)
        # Start with the `for each in intervals1 find and mark the closest in
        # intervals2` direction.
        #
        # Plot the interval we're currently looking at.
        plot_interval(ax[1,1], intervals1[idx1]; score=sims_pairwise[idx1,idx1to2[idx1]], edgecolor=rule_colour(idx1to2[idx1]), facecolor="none")

        # Plot the closest interval to the currently-looked-at interval.
        plot_interval(ax[1,2], intervals2[idx1to2[idx1]]; edgecolor=rule_colour(idx1to2[idx1]), facecolor="none")
    end

    for idx2 in 1:length(intervals2)
        # Start with the `for each in intervals1 find and mark the closest in
        # intervals2` direction.
        #
        # Plot the interval we're currently looking at.
        plot_interval(ax[2,1], intervals2[idx2]; score=sims_pairwise'[idx2,idx2to1[idx2]], edgecolor=rule_colour(idx2to1[idx2]), facecolor="none")

        # Plot the closest interval to the currently-looked-at interval.
        plot_interval(ax[2,2], intervals1[idx2to1[idx2]]; edgecolor=rule_colour(idx2to1[idx2]), facecolor="none")
    end
end


intervals1 = draw_intervals(2, 5, 0.05; spread_max=0.2)
intervals2 = draw_intervals(2, 4, 0.05; spread_max=0.2)


plot_mapping(intervals1, intervals2)
