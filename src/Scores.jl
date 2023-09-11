module Scores

using PyPlot

using ..Intervals

export similarities_pairwise,
    mappings,
    similarities,
    similarity,
    similarity_max,
    plot_interval,
    plot_mapping

const X_MIN::Float64 = Intervals.X_MIN
const X_MAX::Float64 = Intervals.X_MAX

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
function subsethood_mean(interval1::Interval, interval2::Interval)
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
    return similarity
end

function idx_mapping(intervals1, intervals2; simf=subsethood_mean)
    sims_pairwise = similarities_pairwise(intervals1, intervals2; simf=simf)
    return idx_mapping(sims_pairwise)
end

function idx_mapping(sims_pairwise)
    return argmax(sims_pairwise; dims=2), argmax(sims_pairwise; dims=1)
end

"""
The two unidirectional mappings (with respect to the similarity metric given as
`simf`) between the two sets of intervals.
"""
function mappings(
    intervals1::Vector{Interval},
    intervals2::Vector{Interval};
    simf=subsethood_mean,
)
    idx1to2, idx2to1 = idx_mapping(intervals1, intervals2; simf=simf)
    return mappings(idx1to2, idx2to1)
end

"""
Transforms the two cartesian indices into dictionaries.
"""
function mappings(
    idx1to2::Matrix{CartesianIndex{2}},
    idx2to1::Matrix{CartesianIndex{2}},
)
    dict1to2 = Dict(Tuple.(idx1to2))
    # We have to “transpose” the second index before creating the dictionary
    # because here, the second dimension is iterated over in `idx_mapping`.
    dict2to1 = Dict(k => v for (v, k) in Tuple.(idx2to1))
    return dict1to2, dict2to1
end

function similarities(intervals1, intervals2; simf=subsethood_mean)
    sims_pairwise = similarities_pairwise(intervals1, intervals2; simf=simf)
    return similarities(sims_pairwise)
end

function similarities(sims_pairwise)
    idx1to2, idx2to1 = idx_mapping(sims_pairwise)
    return sum(sims_pairwise[idx1to2]), sum(sims_pairwise[idx2to1])
end

function similarity(intervals1, intervals2; simf=subsethood_mean)
    return sum(similarities(intervals1, intervals2; simf=simf))
end

function similarity(sims_pairwise)
    return sum(similarities(sims_pairwise))
end

function similarity_max(intervals1, intervals2; simf=subsethood_mean)
    if simf == subsethood_mean
        # The maximum similarity between two intervals is 1.0. Since we compare
        # all pairwise (in both directions), we simply have to sum the sizes of
        # the sets of intervals.
        length(intervals1) + length(intervals2)
    else
        throw(
            ArgumentError(
                "similarity_max unsupported for the similarity function provided",
            ),
        )
    end
end

function plot_interval(
    ax,
    interval;
    x_min=X_MIN,
    x_max=X_MAX,
    score=nothing,
    kwargs...,
)
    if dimensions(interval) != 2
        error("Plotting only supported for 2D intervals")
    end
    x1_start, x2_start = interval.lbound
    x1_end, x2_end = interval.ubound
    rectangle = plt.Rectangle(
        (x1_start, x2_start),
        x1_end - x1_start,
        x2_end - x2_start;
        kwargs...,
    )
    ax.add_patch(rectangle)
    # Add the score (if given) to the top right corner.
    if score != nothing
        padding = (x_max - x_min) / 100.0
        textcolor = get(kwargs, :edgecolor, "black")
        ax.text(
            x1_end,
            x2_end - padding,
            string(round(score; digits=2));
            verticalalignment="top",
            horizontalalignment="right",
            color=textcolor,
        )
    end
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    return ax
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
    idx1to2, idx2to1 = idx_mapping(sims_pairwise)
    idx1to2, idx2to1 = mappings(idx1to2, idx2to1)

    sims = similarities(sims_pairwise)
    sim = similarity(sims_pairwise)

    # The number of colors used corresponds to the larger number of intervals.
    n_colours = maximum(length.([intervals1, intervals2]))
    cmap = get_cmap("rainbow")
    colnorm = plt.Normalize(1, n_colours)
    rule_colour = idx -> cmap(colnorm(idx))

    # Initialize the figure.
    fig, ax = plt.subplots(2, 2; layout="constrained", figsize=(10, 5))
    fig.suptitle(
        "Mapping between intervals\n" *
        "(overall similarity score: $(round(sim, digits=2)); " *
        "theoretical maximum in this case: " *
        "$(similarity_max(intervals1, intervals2)))",
    )
    ax[1, 1].set_title("Intervals in first set ($(length(intervals1)))")
    ax[1, 2].set_title(
        "Most similar intervals in second set\n" *
        "(overall score in this direction: $(round(sims[1], digits=2)))",
    )
    ax[2, 1].set_title("Intervals in second set ($(length(intervals2)))")
    ax[2, 2].set_title(
        "Most similar intervals in first set\n" *
        "(overall score in this direction: $(round(sims[2], digits=2)))",
    )

    # Compute which interval (indices) aren't mapped to.
    idx_not_mapped_to_from_1_to_2 =
        setdiff(Set(1:length(intervals2)), Set(values(idx1to2)))
    idx_not_mapped_to_from_2_to_1 =
        setdiff(Set(1:length(intervals1)), Set(values(idx2to1)))

    # Plot intervals that are not mapped to.
    for idx in idx_not_mapped_to_from_1_to_2
        plot_interval(
            ax[1, 2],
            intervals2[idx];
            edgecolor="black",
            linestyle="dashed",
            facecolor="none",
        )
    end

    for idx in idx_not_mapped_to_from_2_to_1
        plot_interval(
            ax[2, 2],
            intervals1[idx];
            edgecolor="black",
            linestyle="dashed",
            facecolor="none",
        )
    end

    # Always loop over the smaller set (otherwise we will definitely paint over
    # already coloured stuff).
    for idx1 in 1:length(intervals1)
        # Start with the `for each in intervals1 find and mark the closest in
        # intervals2` direction.
        #
        # Plot the interval we're currently looking at.
        plot_interval(
            ax[1, 1],
            intervals1[idx1];
            score=sims_pairwise[idx1, idx1to2[idx1]],
            edgecolor=rule_colour(idx1to2[idx1]),
            facecolor="none",
        )

        # Plot the closest interval to the currently-looked-at interval.
        plot_interval(
            ax[1, 2],
            intervals2[idx1to2[idx1]];
            edgecolor=rule_colour(idx1to2[idx1]),
            facecolor="none",
        )
    end

    for idx2 in 1:length(intervals2)
        # Start with the `for each in intervals1 find and mark the closest in
        # intervals2` direction.
        #
        # Plot the interval we're currently looking at.
        plot_interval(
            ax[2, 1],
            intervals2[idx2];
            score=sims_pairwise'[idx2, idx2to1[idx2]],
            edgecolor=rule_colour(idx2to1[idx2]),
            facecolor="none",
        )

        # Plot the closest interval to the currently-looked-at interval.
        plot_interval(
            ax[2, 2],
            intervals1[idx2to1[idx2]];
            edgecolor=rule_colour(idx2to1[idx2]),
            facecolor="none",
        )
    end
end

end
