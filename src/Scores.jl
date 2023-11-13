module Scores

using PyPlot

using ..Intervals

export mappings,
    plot_interval,
    plot_mapping,
    plot_traversal_count,
    simf_traversal_count,
    simf_traversal_count_raw,
    simf_traversal_count_root,
    similarities,
    similarities_pairwise,
    similarity,
    similarity_max,
    traversed_indices,
    traversal_count

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

Values are in [0, 1] with 1 being achieved if the intervals are the same.
"""
function subsethood_mean(interval1::Interval, interval2::Interval)
    ssh1 = subsethood(interval1, interval2)
    ssh2 = subsethood(interval2, interval1)
    return (ssh1 + ssh2) / 2.0
end

"""
The indices of data points in `X` that would be traversed by the lower and upper
bounds of one of the intervals if it were transformed into the other.

The indices are returned in two matrices, one matrix for the indices “caused” by
the lower bound and one matrix for the indices “caused” by the upper bound. Each
of these matrices has `size(X, 1)` rows (i.e. one per data point) and
`dimensions(interval1)` columns (`dimensions(interval1) ==
dimensions(interval2)` is assumed). Columns correspond to the dimensions of the
interval; traversals of a certain data point `x` can be caused by different
dimensions.
"""
function traversed_indices(
    interval1::Interval,
    interval2::Interval,
    X::AbstractMatrix{Float64},
)
    # NOTE I could speed this up by precomputing all `elemof(X, interval1)` and
    # `elemof(X, interval2)` vectors and only computing and checking the hull if
    # necessary (i.e. if its not in one of the two intervals themselves) since I
    # can then reuse the `elemof(X, interval)` matrices many times.
    inhull = elemof(X, hull(interval1, interval2))

    lbound_l = min.(interval1.lbound, interval2.lbound)
    lbound_u = max.(interval1.lbound, interval2.lbound)
    ubound_l = min.(interval1.ubound, interval2.ubound)
    ubound_u = max.(interval1.ubound, interval2.ubound)

    idx_trav_l = inhull .&& (lbound_l' .<= X .<= lbound_u')
    idx_trav_u = inhull .&& (ubound_l' .<= X .<= ubound_u')
    return idx_trav_l, idx_trav_u
end

"""
The number of data points in `X` that would be traversed by the lower and upper
bounds of one of the intervals if it were transformed into the other. Note that
some data points may be counted multiple times if they are traversed by multiple
bounds.
"""
function traversal_count(
    interval1::Interval,
    interval2::Interval,
    X::AbstractMatrix{Float64},
)
    idx_trav_l, idx_trav_u = traversed_indices(interval1, interval2, X)
    return sum(idx_trav_l) + sum(idx_trav_u)
end

"""
Given input data `X`, build a traversal count–based similarity function for
intervals.
"""
function simf_traversal_count_raw(X::AbstractMatrix{Float64})
    return (i1, i2) -> -traversal_count(i1, i2, X)
end

"""
Given input data `X`, build a traversal count–based similarity function for
intervals; mitigate punishing large intervals by taking the `DX`th root.

Larger values means “more similar”, the highest possible values is 0 (i.e. this
is a negated dissimilarity measure).
"""
function simf_traversal_count_root(X::AbstractMatrix{Float64})
    function simf(i1, i2)
        idx_trav_l, idx_trav_u = traversed_indices(i1, i2, X)
        ns_trav = [sum(idx_trav_l; dims=1) sum(idx_trav_u; dims=1)]
        return -sum(ns_trav .^ (1 / dimensions(i1)))
    end
    return simf
end

"""
For a certain similarity function, compute for two sets of intervals the
pairwise similarity values.

Returns an array where rows correspond to intervals from the first set and
columns correspond to intervals from the second set. I.e. the value at
index `2,3` corresponds `simf(intervals1[2], intervals2[3])`.
"""
function similarities_pairwise(
    intervals1::AbstractVector{Interval},
    intervals2::AbstractVector{Interval};
    simf::Function=subsethood_mean,
)
    K1 = length(intervals1)
    K2 = length(intervals2)

    similarity = fill(-1.0, (K1, K2))
    # In Julia, should try to iterate over columns first for better performance.
    for i2 in 1:K2
        for i1 in 1:K1
            similarity[i1, i2] = simf(intervals1[i1], intervals2[i2])
        end
    end
    return similarity
end

function idx_mapping(
    intervals1::AbstractVector{Interval},
    intervals2::AbstractVector{Interval};
    simf::Function=subsethood_mean,
)
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
    intervals1::AbstractVector{Interval},
    intervals2::AbstractVector{Interval};
    simf::Function=subsethood_mean,
)
    idx1to2, idx2to1 = idx_mapping(intervals1, intervals2; simf=simf)
    return mappings(idx1to2, idx2to1)
end

"""
Transforms the two cartesian indices into dictionaries.
"""
function mappings(
    idx1to2::AbstractMatrix{CartesianIndex{2}},
    idx2to1::AbstractMatrix{CartesianIndex{2}},
)
    dict1to2 = Dict(Tuple.(idx1to2))
    # We have to “transpose” the second index before creating the dictionary
    # because here, the second dimension is iterated over in `idx_mapping`.
    dict2to1 = Dict(k => v for (v, k) in Tuple.(idx2to1))
    return dict1to2, dict2to1
end

"""
Compute for the two sets of intervals the two measures (one starting from the
first set and one starting from the second set).
"""
function similarities end

function similarities(
    intervals1::AbstractVector{Interval},
    intervals2::AbstractVector{Interval};
    simf::Function=subsethood_mean,
)
    sims_pairwise = similarities_pairwise(intervals1, intervals2; simf=simf)
    return similarities(sims_pairwise)
end

function similarities(sims_pairwise)
    idx1to2, idx2to1 = idx_mapping(sims_pairwise)
    return sum(sims_pairwise[idx1to2]), sum(sims_pairwise[idx2to1])
end

"""
Compute the mean sum of minimum similarities using the given similarity measure.
I.e. take the mean of the two values returned by `similarities` (i.e. mean of
the two measures in the two directions).
"""
function similarity end

function similarity(
    intervals1::AbstractVector{Interval},
    intervals2::AbstractVector{Interval};
    simf::Function=subsethood_mean,
)
    return sum(similarities(intervals1, intervals2; simf=simf)) / 2
end

function similarity(sims_pairwise)
    return sum(similarities(sims_pairwise)) / 2
end

function similarity_max(
    intervals1::AbstractVector{Interval},
    intervals2::AbstractVector{Interval};
    simf::Function=subsethood_mean,
)
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

function plot_mapping(
    intervals1,
    intervals2;
    X=nothing,
    simf=subsethood_mean,
    ax=nothing,
)
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
    sims_pairwise = similarities_pairwise(intervals1, intervals2; simf=simf)
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
    if ax == nothing
        fig, ax = plt.subplots(2, 2; layout="constrained", figsize=(20, 10))
        fig.suptitle(
            "Mapping between intervals\n" *
            "(overall similarity score: $(round(sim, digits=2)))",
        )
    end
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

    # Plot data points if given.
    if !isnothing(X)
        for i in 1:2
            for j in 1:2
                ax[i, j].scatter(
                    view(X, :, 1),
                    view(X, :, 2);
                    c="black",
                    alpha=0.2,
                    marker="+",
                )
            end
        end
    end

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

    return fig, ax
end

function plot_traversal_count(interval1, interval2, X; ax=nothing)
    if dimensions(interval1) != 2 || dimensions(interval2) != 2
        error("only 2-dimensional intervals supported")
    end

    if ax == nothing
        fig, ax = subplots(1)
    end

    idx_trav_l, idx_trav_u = traversed_indices(interval1, interval2, X)
    hll = hull(interval1, interval2)

    ax.scatter(view(X, :, 1), view(X, :, 2); marker="+")
    style = Dict(:alpha => 0.5)
    plot_interval(ax, interval1; edgecolor="C0", facecolor="C0", style...)
    plot_interval(ax, interval2; edgecolor="C1", facecolor="C1", style...)
    plot_interval(
        ax,
        hll;
        edgecolor="gray",
        facecolor="none",
        linestyle="dashed",
    )

    function jitter()
        return randn(size(X_trav, 1)) * 0.002
    end

    X_trav = X[view(idx_trav_l, :, 1), :]
    style = Dict(:marker => "x")
    ax.scatter(
        view(X_trav, :, 1) + jitter(),
        view(X_trav, :, 2) + jitter();
        color="C2",
        style...,
    )
    X_trav = X[view(idx_trav_l, :, 2), :]
    ax.scatter(
        view(X_trav, :, 1) + jitter(),
        view(X_trav, :, 2) + jitter();
        color="C3",
        style...,
    )

    X_trav = X[view(idx_trav_u, :, 1), :]
    ax.scatter(
        view(X_trav, :, 1) + jitter(),
        view(X_trav, :, 2) + jitter();
        color="C4",
        style...,
    )
    X_trav = X[view(idx_trav_u, :, 2), :]
    return ax.scatter(
        view(X_trav, :, 1) + jitter(),
        view(X_trav, :, 2) + jitter();
        color="C5",
        style...,
    )

    if ax == nothing
        return fig, ax
    else
        return ax
    end
end

end
