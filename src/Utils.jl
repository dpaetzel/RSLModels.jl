module Utils

using DataFrames
using Serialization

export data_coverage,
    data_overlaps,
    data_overlap_pairs,
    data_overlap_mean_norm,
    data_overlap_pairs_mean,
    data_overlap_pairs_mean_per_rule,
    data_overlap_pairs_mean_per_ruleset,
    readstats

"""
Assumes that the last column of the matching matrix is all-ones due to being a
default rule's column.
"""
function data_overlaps(matching_matrix::AbstractMatrix{Bool})
    # Do not consider the default rule.
    m = matching_matrix[:, 1:(end - 1)]

    # Compute the number of overlapping rules per training data point.
    return sum(m; dims=2)
end

"""
Assumes that the last column of the matching matrix is all-ones due to being a
default rule's column.
"""
function data_overlap_pairs(matching_matrix::AbstractMatrix{Bool})
    count_overlap = data_overlaps(matching_matrix)

    # Compute the number of pairwise overlaps induced by these rules.
    return binomial.(count_overlap, 2)
end

"""
Mean rate (of all data points) of pairwise overlapping data points in the rule
set.
"""
function data_overlap_pairs_mean_per_ruleset(
    matching_matrix::AbstractMatrix{Bool},
)
    overlap_pairs = data_overlap_pairs(matching_matrix)

    return sum(overlap_pairs) / size(matching_matrix, 1)
end

"""
Mean rate (of all data points) of pairwise overlapping data points per rule.
"""
function data_overlap_pairs_mean_per_rule(
    matching_matrix::AbstractMatrix{Bool},
)
    # Have to remove the default rule in the denominator hence the -1.
    return data_overlap_pairs_mean_per_ruleset(matching_matrix) /
           (size(matching_matrix, 2) - 1)
end

function data_overlap_mean_norm(matching_matrix::AbstractMatrix{Bool})
    return data_overlap_pairs_mean(matching_matrix) /
           binomial(size(matching_matrix, 2), 2)
end

"""
Assumes that the last column of the matching matrix is all-ones due to being a
default rule's column.
"""
function data_coverage(matching_matrix::AbstractMatrix{Bool})
    # Do not consider the default rule.
    m = matching_matrix[:, 1:(end - 1)]

    return sum(any(m; dims=2)) / size(matching_matrix, 1)
end

function readstats(;
    prefix_fname::String="remote-genstats-genall",
    suffix=".stats.jls",
)
    fnames = map(
        fname -> prefix_fname * "/" * fname,
        filter(fname -> endswith(fname, suffix), readdir(prefix_fname)),
    )

    data = deserialize.(fnames)
    df = DataFrame(
        map(
            dict -> merge(
                Dict("params." * string(k) => v for (k, v) in dict[:params]),
                Dict("stats." * string(k) => v for (k, v) in dict[:stats]),
            ),
            data,
        ),
    )

    df[!, "fname"] .= fnames

    df[!, "params.d"] .= Int.(df[:, "params.d"])
    df[!, "params.N"] .= Int.(df[:, "params.N"])
    df[!, "params.nif"] .= Int.(df[:, "params.nif"])
    df[!, "params.seed"] .= Int.(df[:, "params.seed"])
    return df
end

end
