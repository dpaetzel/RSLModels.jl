"""
    mutate(rng)

This is the mutation operator used by (Ryerkerk et al., 2020).

It mutates, on average, one of the metavariables (i.e. conditions) fully but
allows those mutations to be spread over several metavariables and, with some
probabilities, also adds or removes a single metavariable.
"""
function mutate end

function mutate(
    rng::AbstractRNG,
    g::EvaluatedGenotype,
    X::XType,
    rate_mut::Float64,
    rate_std::Float64,
    p_add::Float64,
    p_rm::Float64,
    x_min::Float64,
    x_max::Float64,
    nmatch_min::Int,
)
    return mutate(
        Random.default_rng(),
        g.genotype,
        X,
        rate_mut,
        rate_std,
        p_add,
        p_rm,
        x_min,
        x_max,
        nmatch_min,
    )
end

function mutate(
    rng::AbstractRNG,
    g::Genotype,
    X::XType,
    rate_mut::Float64,
    rate_std::Float64,
    p_add::Float64,
    p_rm::Float64,
    x_min::Float64,
    x_max::Float64,
    nmatch_min::Int,
)
    # TODO Consider whether to add/rm first (so we don't waste bound mutation on
    # conditions that are removed a few lines later anyway). Note, however, that
    # this might result in an additional copy operation.
    g_ = mutate_bounds(rng, g, rate_mut, rate_std, x_min, x_max)

    # Add a metavariable.
    if rand(rng) < p_add
        N, DX = size(X)

        # TODO Performance: Do not recompute this here but cache earlier
        matching_matrix = elemof(X, g)
        inv_weights = vec(float(sum(matching_matrix; dims=2)))
        # If some data points are not covered by any of the conditions, we add
        # the new condition such that it covers at least one of them.
        weights = if any(inv_weights .== 0.0)
            float(inv_weights .== 0.0)
            # Otherwise we add the new condition such that overlap is not increased
            # where there is a lot of overlap already.
        else
            1.0 ./ inv_weights
        end
        # We need to normalize the weights next.
        weights /= sum(weights)

        # We want the new condition to match `nmatch_min + 2` many data points.
        # The `+ 2` is a design decision: We don't want the repair operator to
        # immediately remove the rule again and therefore want to match two data
        # points more than strictly necessary.
        #
        # Draw a single point.
        idx = sample(rng, 1:N, ProbabilityWeights(weights))
        x = X[idx, :]
        distances = norm.(eachrow(x' .- X))
        # `PartialQuickSort` receives an index (or a range of indices) for which
        # it then computes the values lying there in a hypothetical sorted
        # array. I.e. performs just enough of quick sort to be sure what would
        # be at these indexes in the sorted array.
        idx =
            sortperm(distances; alg=PartialQuickSort(1:(nmatch_min + 2)))[1:(nmatch_min + 2)]
        x = X[idx, :]

        # Compute the minimum interval containing the points.
        condition = Interval(
            vec(minimum(x; dims=1)),
            vec(maximum(x; dims=1));
            lopen=falses(DX),
            uopen=falses(DX),
        )

        # TODO Make this a test
        @assert all(elemof(x, condition))

        push!(g_, condition)
    end

    # Remove a metavariable if it leaves at least `nmatch_min`.
    if length(g_) > nmatch_min && rand(rng) < p_rm
        idx = rand(rng, eachindex(g_))
        deleteat!(g_, idx)
    end

    report = (;)
    return g_, report
end

"""
Mutates the bounds of the conditions of the given genotype.
"""
function mutate_bounds end

function mutate_bounds(
    rng::AbstractRNG,
    g::EvaluatedGenotype,
    rate_mut::Float64,
    rate_std::Float64,
    x_min::Float64,
    x_max::Float64,
)
    return mutate_bounds(rng, g.genotype, rate_mut, rate_std, x_min, x_max)
end

function mutate_bounds(
    rng::AbstractRNG,
    g::Genotype,
    rate_mut::Float64,
    rate_std::Float64,
    x_min::Float64,
    x_max::Float64,
)
    DX = dimensions(g[1])

    # TODO Consider to make intervals mutable for performance
    g_ = deepcopy(g)

    # Number of metavariables.
    l = length(g_)

    # We have `2 * DX` design variables per metavariable (each condition is one
    # metavariable).
    n_dvars = 2 * DX
    p = rate_mut * 1.0 / (n_dvars * l)
    std_mut = rate_std * (x_max - x_min)

    dist_mask = Bernoulli(p)
    dist_mut = Normal(0, std_mut .* (x_max .- x_min))

    # Do this in-place so we do not have to malloc repeatedly. However, this
    # seems to bring something like 100μs for `l == 1000` and `DX == 10`.
    mask = Vector{Bool}(undef, DX)
    mutation = Vector{Float64}(undef, DX)

    # TODO Look at efficiency of this combination of loops and rand
    #
    # Go over all metavariables (i.e. all conditions).
    for idx in eachindex(g_)
        # Declaratively apply the mutation rate (mask is 1 if the design
        # variable should be mutated and 0 otherwise). Note that we draw too
        # many values from the normal since we only use the ones where `mask !=
        # 0`; we should look at whether this can be sped up by drawing the
        # number of mutations from a binomial here and then only drawing the
        # necessary alterations (but I suspect that this is slower—it was in
        # Python, anyway).
        rand!(rng, dist_mask, mask)
        rand!(rng, dist_mut, mutation)
        g_[idx].lbound[:] = g_[idx].lbound .+ mask .* mutation
        g_[idx].lbound[:] = mirror.(g_[idx].lbound, x_min, x_max)

        rand!(rng, dist_mask, mask)
        rand!(rng, dist_mut, mutation)
        g_[idx].ubound[:] = g_[idx].ubound .+ mask .* mutation
        g_[idx].ubound[:] = mirror.(g_[idx].ubound, x_min, x_max)
    end

    return g_
end

"""
Restrict the given value to the given bounds using the mirror correction
strategy (see, e.g., Kononova et al.'s 2022 article *Differential Evolution
Outside the Box*).

Note that the minimum and maximum values given by `a_min` and `a_max` are
themselves seen as being included in the allowed range.
"""
function mirror(x::Float64, x_min::Float64, x_max::Float64)
    x_ = x - (x > x_max) * 2 * (x - x_max)
    x_ = x_ + (x_ < x_min) * 2 * (x_min - x_)

    if x_ > x_max || x_ < x_min
        return mirror(x_, x_min, x_max)
    else
        return x_
    end
end

# function mutate_bounds_center_spread()
#     centers = (g_[idx].lbound .+ g_[idx].ubound) ./ 2
#     spreads = (g_[idx].ubound .- g_[idx].lbound) ./ 2

#     # TODO Expose a factor for this in config
#     p = 1.0 / (n_dvars * l)

#     # TODO Consider to ensure that mutation actually changes match vector

#     # I think it makes sense to first mutate the center and afterwards the
#     # spread.
#     # TODO Expose std_mut parameter
#     # TODO Check whether used std is sensible here
#     mask = rand(rng, Bernoulli(p), DX)
#     std_mut = 0.03
#     x = rand(rng, Normal(0, std_mut .* (config.x_max .- config.x_min)), DX)
#     centers[:] = centers .+ mask .* x
#     centers[:] = mirror.(centers, config.x_min, config.x_max)

#     spread_max = (config.x_max - config.x_min) / 2
#     mask = rand(rng, Bernoulli(p), DX)
#     x = rand(rng, Normal(0, std_mut .* spread_max), DX)
#     spreads[:] = spreads .+ mask .* x
#     spreads[:] = mirror.(spreads, 0.0, spread_max)

#     return g_[idx] = Intervals.Interval(
#         centers - spreads,
#         centers + spreads;
#         lopen=g_[idx].lopen,
#         uopen=g_[idx].uopen,
#     )
# end
