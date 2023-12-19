"""
This is the mutation operator used by (Ryerkerk et al., 2020).

It mutates, on average, one of the metavariables (i.e. conditions) fully but
allows those mutations to be spread over several metavariables.
"""
function mutate end

function mutate(rng, g::EvaluatedGenotype, X::XType, config::GARegressor)
    return mutate(rng, g.genotype, X, config)
end

function mutate(rng, g::Genotype, X::XType, config::GARegressor)
    # TODO Consider whether to add/rm first (so we don't waste bound mutation on
    # conditions that are removed a few lines later anyway). Note, however, that
    # this might result in an additional copy operation.
    g_ = mutate_bounds(rng, g, config)

    # TODO Expose p_mu_add
    # Add a metavariable.
    if rand(rng) < 0.05
        N, DX = size(X)
        # TODO Consider to enforce matching a configurable number of data points
        # Draw a random data point.
        idx = rand(rng, 1:N)
        x = X[idx, :]

        condition = draw_interval(
            rng,
            x;
            spread_min=config.spread_min,
            spread_max=config.spread_max,
            params_spread=(a=config.params_spread_a, b=config.params_spread_b),
            x_min=config.x_min,
            x_max=config.x_max,
        )

        push!(g_, condition)
    end

    # TODO Expose p_mu_rm
    # Remove a metavariable.
    if rand(rng) < 0.05
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
    config::GARegressor,
)
    return mutate_bounds(rng, g.genotype, config)
end

function mutate_bounds(rng::AbstractRNG, g::Genotype, config::GARegressor)
    DX = dimensions(g[1])

    # TODO Consider to make intervals mutable for performance
    g_ = deepcopy(g)

    # Number of metavariables.
    l = length(g_)

    # We have `2 * DX` design variables per metavariable (each condition is one
    # metavariable).
    n_dvars = 2 * DX

    # TODO Expose a factor for this in config
    p = 1.0 / (n_dvars * l)

    # TODO Look at efficiency of this combination of loops and rand

    # Go over all metavariables (i.e. all conditions).
    for idx in eachindex(g_)
        # TODO Expose std_mut parameter
        # TODO Check whether used std is sensible here
        # Declaratively apply the mutation rate (mask is 1 if the design
        # variable should be mutated and 0 otherwise). Note that we draw too
        # many values from the normal since we only use the ones where `mask !=
        # 0`; we should look at whether this can be sped up by drawing the
        # number of mutations from a binomial here and then only drawing the
        # necessary alterations (but I suspect that this is slower—it was in
        # Python, anyway).
        mask = rand(rng, Bernoulli(p), DX)
        std_mut = 0.05
        x = rand(rng, Normal(0, std_mut .* (config.x_max .- config.x_min)), DX)
        g_[idx].lbound[:] = g_[idx].lbound .+ mask .* x
        g_[idx].lbound[:] = mirror.(g_[idx].lbound, config.x_min, config.x_max)

        mask = rand(rng, Bernoulli(p), DX)
        x = rand(rng, Normal(0, std_mut .* (config.x_max .- config.x_min)), DX)
        g_[idx].ubound[:] = g_[idx].ubound .+ mask .* x
        g_[idx].ubound[:] = mirror.(g_[idx].ubound, config.x_min, config.x_max)
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
        return _mirror(x_, x_min, x_max)
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
