const Genotype = AbstractVector{Intervals.Interval}

const XType::DataType = AbstractMatrix{Float64}
const YType::DataType = AbstractVector{Float64}

const draw_genotype = draw_intervals

unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

"""
Wrapper to store solutions together with the models they induce as well as their
fitness score.
"""
@auto_hash_equals struct EvaluatedGenotype
    genotype::Genotype
    phenotype::Models.Model
    fitness::Float64
end

function Base.length(g::EvaluatedGenotype)
    return Base.length(g.genotype)
end

"""
Express the given genotype, yielding a phenotype.

This is achieved by fitting a rule set model for based on the set of conditions
specified by the genotype.
"""
function express(
    genotype::Genotype,
    X::XType,
    y::YType,
    x_min,
    x_max,
    nmatch_min::Int,
)
    # TODO Consider to cache matching
    N, DX = size(X)

    conditions = deepcopy(genotype)

    dists_out = []
    idx_rm = []
    for idx in eachindex(conditions)
        y_matched = view(y, elemof(X, conditions[idx]))
        # TODO Expose this parameter as match_min or something like that
        # TODO Disable this test if `repair` is used
        # If too few training data points are matched, ignore the condition.
        if length(y_matched) < nmatch_min
            push!(idx_rm, idx)
        else
            dist_out = fit_mle(Normal, y_matched)
            push!(dists_out, dist_out)
        end
    end
    # Delete conditions that do not match enough training data points.
    deleteat!(conditions, idx_rm)

    # Inverse variance–based mixing (see drugowitsch2008b).
    vars = var.(dists_out)
    coefs_mix = 1 ./ vars
    # Fix NaN stemming from zero variance (which can occur if all training data
    # points that a rule matches have the same `y`).
    coefs_mix[.!ismissing.(vars) .&& iszero.(vars)] .= Inf

    lmodels = ConstantModel.(dists_out, coefs_mix, false)

    # Append the default rule.
    push!(conditions, maxgeneral(DX; x_min=x_min, x_max=x_max))
    push!(lmodels, ConstantModel(fit_mle(Normal, y), nextfloat(0.0), true))

    return Models.Model(conditions, lmodels; x_min=x_min, x_max=x_max)
end

"""
Evaluate the given solution genotype using the given fitness function.
"""
function evaluate(
    genotype::Genotype,
    X::XType,
    y::YType,
    x_min::Float64,
    x_max::Float64,
    ffunc::Function,
    nmatch_min::Int,
)
    phenotype = express(genotype, X, y, x_min, x_max, nmatch_min)
    fitness = ffunc(phenotype)
    return EvaluatedGenotype(genotype, phenotype, fitness)
end

function evaluate(
    solution::EvaluatedGenotype,
    X::XType,
    y::YType,
    x_min::Float64,
    x_max::Float64,
    ffunc::Function,
    nmatch_min::Int,
)
    phenotype = express(solution.genotype, X, y, x_min, x_max, nmatch_min)
    fitness = ffunc(phenotype)
    # Note that we don't return a report for `evaluate` yet. If you want to
    # implement this, consider the fact that `evaluate` typically gets
    # broadcasted over a vector of solutions/genotypes (and we should therefore
    # specialize for instead of broadcast over a vector so that reports are
    # properly concatenated instead of returning a vector of pairs).
    return EvaluatedGenotype(solution.genotype, phenotype, fitness)
end

struct GAResult
    best::EvaluatedGenotype
    pop_rest::AbstractVector{EvaluatedGenotype}
end

"""
Runs the GA based on the given config and training data.
"""
function runga end

function runga(X::XType, y::YType, config::GARegressor)
    # Note that this does not perform checks on the config but instead assumes
    # that it is valid.

    N, DX = size(X)

    rng = config.rng isa Int ? MersenneTwister(config.rng) : config.rng

    if config.fiteval == :mae
        ffunc = mkffunc(MAEFitness(X, y))
    elseif config.fiteval == :dissimilarity
        if config.dgmodel == nothing
            throw(
                ArgumentError(
                    "dgmodel != nothing required for fiteval == :dissimilarity",
                ),
            )
        end
        ffunc = mkffunc(DissimFitness(config.dgmodel, X))
    elseif config.fiteval == :likelihood
        ffunc = mkffunc(LikelihoodFitness(X, y))
    else
        throw(
            ArgumentError(
                "$(config.fiteval) is not a supported fitness scheme",
            ),
        )
    end

    # Cache initialization parameters.
    params_dist = if config.init == :inverse
        # Note that if the range is very wide this may be overly expensive.
        # TODO Consider to compute `selectparam` on-demand if very many lengths
        lens = (config.init_length_min):(config.init_length_max)
        df = selectparams("kdata", lens...)
        # TODO Make coverage configurable (but then we need to ensure that the
        # sample contains that coverage)
        # For now, we always take the highest-coverage parametrizations. Prepare
        # your sample accordingly.
        combine(
            groupby(df, [:DX, :K]),
            x -> sort(x, :rate_coverage_min)[end, :],
        )
    else
        nothing
    end

    # Initialize.
    #
    # Count number of evaluations.
    n_eval::Int = 0
    pop_uneval::Vector{Genotype}, report = if config.init == :inverse
        init_inverse(
            rng,
            ffunc,
            X,
            config.size_pop,
            config.x_min,
            config.x_max,
            params_dist,
        )
    else
        init_custom(
            rng,
            ffunc,
            X,
            config.size_pop,
            config.x_min,
            config.x_max,
            config.init_spread_min,
            config.init_spread_max,
            config.init_params_spread_a,
            config.init_params_spread_b,
            config.init_rate_coverage_min,
        )
    end
    pop_uneval[:], reports =
        unzip(repair.(Ref(rng), pop_uneval, Ref(X), Ref(config.nmatch_min)))
    pop::Vector{EvaluatedGenotype} =
        evaluate.(
            pop_uneval,
            Ref(X),
            Ref(y),
            config.x_min,
            config.x_max,
            ffunc,
            config.nmatch_min,
        )
    n_eval += length(pop)
    idx_best::Int = fittest_idx(pop)
    best::EvaluatedGenotype = pop[idx_best]
    len_best::Int = length(best)

    width_window::Int = 7
    lambda_window::Float64 = 0.0004

    # Bias factor for ryerkerk2020's biased window mechanism.
    bias_window::Float64 = 0

    for iter in 1:(config.n_iter)
        # TODO Consider to speed this up by inplace mutating a fixed array
        offspring_uneval = []

        idx_rand = reshape(randperm(config.size_pop), 2, :)

        # Iterating over columns is faster b/c of Julia arrays using
        # column-major order.
        for (idx1, idx2) in eachcol(idx_rand)
            if rand(rng) <= config.recomb_rate
                # TODO Gotta copy in crossover if needed
                g1, g2, report = crossover(rng, pop[idx1], pop[idx2])
            else
                g1, g2 = (deepcopy(pop[idx1]), deepcopy(pop[idx2]))
            end

            g1, report = mutate(rng, g1, X, config)
            g2, report = mutate(rng, g2, X, config)

            g1, report = repair(rng, g1, X, config.nmatch_min)
            g2, report = repair(rng, g2, X, config.nmatch_min)

            push!(offspring_uneval, g1)
            push!(offspring_uneval, g2)
        end

        offspring =
            evaluate.(
                offspring_uneval,
                Ref(X),
                Ref(y),
                Ref(config.x_min),
                Ref(config.x_max),
                Ref(ffunc),
                Ref(config.nmatch_min),
            )
        n_eval += length(offspring)

        # (4) in ryerkerk2020.
        len_lbound =
            min(Int(ceil(len_best - width_window / 2 + bias_window)), len_best)
        len_ubound = max(
            Int(floor(len_best + width_window / 2 + bias_window)),
            len_best,
        )
        lengths = collect(len_lbound:len_ubound)

        selection, report =
            select(rng, vcat(pop, offspring), config.size_pop, lengths)
        pop[:] = selection

        idx_best = fittest_idx(pop)
        best = pop[idx_best]

        len_best_prev = len_best
        len_best = length(best)
        # (3) in ryerkerk2020.
        bias_window =
            len_best - len_best_prev +
            bias_window * exp(-lambda_window * sqrt(abs(bias_window)))
    end

    deleteat!(pop, idx_best)
    report = (; bias_window=bias_window, n_eval=n_eval)
    return GAResult(best, pop), report
end

"""
Creates an initial population by drawing each solution from a distribution which
corresponds to one (randomly drawn) row in `params_dist`.

This somewhat approximates an initialization scheme where the user chooses a set
of solution lengths and then we repeatedly draw one of these lengths at random
and create an individual of that length until the configured population size is
reached. The only difference is that `init_inverse` probably does not fully hit
the chosen lengths due to the probabilistic nature of generating a random
solution based on minimum coverage termination criterion.
"""
function init_inverse(
    rng::AbstractRNG,
    ffunc::Function,
    X::XType,
    size_pop::Int,
    x_min::Float64,
    x_max::Float64,
    params_dist::DataFrame,
)
    N, DX = size(X)
    params_dist = subset(params_dist, :DX => dx -> dx .== DX)

    rows = sample(eachrow(params_dist), size_pop)

    pop::Vector{Genotype} = [
        draw_genotype(
            rng,
            # Draw intervals using the training data for checking the coverage
            # criterion. (An alternative would be to supply `DX` here which
            # would result in `draw_genotype` drawing a random sample from the
            # input space and trying to fulfil the coverage criterion on that.)
            X;
            spread_min=row.spread_min,
            params_spread=(a=row.a, b=row.b),
            rate_coverage_min=row.rate_coverage_min,
            remove_final_fully_overlapped=true,
            x_min=x_min,
            x_max=x_max,
        ) for row in rows
    ]
    # TODO Consider whether to sample differently (e.g. ensure that all lengths
    # are used at least once)

    report = (;)
    return pop, report
end

"""
Creates an initial population by drawing `size_pop` solutions from the solution
distribution defined by the `spread_*`, `params_spread_*` and
`rate_coverage_min` parameters.

Note that it can be difficult to control solution length this way (see
`init_inverse` for a way to deal with this).
"""
function init_custom(
    rng::AbstractRNG,
    ffunc::Function,
    X::XType,
    size_pop::Int,
    x_min::Float64,
    x_max::Float64,
    spread_min::Float64,
    spread_max::Float64,
    params_spread_a::Float64,
    params_spread_b::Float64,
    rate_coverage_min::Float64,
)
    N, DX = size(X)
    pop::Vector{Genotype} = [
        draw_genotype(
            rng,
            # Draw intervals using the training data for checking the coverage
            # criterion. (An alternative would be to supply `DX` here which
            # would result in `draw_genotype` drawing a random sample from the
            # input space and trying to fulfil the coverage criterion on that.)
            X;
            spread_min=spread_min,
            spread_max=spread_max,
            params_spread=(a=params_spread_a, b=params_spread_b),
            rate_coverage_min=rate_coverage_min,
            # We always remove fully overlapped rules during initialization.
            remove_final_fully_overlapped=true,
            # TODO Consider to reduce this number for faster initialization
            # n_samples=Parameters.n(dims),
            x_min=x_min,
            x_max=x_max,
        ) for _ in 1:(size_pop)
    ]
    report = (;)
    return pop, report
end

function fittest_idx(pop::AbstractVector{EvaluatedGenotype})
    return argmax(idx -> getproperty(pop[idx], :fitness), eachindex(pop))
end

function crossover end

function crossover(
    rng::AbstractRNG,
    g1::EvaluatedGenotype,
    g2::EvaluatedGenotype,
)
    return crossover(rng, g1.genotype, g2.genotype)
end

function crossover(rng::AbstractRNG, g1::Genotype, g2::Genotype)
    @warn "`crossover` not implemented yet" maxlog = 5
    report = (;)
    return deepcopy(g1), deepcopy(g2), report
end

"""
Ensure that each rule matches at least a certain number of training data points.
"""
function repair end

function repair(
    rng::AbstractRNG,
    g::EvaluatedGenotype,
    X::XType,
    nmatch_min::Int,
)
    return repair(rng, g.genotype, X, nmatch_min)
end

function repair(rng::AbstractRNG, g::Genotype, X::XType, nmatch_min::Int)
    # TODO Consider to make this repair! and mutate g (i.e. deleteat!) directly
    g_ = deepcopy(g)

    k = nmatch_min

    idx_rm = []
    for idx in eachindex(g_)
        if count(elemof(X, g_[idx])) < k
            push!(idx_rm, idx)
        end
    end
    deleteat!(g_, idx_rm)
    n_removed = length(idx_rm)
    if n_removed > 0
        @debug "Removed $n_removed conditions due to less than $k training " *
               "data matches" operator = "repair"
    end

    report = (; n_removed=n_removed)
    return g_, report
end
