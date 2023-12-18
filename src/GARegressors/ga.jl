Genotype = AbstractVector{Intervals.Interval}

const XType::DataType = AbstractMatrix{Float64}
const YType::DataType = AbstractVector{Float64}

draw_genotype = draw_intervals

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
Schemes for evaluating fitness.

Use `mkffunc` to instantiate these schemes (i.e. to create fitness functions).
"""
abstract type FitnessEvaluation end

"""
A mean absolute error–based fitness evaluation scheme.
"""
struct MAEFitness <: FitnessEvaluation
    X::XType
    y::YType
end

"""
An idealized dissimilarity-to-data-generating-process–based fitness evaluation
scheme.
"""
struct DissimFitness <: FitnessEvaluation
    model::Models.Model
    X::XType
end

# TODO Add CV-based fitness evaluation scheme
# TODO struct MLEFitness <: FitnessEvaluation end
# TODO struct MAPFitness <: FitnessEvaluation end

"""
Define a fitness measure based on a given `FitnessEvaluation` scheme.

A fitness measure is a function accepting an `RSLModels.Models.Model` and
returning a single floating point number.
"""
function mkffunc end

function mkffunc(fiteval::MAEFitness)
    function _cost(phenotype)
        y_pred = output_mean(phenotype, fiteval.X)
        return -mae(y_pred, fiteval.y)
    end

    return _cost
end

function mkffunc(fiteval::DissimFitness)
    function _cost(phenotype)
        return similarity(
            fiteval.model.conditions,
            phenotype.conditions;
            simf=simf_traversal_count_root(fiteval.X),
        )
    end
    return _cost
end

"""
Express the given genotype, yielding a phenotype.

This is achieved by fitting a rule set model for based on the set of conditions
specified by the genotype.
"""
function express(genotype::Genotype, X::XType, y::YType, x_min, x_max)
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
        if length(y_matched) < 2
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
function evaluate end

function evaluate(
    genotype::Genotype,
    X::XType,
    y::YType,
    x_min::Float64,
    x_max::Float64,
    ffunc::Function,
)
    phenotype = express(genotype, X, y, x_min, x_max)
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
)
    phenotype = express(solution.genotype, X, y, x_min, x_max)
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
    runga(DX, ffunc, rng, config)
"""
function runga end

function runga(config::GARegressor, X::XType, y::YType)
    # Note that this does not perform checks on the config but instead assumes
    # that it is valid.

    N, DX = size(X)

    rng = config.rng isa Int ? MersenneTwister(config.rng) : config.rng

    if config.fiteval == :mae
        ffunc = mkffunc(MAEFitness(X, y))
    elseif config.fiteval == :dissimilarity
        ffunc = mkffunc(DissimFitness(config.dgmodel, X))
    end

    # Initialize.
    pop, report = init(config, ffunc, X, y)
    pop = evaluate.(pop, Ref(X), Ref(y), config.x_min, config.x_max, ffunc)
    idx_best::Int = fittest_idx(pop)
    best::EvaluatedGenotype = pop[idx_best]
    len_best::Int = length(best)

    # TODO Expose width_window in config; ensure odd-valued (see ryerkerk2020)
    width_window::Int = 7
    # TODO Expose lambda_window in config; see ryerkerk2020
    lambda_window::Float64 = 0.0004

    # Bias factor for ryerkerk2020's biased window mechanism.
    bias_window::Float64 = 0

    for iter in 1:(config.n_iter)
        sols_new = []

        idx_rand = reshape(randperm(config.size_pop), 2, :)

        # Iterating over columns is faster b/c of Julia arrays using
        # column-major order.
        for (idx1, idx2) in eachcol(idx_rand)
            if rand(rng) <= config.rate_crossover
                # TODO Gotta copy in crossover if needed
                g1, g2, report = crossover(rng, pop[idx1], pop[idx2])
            else
                g1, g2 = (deepcopy(pop[idx1]), deepcopy(pop[idx2]))
            end

            g1, report = mutate(rng, g1, X, config)
            g2, report = mutate(rng, g2, X, config)

            g1, report = repair(rng, g1, X)
            g2, report = repair(rng, g2, X)

            push!(sols_new, g1)
            push!(sols_new, g2)
        end

        sols_new, report =
            evaluate.(
                sols_new,
                Ref(X),
                Ref(y),
                Ref(config.x_min),
                Ref(config.x_max),
                Ref(ffunc),
            )

        # (4) in ryerkerk2020.
        len_lbound =
            min(Int(ceil(len_best - width_window / 2 + bias_window)), len_best)
        len_ubound = max(
            Int(floor(len_best + width_window / 2 + bias_window)),
            len_best,
        )
        lengths = collect(len_lbound:len_ubound)

        pop[:], report = select(rng, pop, config.size_pop, lengths)

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
    return GAResult(best, pop)
end

function init(config, ffunc, X, y)
    N, DX = size(X)
    pop = [
        draw_genotype(
            DX;
            spread_min=config.spread_min,
            # TODO Extract remaining parameters to config
            spread_max=config.spread_max,
            params_spread=(a=config.params_spread_a, b=config.params_spread_b),
            rate_coverage_min=0.8,
            remove_final_fully_overlapped=true,
            # TODO Consider to reduce this number for faster initialization
            # n_samples
            x_min=config.x_min,
            x_max=config.x_max,
        ) for _ in 1:(config.size_pop)
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
    @warn "`crossover` not implemented yet"
    report = (;)
    return deepcopy(g1), deepcopy(g2), report
end

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
    N, DX = size(X)

    g_ = deepcopy(g)

    # Number of metavariables.
    l = length(g_)

    # We have `2 * DX` design variables per metavariable (each condition is one
    # metavariable).
    n_dvars = 2 * DX

    # TODO Look at efficiency of this combination of loops and rand

    # Go over all metavariables (i.e. all conditions).
    for idx in eachindex(g_)
        for idx_lower in eachindex(g_[idx].lbound)
            if rand(rng) < 1.0 / (n_dvars * l)
                val = g_[idx].lbound[idx_lower]
                g_[idx].lbound[idx_lower] =
                # TODO Expose rate_mu parameter
                # TODO Check whether used std is sensible here
                    rand(Normal(val, 0.05 * (config.x_max - config.x_min)))
            end
        end
    end

    # TODO Expose p_mu_add
    # Add a metavariable.
    if rand(rng) < 0.05
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
Ensure that each rule matches at least a certain number of training data points.
"""
function repair end

function repair(rng::AbstractRNG, g::EvaluatedGenotype, X::XType)
    return repair(rng, g.genotype, X)
end

function repair(rng::AbstractRNG, g::Genotype, X::XType)
    g_ = deepcopy(g)

    # TODO Expose repair k parameter/derive meaningful value
    k = 2

    idx_rm = []
    for idx in eachindex(g_)
        if count(elemof(X, g_[idx])) < k
            push!(idx_rm, idx)
        end
    end
    deleteat!(g_, idx_rm)
    n_removed = length(idx_rm)
    if n_removed > 0
        @info "Removed $n_removed conditions due to less than $k training " *
              "data matches" operator = "repair"
    end

    report = (n_removed = n_removed)
    return g_, report
end
