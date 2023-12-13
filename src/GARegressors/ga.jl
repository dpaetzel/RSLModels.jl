Genotype = Vector{Intervals.Interval}

draw_genotype = draw_intervals

"""
Wrapper to store solutions together with the models they induce as well as their
fitness score.
"""
struct EvaluatedGenotype
    genotype::Genotype
    phenotype::Models.Model
    fitness::Float64
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
    X::AbstractMatrix
    y::AbstractVector
end

"""
An idealized dissimilarity-to-data-generating-process–based fitness evaluation
scheme.
"""
struct DissimFitness <: FitnessEvaluation
    model::Models.Model
    X::AbstractMatrix
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
function express(genotype::Genotype, X, y, x_min, x_max)
    # TODO Consider to cache matching
    N, DX = size(X)

    conditions = deepcopy(genotype)

    dists_out = []
    idx_rm = []
    for idx in eachindex(conditions)
        y_matched = view(y, elemof(X, conditions[idx]))
        # TODO Expose this parameter as match_min or something like that
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

function evaluate(genotype::Genotype, X, y, x_min, x_max, ffunc)
    phenotype = express(genotype, X, y, x_min, x_max)
    fitness = ffunc(phenotype)
    return EvaluatedGenotype(genotype, phenotype, fitness)
end

function evaluate(solution::EvaluatedGenotype, X, y, x_min, x_max, ffunc)
    phenotype = express(solution.genotype, X, y, x_min, x_max)
    fitness = ffunc(phenotype)
    return EvaluatedGenotype(solution.genotype, phenotype, fitness)
end

struct GAResult
    best::EvaluatedGenotype
    pop_rest::Vector{EvaluatedGenotype}
end

"""
    runga(DX, ffunc, rng, config)
"""
function runga end

function runga(config::GARegressor, X, y)
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
    pop = [
        evaluate(
            draw_genotype(
                DX;
                spread_min=config.spread_min,
                # TODO Extract remaining parameters to config
                spread_max=Inf,
                params_spread=(a=1.0, b=1.0),
                rate_coverage_min=0.8,
                remove_final_fully_overlapped=true,
                # TODO Consider to reduce this number for faster initialization
                # n_samples
                x_min=config.x_min,
                x_max=config.x_max,
            ),
            X,
            y,
            config.x_min,
            config.x_max,
            ffunc,
        ) for _ in 1:(config.size_pop)
    ]

    for iter in 1:(config.n_iter)
        sols_new = []

        idx_rand = reshape(randperm(config.size_pop), 2, :)

        # Iterating over columns is faster b/c of Julia arrays using
        # column-major order.
        for (idx1, idx2) in eachcol(idx_rand)
            if rand(rng) <= config.rate_crossover
                # TODO Gotta copy in crossover if needed
                x1, x2 = crossover(rng, pop[idx1], pop[idx2])
            else
                x1, x2 = (deepcopy(pop[idx1]), deepcopy(pop[idx2]))
            end

            x1 = mutate(rng, x1)
            x2 = mutate(rng, x2)

            x1 = repair(rng, x1)
            x2 = repair(rng, x2)

            push!(sols_new, x1)
            push!(sols_new, x2)
        end

        sols_new =
            evaluate.(
                sols_new,
                Ref(X),
                Ref(y),
                Ref(config.x_min),
                Ref(config.x_max),
                Ref(ffunc),
            )
        pop[:] = select(rng, pop, sols_new)
    end

    idx_best = argmax(idx -> getproperty(pop[idx], :fitness), eachindex(pop))
    best = pop[idx_best]
    deleteat!(pop, idx_best)
    return GAResult(best, pop)
end

function crossover(rng, x1, x2)
    @warn "`crossover` not implemented yet"
    return deepcopy(x1), deepcopy(x2)
end

function mutate(rng, x)
    @warn "`mutate` not implemented yet"
    x_ = deepcopy(x)

    # Change a design variable.
    # if rand(rng) <

    # Add a metavariable.
    # if rand(rng) < 0.05
    # end
    # Remove a metavariable.
    # if rand(rng) < 0.05
    # end
    return x_
end

function repair(rng, x)
    @warn "`repair` not implemented yet"
    x_ = deepcopy(x)

    # TODO Ensure that at least k training data points are matched

    return x_
end

function select(rng, pop, sols_new)
    @warn "`select` not implemented yet"

    # Simplistic tournament selection.
    size_trnmt = 4

    size_pop = size(pop, 1)
    pop = [pop; sols_new]

    pop_new = []
    while size(pop_new, 1) < size_pop
        idx_trnmt = rand(1:size_pop, size_trnmt)
        winner = argmax(model -> getproperty(model, :fitness), pop[idx_trnmt])
        push!(pop_new, winner)
    end

    return pop_new
end
