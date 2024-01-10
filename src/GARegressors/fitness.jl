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

"""
A log-likelihood–based fitness, i.e. fitness corresponds to

```math
log \\prod_i p(y_i | θ, X_i) = \\sum_i log p(y_i | θ, X_i)
```

where `X`, `y` is the training data and `θ` are the model parameters.
"""
struct LikelihoodFitness <: FitnessEvaluation
    X::XType
    y::YType
end

# TODO Add more fitness evaluation schemes
# TODO struct CVFitness <: FitnessEvaluation end
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

function mkffunc(fiteval::LikelihoodFitness)
    function _cost(phenotype)
        y_dist = output_dist(phenotype, fiteval.X)
        return sum(logpdf.(y_dist, fiteval.y))
    end

    return _cost
end
