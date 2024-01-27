"""
Schemes for evaluating fitness.

Use `mkffunc` to instantiate these schemes (i.e. to create fitness functions).
"""
abstract type FitnessEvaluation end

"""
A mean absolute error–based fitness evaluation scheme.
"""
struct NegMAEFitness <: FitnessEvaluation
    X::XType
    y::YType
end

"""
An idealized dissimilarity-to-data-generating-process–based fitness evaluation
scheme.
"""
struct SimFitness <: FitnessEvaluation
    model::Models.Model
    X::XType
end

"""
A log-likelihood–based fitness, i.e. fitness corresponds to

```math
\\log \\prod_i p(y_i | θ, X_i) = \\sum_i \\log p(y_i | θ, X_i)
```

where `X`, `y` is the training data and `θ` are the model parameters.
"""
struct LikelihoodFitness <: FitnessEvaluation
    X::XType
    y::YType
end

"""
A log-posterior-probability-based fitness, i.e. fitness corresponds to

```math
\\log (p(θ) \\prod_i p(y_i | θ, X_i)) = \\log p(θ) + \\sum_i \\log p(y_i | θ, X_i)
```

where p(θ) is the prior on the model parameters. In our case, it has a Negative
Binomial distribution that depends solely on the number of rules in the model

```math
p(θ) = p(K) = NegativeBinomial(2, 0.2)
```

This encodes the following beliefs:

- mean: 8.0
- mode: 4
- central 50%: [3, 11]
- central 80%: [1, 16]
- central 95%: [1, 20]
- last 5%: [20, ∞)
"""
struct PosteriorFitness <: FitnessEvaluation
    X::XType
    y::YType
end

"""
A negative AICc-based fitness.
"""
struct NegAICFitness <: FitnessEvaluation
    X::XType
    y::YType
end

# TODO Add more fitness evaluation schemes
# TODO struct CVFitness <: FitnessEvaluation end

"""
Define a fitness measure based on a given `FitnessEvaluation` scheme.

A fitness measure is a function accepting an `RSLModels.Models.Model` and
returning a single floating point number.
"""
function mkffunc end

function mkffunc(fiteval::NegMAEFitness)
    function _fitness(phenotype)
        y_pred = output_mean(phenotype, fiteval.X)
        return -mae(y_pred, fiteval.y)
    end

    return _fitness
end

function mkffunc(fiteval::SimFitness)
    function _fitness(phenotype)
        return similarity(
            fiteval.model.conditions,
            phenotype.conditions;
            simf=simf_traversal_count_root(fiteval.X),
        )
    end
    return _fitness
end

function mkffunc(fiteval::LikelihoodFitness)
    function _fitness(phenotype)
        y_dist = output_dist(phenotype, fiteval.X)
        return sum(logpdf.(y_dist, fiteval.y))
    end

    return _fitness
end

function mkffunc(fiteval::PosteriorFitness)
    prior = NegativeBinomial(2, 0.2)
    function _fitness(phenotype)
        y_dist = output_dist(phenotype, fiteval.X)
        loglikelihood = sum(logpdf.(y_dist, fiteval.y))
        logprior = logpdf(prior, length(phenotype.conditions))
        return loglikelihood + logprior
    end

    return _fitness
end

function mkffunc(fiteval::NegAICFitness)
    function _fitness(phenotype)
        y_dist = output_dist(phenotype, fiteval.X)
        loglikelihood = sum(logpdf.(y_dist, fiteval.y))
        N, DX = size(fiteval.X)
        K = length(phenotype.conditions)
        nparams = 2 * K * DX + 2 * K
        AICc =
            -2 * loglikelihood +
            2 * nparams +
            (2 * nparams * (nparams + 1)) / (N - nparams - 1)
        # Since fitness is maximized, we invert this.
        return -AICc
    end

    return _fitness
end
