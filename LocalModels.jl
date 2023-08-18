module LocalModels

using Distributions
using LinearAlgebra
using Random
using StatsBase

export ConstantModel, draw_constantmodel, output

struct ConstantModel
    coef::Real
    dist_noise::Distribution
    coef_mix::Real
end

function draw_constantmodel()
    return draw_constantmodel(Random.default_rng())
end

function draw_constantmodel(rng::AbstractRNG)
    coef = rand(rng, Uniform(0.0, 1.0))

    # TODO Derive more sensible and less arbitrary min and max noise values,
    # e.g. based on coef distribution bounds
    std_noise_min = 0.001
    std_noise = rand(rng, Uniform(std_noise_min, 0.2))
    # TODO Consider to fix coef_mix based on other rules
    coef_mix = rand(rng, Uniform(0.0, 1.0))
    return ConstantModel(coef, Normal(0.0, std_noise), coef_mix)
end

function output(model::ConstantModel)
    return output(Random.default_rng(), model)
end

function output(rng::AbstractRNG, model::ConstantModel)
    # Note that outputs of constant models do not depend on inputs.
    return model.coef + rand(rng, model.dist_noise)
end

function output(models::AbstractVector{ConstantModel})
    return output(Random.default_rng(), models)
end

"""
Assumes that all the given models are responsible (i.e. mixes all of them).
"""
function output(rng::AbstractRNG, models::AbstractVector{ConstantModel})
    outs = map(model -> output(rng, model), models)
    coefs_mix = map(model -> model.coef_mix, models)
    mixing = coefs_mix ./ sum(coefs_mix)
    return vec(sum(mixing .* outs))
end

function output(models::AbstractVector{ConstantModel}; matching_matrix::AbstractMatrix)
    return output(Random.default_rng(), models; matching_matrix=matching_matrix)
end

function output(
    rng::AbstractRNG,
    models::AbstractVector{ConstantModel};
    matching_matrix::AbstractMatrix,
)
    outs = map(model -> output(rng, model), models)
    coefs_mix = map(model -> model.coef_mix, models)
    coefs_mix = coefs_mix' .* matching_matrix
    mixing = coefs_mix ./ sum(coefs_mix)
    # The first dimension of the matching matrix (and therefore of the
    # data structure that this results in) is the dimension of the number of
    # input data points. The second dimension are the models, therefore sum over
    # the second dimension.
    return vec(sum(outs' .* mixing; dims=2))
end

end
