module LocalModels

using AutoHashEquals
using Distributions
using LinearAlgebra
using Mmap
using Random
using StatsBase

using ..AbstractModels

export ConstantModel, draw_constantmodel

str_independent = """
Note that, for constant models, the output distribution is independent from
inputs (i.e. only the number of inputs matters for this function).
"""

@auto_hash_equals struct ConstantModel
    "Output distribution."
    dist_out::Distribution
    "Mixing coefficient."
    coef_mix::Real
    # Whether this is a default rule (and should thus not undergo mixing
    # coefficient fixing).
    isdefault::Bool
    function ConstantModel(
        dist_out::Distribution,
        coef_mix::Real,
        # Whether this is a default rule (and should thus not undergo mixing
        # coefficient fixing).
        isdefault::Bool,
    )
        return new(dist_out, coef_mix, isdefault)
    end
    function ConstantModel(
        dist_out::Distribution,
        coef_mix::Real;
        # Whether this is a default rule (and should thus not undergo mixing
        # coefficient fixing).
        isdefault::Bool=false,
    )
        return new(dist_out, coef_mix, isdefault)
    end
end

function draw_constantmodel(isdefault::Bool=false)
    return draw_constantmodel(Random.default_rng(); isdefault=isdefault)
end

function draw_constantmodel(rng::AbstractRNG; isdefault::Bool=false)
    mean_out = rand(rng, Uniform(0.0, 1.0))

    # TODO Derive more sensible and less arbitrary min and max noise values,
    # e.g. based on coef distribution bounds
    std_noise_out_min = 0.001
    std_noise_out = rand(rng, Uniform(std_noise_out_min, 0.2))
    # TODO Consider to fix coef_mix based on other rules
    if isdefault
        # If this is a default rule, set its mixing weight to the smallest
        # positive number.
        coef_mix = nextfloat(0.0)
    else
        coef_mix = 1.0 / rand(rng, Uniform(std_noise_out, 1.0))
    end
    return ConstantModel(
        Normal(mean_out, std_noise_out),
        coef_mix;
        isdefault=isdefault,
    )
end

"""
    output([rng::AbstractRNG], model::ConstantModel[, X::Float64])

Since constant models' outputs are independent of inputs `X`, providing an input
`X` is optional (providing it doesn't do anything and is only allowed to allow
for a unified interface).
"""
function AbstractModels.output(model::ConstantModel)
    return AbstractModels.output(Random.default_rng(), model)
end

function AbstractModels.output(rng::AbstractRNG, model::ConstantModel)
    # Note that outputs of constant models do not depend on inputs.
    return rand(rng, model.dist_out)
end

function AbstractModels.output(model::ConstantModel, X::Float64)
    return AbstractModels.output(Random.default_rng(), model)
end

function AbstractModels.output(
    rng::AbstractRNG,
    model::ConstantModel,
    X::Float64,
)
    # Note that outputs of constant models do not depend on inputs.
    return rand(rng, model.dist_out)
end

"""
    output([rng::AbstractRNG], model::ConstantModel, X::AbstractMatrix{Float64}; usemmap=false)

Given inputs `X`, sample for each the respective output distribution.

$str_independent
"""
function AbstractModels.output(
    model::ConstantModel,
    X::AbstractMatrix{Float64};
    usemmap=false,
)
    return AbstractModels.output(
        Random.default_rng(),
        model,
        X;
        usemmap=usemmap,
    )
end

function AbstractModels.output(
    rng::AbstractRNG,
    model::ConstantModel,
    X::AbstractMatrix{Float64};
    usemmap=false,
)
    N = size(X, 1)
    if usemmap
        (path_y, io_y) = mktemp(tempdir())
        y = mmap(io_y, Vector{Float64}, N)
        for i in 1:N
            y[i] = AbstractModels.output(rng, model)
        end
        return y
    else
        return [AbstractModels.output(rng, model) for _ in 1:N]
    end
end

# TODO Implement usemmap for output_mean?
# TODO Implement usemmap for output_variance?
"""
    output_mean(model::ConstantModel, X::AbstractMatrix{Float64})

Given inputs `X`, return the output distribution's mean for each.

$str_independent
"""
function AbstractModels.output_mean(
    model::ConstantModel,
    X::AbstractMatrix{Float64},
)
    mean_out = mean(model.dist_out)
    return repeat([mean_out], size(X, 1))
end

"""
    output_mean(models::AbstractVector{ConstantModel}, X::AbstractMatrix{Float64})

Given inputs `X`, return for each model its the output distribution mean for
each input.

$str_independent
"""
function AbstractModels.output_mean(
    models::AbstractVector{ConstantModel},
    X::AbstractMatrix{Float64},
)
    return hcat(map(model -> AbstractModels.output_mean(model, X), models)...)
end

"""
    output_variance(model::ConstantModel, X::AbstractMatrix{Float64})

Given inputs `X`, return the output distribution's variance for each.

$str_independent
"""
function AbstractModels.output_variance(
    model::ConstantModel,
    X::AbstractMatrix{Float64},
)
    return repeat([model.dist_out.σ], size(X, 1))
end

end
