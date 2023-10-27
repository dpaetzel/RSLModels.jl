module LocalModels

using AutoHashEquals
using Distributions
using LinearAlgebra
using Mmap
using Random
using StatsBase

export ConstantModel,
    draw_constantmodel, mixing, mix, output, output_mean, output_variance

@auto_hash_equals struct ConstantModel
    coef::Real
    dist_noise::Distribution
    coef_mix::Real
    # Whether this is a default rule (and should thus not undergo mixing
    # coefficient fixing).
    isdefault::Bool
    function ConstantModel(
        coef::Real,
        dist_noise::Distribution,
        coef_mix::Real,
        # Whether this is a default rule (and should thus not undergo mixing
        # coefficient fixing).
        isdefault::Bool,
    )
        return new(coef, dist_noise, coef_mix, isdefault)
    end
    function ConstantModel(
        coef::Real,
        dist_noise::Distribution,
        coef_mix::Real;
        # Whether this is a default rule (and should thus not undergo mixing
        # coefficient fixing).
        isdefault::Bool=false,
    )
        return new(coef, dist_noise, coef_mix, isdefault)
    end
end

function draw_constantmodel(isdefault::Bool=false)
    return draw_constantmodel(Random.default_rng(); isdefault=isdefault)
end

function draw_constantmodel(rng::AbstractRNG; isdefault::Bool=false)
    coef = rand(rng, Uniform(0.0, 1.0))

    # TODO Derive more sensible and less arbitrary min and max noise values,
    # e.g. based on coef distribution bounds
    std_noise_min = 0.001
    std_noise = rand(rng, Uniform(std_noise_min, 0.2))
    # TODO Consider to fix coef_mix based on other rules
    if isdefault
        # If this is a default rule, set its mixing weight to the smallest
        # positive number.
        coef_mix = nextfloat(0.0)
    else
        coef_mix = 1.0 / rand(rng, Uniform(std_noise, 1.0))
    end
    return ConstantModel(
        coef,
        Normal(0.0, std_noise),
        coef_mix;
        isdefault=isdefault,
    )
end

# The output at a certain point is independent of the input because of the model
# being constant.
function output(model::ConstantModel)
    return output(Random.default_rng(), model)
end

function output(rng::AbstractRNG, model::ConstantModel)
    # Note that outputs of constant models do not depend on inputs.
    return model.coef + rand(rng, model.dist_noise)
end

# Predicting with a constant model for several inputs does not depend on the
# inputs themselves but only on the number of them. We nevertheless provide `X`
# fully here for abstraction reasons (e.g. if we add another kind of local model
# that actually does depend on the inputs).
function output(
    model::ConstantModel,
    X::AbstractMatrix{Float64};
    usemmap=false,
)
    return output(Random.default_rng(), model, X; usemmap=usemmap)
end

function output(
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
            y[i] = output(rng, model)
        end
        return y
    else
        return [output(rng, model) for _ in 1:N]
    end
end

function output(
    models::AbstractVector{ConstantModel},
    X::AbstractMatrix{Float64},
    matching_matrix::AbstractMatrix{Bool};
    usemmap=false,
)
    return output(
        Random.default_rng(),
        models,
        X,
        matching_matrix;
        usemmap=usemmap,
    )
end

function output(
    rng::AbstractRNG,
    models::AbstractVector{ConstantModel},
    X::AbstractMatrix{Float64},
    matching_matrix::AbstractMatrix{Bool};
    usemmap=false,
)
    N = size(X, 1)
    K = length(models)

    if usemmap
        # Note that this duplicates code somewhat from `mixing` (see below).
        (path_y, io_y) = mktemp(tempdir())
        y = mmap(io_y, Vector{Float64}, N)
        outs = Vector{Float64}(undef, K)
        mixs = Vector{Float64}(undef, K)
        for n in 1:N
            for k in 1:K
                # This would be more general but constant local models don't
                # care about x.
                # outs[k] = output(rng, models[k], X[n])
                outs[k] = output(rng, models[k])
                mixs[k] = models[k].coef_mix * matching_matrix[n, k]
            end
            # Mixing coefficients are allowed to be `Inf` but these cases need
            # to be handled properly. If at least one of the mixing coefficients
            # is `Inf`, we set all non-`Inf` mixing coefficients to 0.0 and all
            # `Inf`s to 1.0. This yields equal mixing between the rules with
            # `Inf`s.
            if any(isinf.(mixs))
                mixs[(!).(isinf.(mixs))] .= 0.0
                mixs[isinf.(mixs)] .= 1.0
            end
            y[n] = sum(outs .* (mixs ./ sum(mixs)))
        end
        return y
    else
        outs = hcat(map(model -> output(rng, model, X), models)...)
        return mix(models, outs, matching_matrix)
    end
end

function output_mean(model::ConstantModel, X::AbstractMatrix{Float64})
    return repeat([model.coef], size(X, 1))
end

function output_mean(
    models::AbstractVector{ConstantModel},
    X::AbstractMatrix{Float64},
)
    return hcat(map(model -> output_mean(model, X), models)...)
end

function output_mean(
    models::AbstractVector{ConstantModel},
    X::AbstractMatrix{Float64},
    matching_matrix::AbstractMatrix{Bool},
)
    # TODO Consider to abstract from `output_mean` vs `output` (only difference
    # to `output`)
    outs = output_mean(models, X)
    return mix(models, outs, matching_matrix)
end

function output_variance(model::ConstantModel, X::AbstractMatrix{Float64})
    return repeat([model.dist_noise.σ], size(X, 1))
end

function output_variance(
    models::AbstractVector{ConstantModel},
    X::AbstractMatrix{Float64},
    matching_matrix::AbstractMatrix{Bool},
)
    vars = [m.dist_noise.σ for m in models]
    mix = mixing(models, matching_matrix)
    return vec(sum(vars' .* mix .^ 2; dims=2))
end

"""
Check the matching matrix for all data points being matched by at least one
rule.
"""
function check_matching_matrix(matching_matrix::AbstractMatrix{Bool})
    # Check whether any row is all zeroes.
    if any(sum(matching_matrix; dims=2) .== 0)
        println(matching_matrix)
        error("some samples are unmatched, add a default rule")
    end
end

"""
Compute a mixing weight for each local model for each data point.
"""
function mixing(
    models::AbstractVector{ConstantModel},
    matching_matrix::AbstractMatrix{Bool},
)
    check_matching_matrix(matching_matrix)

    coefs_mix = [model.coef_mix for model in models]
    coefs_mix = coefs_mix' .* matching_matrix

    # Mixing coefficients are allowed to be `Inf` but these cases need to be
    # handled properly. If at least one of the mixing coefficients is `Inf`, we
    # set all non-`Inf` mixing coefficients to 0.0 and all `Inf`s to 1.0. This
    # yields equal mixing between the rules with `Inf`s.
    for row in eachrow(coefs_mix)
        if any(isinf.(row))
            row[(!).(isinf.(row))] .= 0.0
            row[isinf.(row)] .= 1.0
        end
    end

    return coefs_mix ./ sum(coefs_mix; dims=2)
end

function mix(
    models::AbstractVector{ConstantModel},
    outputs::AbstractMatrix{Float64},
    matching_matrix::AbstractMatrix{Bool},
)

    # The first dimension of the matching matrix (and therefore of the
    # data structure that this results in) is the dimension of the number of
    # input data points. The second dimension are the models, therefore sum over
    # the second dimension.
    return vec(sum(outputs .* mixing(models, matching_matrix); dims=2))
end

end
