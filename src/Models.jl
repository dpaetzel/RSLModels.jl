module Models

using AutoHashEquals
using Distributions
using Mmap
using Random

using ..AbstractModels
using ..Intervals
using ..LocalModels
using ..Parameters

export Model, dimensions, draw_data, draw_model, match
export output, output_mean, output_variance

const X_MIN::Float64 = Intervals.X_MIN
const X_MAX::Float64 = Intervals.X_MAX

@auto_hash_equals struct Model
    conditions::AbstractVector
    local_models::AbstractVector
    x_min::Float64
    x_max::Float64
    function Model(
        conditions::AbstractVector,
        local_models::AbstractVector;
        x_min=X_MIN,
        x_max=X_MAX,
    )
        dims = Intervals.dimensions(conditions[1])
        if length(conditions) != length(local_models)
            error(
                "number of conditions and number of local models are not equal",
            )
        elseif !all(Intervals.dimensions.(conditions) .== dims)
            error("conditions have to have the same dimension")
        end
        return new(conditions, local_models, x_min, x_max)
    end
end

function Intervals.dimensions(model::Model)
    return Intervals.dimensions(model.conditions[1])
end

function draw_model(
    dims::Integer;
    # Note that these are the same default values as the ones given for
    # `Intervals.draw_intervals`.
    spread_min=0.0,
    spread_max=Inf,
    params_spread::NamedTuple{(:a, :b),Tuple{Float64,Float64}}=(a=1.0, b=1.0),
    rate_coverage_min::Float64=0.8,
    remove_final_fully_overlapped::Bool=true,
    x_min=X_MIN,
    x_max=X_MAX,
)
    return draw_model(
        Random.default_rng(),
        dims;
        spread_min=spread_min,
        spread_max=spread_max,
        params_spread=params_spread,
        rate_coverage_min=rate_coverage_min,
        remove_final_fully_overlapped=remove_final_fully_overlapped,
        x_min=x_min,
        x_max=x_max,
    )
end

function draw_model(
    rng::AbstractRNG,
    dims::Integer;
    # Note that these are the same default values as the ones given for
    # `Intervals.draw_intervals`.
    spread_min=0.0,
    spread_max=Inf,
    params_spread::NamedTuple{(:a, :b),Tuple{Float64,Float64}}=(a=1.0, b=1.0),
    rate_coverage_min::Float64=0.8,
    remove_final_fully_overlapped::Bool=true,
    x_min=X_MIN,
    x_max=X_MAX,
)
    conditions = draw_intervals(
        rng,
        dims;
        spread_min=spread_min,
        spread_max=spread_max,
        params_spread=params_spread,
        rate_coverage_min=rate_coverage_min,
        remove_final_fully_overlapped=remove_final_fully_overlapped,
        x_min=x_min,
        x_max=x_max,
    )
    local_models = [draw_constantmodel(rng) for _ in 1:length(conditions)]

    # Append the default rule. `push!` is faster than other things I tried.
    push!(conditions, maxgeneral(dims; x_min=x_min, x_max=x_max))
    push!(local_models, draw_constantmodel(rng; isdefault=true))

    return Model(conditions, local_models; x_min=x_min, x_max=x_max)
end

function draw_inputs(
    dims::Integer,
    n::Int=Parameters.n(dims);
    x_min=X_MIN,
    x_max=X_MAX,
)
    return draw_inputs(Random.default_rng(), dims, n; x_min=x_min, x_max=x_max)
end

function draw_inputs(
    rng::AbstractRNG,
    dims::Integer,
    n::Int=Parameters.n(dims);
    x_min=X_MIN,
    x_max=X_MAX,
    usemmap=false,
)
    if usemmap
        (path_X, io_X) = mktemp()
        X = mmap(io_X, Array{Float64,2}, (n, dims))
        rand!(rng, Uniform(x_min, x_max), X)
        return X
    else
        return rand(rng, Uniform(x_min, x_max), n, dims)
    end
end

function match(model::Model, X::AbstractMatrix; usemmap=false)
    return elemof(X, model.conditions; usemmap=usemmap)
end

function draw_data(model::Model, n; usemmap=false)
    return draw_data(Random.default_rng(), model, n; usemmap=usemmap)
end

function draw_data(rng::AbstractRNG, model::Model, n; usemmap=false)
    X = draw_inputs(
        rng,
        dimensions(model),
        n;
        x_min=model.x_min,
        x_max=model.x_max,
        usemmap=usemmap,
    )
    matching_matrix = match(model, X; usemmap=usemmap)
    y = output(rng, model.local_models, X, matching_matrix; usemmap=usemmap)

    return X, y, matching_matrix
end

"""
    output([rng::AbstractRNG], model::Model, X; <keyword arguments>)

Given inputs `X`, sample for each the respective output distribution.

# Arguments

- `rng::AbstractRNG`: RNG to use for sampling.
- `model::Model`: Model for generating outputs.
- `X::AbstractMatrix{Float64}`: Inputs to generate output for.
- `matching_matrix::AbstractMatrix{Bool}=match(model, X)`: Precomputed matching
  matrix.
- `usemmap::Bool=false`: Whether to memory-map large arrays.
"""
function AbstractModels.output(
    model::Model,
    X::AbstractMatrix{Float64};
    matching_matrix::AbstractMatrix{Bool}=match(model, X),
    usemmap=false,
)
    return AbstractModels.output(
        Random.default_rng(),
        model,
        X;
        matching_matrix=matching_matrix,
        usemmap=usemmap,
    )
end

function AbstractModels.output(
    rng::AbstractRNG,
    model::Model,
    X::AbstractMatrix{Float64};
    matching_matrix::AbstractMatrix{Bool}=match(model, X),
    usemmap=false,
)
    N = size(X, 1)
    models = model.local_models
    K = length(models)

    if usemmap
        # Note that this duplicates code somewhat from `mixing` (see below).
        (path_y, io_y) = mktemp(tempdir())
        y = mmap(io_y, Vector{Float64}, N)
        outs = Vector{Float64}(undef, K)
        mixs = Vector{Float64}(undef, K)
        for n in 1:N
            for k in 1:K
                outs[k] = AbstractModels.output(rng, models[k], X[n])
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
        outs =
            hcat(map(model -> AbstractModels.output(rng, model, X), models)...)
        return mix(models, outs, matching_matrix)
    end
end

"""
    output_mean(model::Model, X; matching_matrix=match(model, X))

Given inputs `X`, return the output distribution's mean for each.
"""
function AbstractModels.output_mean(
    model::Model,
    X::AbstractMatrix{Float64};
    matching_matrix::AbstractMatrix{Bool}=match(model, X),
)
    outs = AbstractModels.output_mean(model.local_models, X)
    return mix(model.local_models, outs, matching_matrix)
end

"""
    output_variance(model::Model, X; matching_matrix=match(model, X))

Given inputs `X`, return the output distribution's variance for each.
"""
function AbstractModels.output_variance(
    model::Model,
    X::AbstractMatrix{Float64};
    matching_matrix::AbstractMatrix{Bool}=match(model, X),
)
    vars = [m.dist_noise.σ for m in model.local_models]
    mix = mixing(model.local_models, matching_matrix)
    return vec(sum(vars' .* mix .^ 2; dims=2))
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

"""
Mix the given outputs of the given models based on the given matching matrix.
"""
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

end
