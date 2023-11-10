module Models

using AutoHashEquals
using Distributions
using Mmap
using Random

using ..Intervals
using ..LocalModels
using ..Parameters

export Model, dimensions, draw_data, draw_model, match

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

function LocalModels.output(
    model::Model,
    X::AbstractMatrix{Float64},
    matching_matrix::AbstractMatrix{Bool},
)
    return output(model.local_models, X, matching_matrix)
end

function LocalModels.output_mean(
    model::Model,
    X::AbstractMatrix{Float64},
    matching_matrix::AbstractMatrix{Bool},
)
    return output_mean(model.local_models, X, matching_matrix)
end

function LocalModels.output_mean(model::Model, X::AbstractMatrix{Float64})
    matching_matrix = match(model, X)
    return output_mean(model.local_models, X, matching_matrix)
end

function LocalModels.output_variance(model::Model, X::AbstractMatrix{Float64})
    matching_matrix = match(model, X)
    return output_variance(model.local_models, X, matching_matrix)
end

function LocalModels.output_variance(
    model::Model,
    X::AbstractMatrix{Float64},
    matching_matrix::AbstractMatrix{Bool},
)
    return output_variance(model.local_models, X, matching_matrix)
end

end
