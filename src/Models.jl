module Models

using Random
using Distributions

using ..Intervals
using ..LocalModels

export Model, dimensions, draw_model, match, draw_data

const X_MIN::Float64 = Intervals.X_MIN
const X_MAX::Float64 = Intervals.X_MAX

struct Model
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

function draw_model(dims::Integer, n_components)
    return draw_model(Random.default_rng(), dims, n_components)
end

function draw_model(
    rng::AbstractRNG,
    dims::Integer,
    n_components;
    # Note that these are the same default values as the ones given for
    # `Intervals.draw_intervals`.
    spread_min=Intervals.spread_ideal_cubes(dims, n_components),
    spread_max=Inf,
    x_min=X_MIN,
    x_max=X_MAX,
    volume_min=Intervals.volume_min_factor(
        dims,
        n_components;
        x_min=x_min,
        x_max=x_max,
    ),
)
    conditions = draw_intervals(
        rng,
        dims,
        n_components;
        spread_min=spread_min,
        spread_max=spread_max,
        x_min=x_min,
        x_max=x_max,
        volume_min=volume_min,
    )
    local_models = [draw_constantmodel(rng) for _ in 1:n_components]

    return Model(conditions, local_models; x_min=x_min, x_max=x_max)
end

function draw_inputs(dims::Integer, n=1; x_min=X_MIN, x_max=X_MAX)
    return draw_inputs(Random.default_rng(), dims, n; x_min=x_min, x_max=x_max)
end

function draw_inputs(
    rng::AbstractRNG,
    dims::Integer,
    n=1;
    x_min=X_MIN,
    x_max=X_MAX,
)
    return rand(rng, Uniform(x_min, x_max), n, dims)
end

function match(model::Model, X::AbstractMatrix)
    return elemof(X, model.conditions)
end

function draw_data(model::Model, n)
    return draw_data(Random.default_rng(), model, n)
end

function draw_data(rng::AbstractRNG, model::Model, n)
    X = draw_inputs(
        rng,
        dimensions(model),
        n;
        x_min=model.x_min,
        x_max=model.x_max,
    )
    matching_matrix = match(model, X)
    y = output(rng, model.local_models, matching_matrix)

    return X, y
end

end
