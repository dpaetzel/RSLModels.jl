using Distributions
using LinearAlgebra
using Random
using StatsBase
using NPZ
using Serialization

include("Intervals.jl")
include("LocalModels.jl")
using .Intervals
using .LocalModels

# We want to track global assignments as well when doing
# `includet(thisfile.jl)`.
__revise_mode__ = :evalassign

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
        dimensions = Intervals.dimensions(conditions[1])
        if length(conditions) != length(local_models)
            error(
                "number of conditions and number of local models are not equal",
            )
        elseif !all(Intervals.dimensions.(conditions) .== dimensions)
            error("conditions have to have the same dimension")
        end
        return new(conditions, local_models, x_min, x_max)
    end
end

function dimensions(model::Model)
    return Intervals.dimensions(model.conditions[1])
end

function draw_model(dimensions::Integer, n_components)
    return draw_model(Random.default_rng(), dimensions, n_components)
end

# TODO Expose spread_{min,max}, volume_min, …
function draw_model(
    rng::AbstractRNG,
    dimensions::Integer,
    n_components;
    x_min=X_MIN,
    x_max=X_MAX,
)
    conditions =
        draw_intervals(rng, dimensions, n_components; x_min=x_min, x_max=x_max)
    local_models = [draw_constantmodel(rng) for _ in 1:n_components]

    return Model(conditions, local_models; x_min=x_min, x_max=x_max)
end

function draw_inputs(dimensions::Integer, n=1; x_min=X_MIN, x_max=X_MAX)
    return draw_inputs(
        Random.default_rng(),
        dimensions,
        n;
        x_min=x_min,
        x_max=x_max,
    )
end

function draw_inputs(
    rng::AbstractRNG,
    dimensions::Integer,
    n=1;
    x_min=X_MIN,
    x_max=X_MAX,
)
    return rand(rng, Uniform(x_min, x_max), n, dimensions)
end

function match(interval::Interval, x)
    return interval.lbound <= x <= interval.ubound
end

# TODO Consider to check for dimensions matching here
function match(interval::Interval, X::AbstractMatrix)
    return all(interval.lbound' .<= X .<= interval.ubound'; dims=2)
end

function match(intervals::AbstractVector{Interval}, X::AbstractMatrix)
    matching_matrix = Matrix{Float64}(undef, size(X)[1], length(intervals))
    for i in 1:length(intervals)
        matching_matrix[:, i] = match(intervals[i], X)
    end
    return matching_matrix
end

function match(model::Model, X::AbstractMatrix)
    return match(model.conditions, X)
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
    y = output(rng, model.local_models; matching_matrix=matching_matrix)

    return X, y
end

struct Task
    seed::Integer
    model::Model
    X::AbstractMatrix
    y::AbstractVector
    X_test::AbstractMatrix
    y_test::AbstractVector
end

function generate(
    dimensions::Integer,
    n_components,
    n_train;
    n_test=10 * n_train,
    seed::Union{Nothing,Integer}=nothing,
    x_min=X_MIN,
    x_max=X_MAX,
)
    if seed == nothing
        seed = rand(UInt)
    end
    rng = Random.Xoshiro(seed)

    model = draw_model(rng, dimensions, n_components; x_min=x_min, x_max=x_max)
    X, y = draw_data(rng, model, n_train)
    X_test, y_test = draw_data(rng, model, n_test)

    return Task(seed, model, X, y, X_test, y_test)
end

# For now, we write the training and test data to an NPZ file as well (in
# addition to serializing the Julia way) b/c we want to read that data from a
# Python environment.
function save(fname_prefix::String, task::Task)
    npzwrite(
        "$(fname_prefix).data.npz",
        Dict(
            "X" => task.X,
            "y" => task.y,
            "X_test" => task.X_test,
            "y_test" => task.y_test,
        ),
    )

    serialize("$(fname_prefix).task.jls", task)

    return nothing
end

function load(fname_prefix::String)
    return deserialize("$(fname_prefix).task.jls")
end

task = generate(5, 10, 1000; seed=1337)

# TODO Extract plot_mapping and related stuff from Intervals to Score module
# TODO Include Score module here and add training data points to plots
# TODO Compute min/max variants of an interval given input data X
# TODO Compute min/max variants of many intervals given input data X
# TODO Compute min/max variants of many intervals given input data X

# TODO Consider to enforce that each interval is sampled k times
