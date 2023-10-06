module Tasks

using NPZ
using Random
using Serialization

using ..Intervals
using ..Models

export Task, dimensions, generate, save, load

const X_MIN::Float64 = Models.X_MIN
const X_MAX::Float64 = Models.X_MAX

struct Task
    seed::Integer
    model::Model
    X::AbstractMatrix
    y::AbstractVector
    X_test::AbstractMatrix
    y_test::AbstractVector
    match_X::AbstractMatrix
end

function Intervals.dimensions(task::Task)
    return Models.dimensions(task.model)
end

function generate(
    dims::Integer,
    n_components,
    n_train;
    n_test=10 * n_train,
    seed::Union{Nothing,Integer}=nothing,
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
    if seed == nothing
        seed = rand(UInt)
    end
    rng = Random.Xoshiro(seed)

    model = draw_model(
        rng,
        dims,
        n_components;
        spread_min=spread_min,
        spread_max=spread_max,
        x_min=x_min,
        x_max=x_max,
        volume_min=volume_min,
    )
    X, y = draw_data(rng, model, n_train)
    X_test, y_test = draw_data(rng, model, n_test)
    match_X = Models.match(model, X)

    return Task(seed, model, X, y, X_test, y_test, match_X)
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

end
