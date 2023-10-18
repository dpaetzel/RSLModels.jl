module Tasks

using AutoHashEquals
using LibGit2
using NPZ
using Random
using Serialization

using ..Intervals
using ..Models

export Task, dimensions, generate, save, load

const X_MIN::Float64 = Models.X_MIN
const X_MAX::Float64 = Models.X_MAX

"""
- `git_commit`: Git commit hash of the current directory when `generate` was called.
- `hash_inputs`: Hash of all the inputs to `generate`.
- `hash`: Hash of `hash_inputs` and `git_commit`.
"""
@auto_hash_equals struct Task
    seed::Integer
    model::Model
    X::AbstractMatrix
    y::AbstractVector
    X_test::AbstractMatrix
    y_test::AbstractVector
    match_X::AbstractMatrix
    git_commit::AbstractString
    git_dirty::Bool
    hash_inputs::UInt
    hash::UInt
end

function Intervals.dimensions(task::Task)
    return Models.dimensions(task.model)
end

function generate(
    dims::Integer,
    nif::Integer,
    n_train::Integer;
    n_test=10 * n_train,
    seed::Union{Nothing,Integer}=nothing,
    # Note that these are the same default values as the ones given for
    # `Intervals.draw_intervals`.
    spread_min=Intervals.spread_ideal_cubes(dims, nif),
    spread_max=Inf,
    rate_coverage_min::Float64=0.8,
    remove_final_fully_overlapped::Bool=true,
    x_min=X_MIN,
    x_max=X_MAX,
)
    if seed == nothing
        seed = rand(UInt)
    end
    rng = Random.Xoshiro(seed)

    model = draw_model(
        rng,
        dims,
        nif;
        spread_min=spread_min,
        spread_max=spread_max,
        rate_coverage_min=rate_coverage_min,
        remove_final_fully_overlapped=remove_final_fully_overlapped,
        x_min=x_min,
        x_max=x_max,
    )
    X, y = draw_data(rng, model, n_train)
    X_test, y_test = draw_data(rng, model, n_test)
    match_X = Models.match(model, X)

    git_commit = LibGit2.head(".")
    git_dirty = LibGit2.isdirty(GitRepo("."))
    hash_inputs = hash((
        dims,
        nif,
        n_train,
        n_test,
        seed,
        spread_min,
        spread_max,
        rate_coverage_min,
        remove_final_fully_overlapped,
        x_min,
        x_max,
    ))

    return Task(
        seed,
        model,
        X,
        y,
        X_test,
        y_test,
        match_X,
        git_commit,
        git_dirty,
        hash_inputs,
        hash((hash_inputs, git_commit)),
    )
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
            "git_dirty" => [task.git_dirty],
            "hash" => [task.hash],
        ),
    )

    serialize("$(fname_prefix).task.jls", task)

    return nothing
end

function load(fname_prefix::String)
    return deserialize("$(fname_prefix).task.jls")
end

end
