module Tasks

using AutoHashEquals
using Base.Filesystem
using LibGit2
using Mmap
using NPZ
using Random
using Serialization

using ..Intervals
using ..Models

export Task, dimensions, generate, generate_data, write_npz

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
    n_train::Integer
    n_test::Integer
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
        n_train,
        n_test,
        git_commit,
        git_dirty,
        hash_inputs,
        hash((hash_inputs, git_commit)),
    )
end

"""
Generate data for the given task.

Since data may not fit into RAM, memory-map each array created to a temporary
file under `dir_parent`.
"""
function generate_data(task::Task; dir_parent=tempdir())
    n_train = task.n_train
    n_test = task.n_test
    dx = dimensions(task.model)
    K = length(task.model.local_models)

    (path_X, io_X) = mktemp(dir_parent)
    X = mmap(io_X, Array{Float64,2}, (n_train, dx))
    (path_y, io_y) = mktemp(dir_parent)
    y = mmap(io_X, Array{Float64,2}, (n_train, 1))

    (path_X_test, io_X_test) = mktemp(dir_parent)
    X_test = mmap(io_X_test, Array{Float64,2}, (n_test, dx))
    (path_y_test, io_y_test) = mktemp(dir_parent)
    y_test = mmap(io_X, Array{Float64,2}, (n_test, 1))

    (path_match_X, io_match_X) = mktemp(dir_parent)
    match_X = mmap(io_X, Array{Float64,2}, (n_train, K))

    # Always add the same magic number to the task's random seed.
    magic = 1337
    rng = Random.Xoshiro(task.seed + magic)

    X, y = draw_data(rng, task.model, task.n_train)
    X_test, y_test = draw_data(rng, task.model, task.n_test)
    match_X = Models.match(task.model, X)

    return (X, y, X_test, y_test, match_X)
end

function write_npz(fname_prefix::String, task::Task)
    X, y, X_test, y_test, match_X = generate_data(task)

    return write_npz(fname_prefix, task, X, y, X_test, y_test, match_X)
end

function write_npz(
    fname_prefix::String,
    task::Task,
    X::AbstractArray{Float64},
    y::AbstractArray{Float64},
    X_test::AbstractArray{Float64},
    y_test::AbstractArray{Float64},
    match_X::AbstractArray{Bool},
)
    return npzwrite(
        "$(fname_prefix).data.npz",
        Dict(
            "X" => X,
            "y" => y,
            "X_test" => X_test,
            "y_test" => y_test,
            "git_dirty" => task.git_dirty,
            "hash" => task.hash,
        ),
    )
end

# function save(fname_prefix::String, task::Task)
#     serialize("$(fname_prefix).task.jls", task)
#     write_npz(fname_prefix, task)

#     return nothing
# end

# function load(fname_prefix::String)
#     deserialize("$(fname_prefix).task.jls")
#     # TODO Consider to load npz here as well (and memmap it)
#     return nothing
# end

end
