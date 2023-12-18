# TODO Change name
mutable struct GARegressor <: MMI.Probabilistic
    size_pop::Int
    n_iter::Int
    rate_crossover::Float64
    fiteval::Symbol
    spread_min::Float64
    spread_max::Float64
    params_spread_a::Float64
    params_spread_b::Float64
    x_min::Float64
    x_max::Float64
    rng::Union{Int,AbstractRNG}
end

# TODO Undo struct rename
# GARegressor = GARegressor

params_default = Dict(
    :size_pop => 32,
    :n_iter => 100,
    :rate_crossover => 0.9,
    :fiteval => :mae,
    :dgmodel => nothing,
    :spread_min => 0.0,
    :spread_max => Inf,
    :params_spread_a => 1.0,
    :params_spread_b => 1.0,
    :x_min => Intervals.X_MIN,
    :x_max => Intervals.X_MAX,
    :rng => Random.default_rng(),
)

function GARegressor(;
    size_pop=params_default[:size_pop],
    n_iter=params_default[:n_iter],
    rate_crossover=params_default[:rate_crossover],
    fiteval=params_default[:fiteval],
    dgmodel=params_default[:dgmodel],
    spread_min=params_default[:spread_min],
    spread_max=params_default[:spread_max],
    params_spread_a=params_default[:params_spread_a],
    params_spread_b=params_default[:params_spread_b],
    x_min=params_default[:x_min],
    x_max=params_default[:x_max],
    rng=params_default[:rng],
)
    model = GARegressor(
        size_pop,
        n_iter,
        rate_crossover,
        fiteval,
        spread_min,
        spread_max,
        params_spread_a,
        params_spread_b,
        x_min,
        x_max,
        rng,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end
