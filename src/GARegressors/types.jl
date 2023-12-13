# TODO Change name
mutable struct GARegressor <: MMI.Probabilistic
    size_pop::Int
    n_iter::Int
    rate_crossover::Float64
    fiteval::Symbol
    spread_min::Float64
    x_min::Float64
    x_max::Float64
    rng::Union{Int,AbstractRNG}
end

params_default = Dict(
    :size_pop => 32,
    :n_iter => 100,
    :rate_crossover => 0.9,
    :fiteval => :mae,
    :dgmodel => nothing,
    :spread_min => 0.0,
    :x_min => Intervals.X_MIN,
    :x_max => Intervals.X_MAX,
    :rng => Random._GLOBAL_RNG,
)

function GARegressor(;
    size_pop=params_default[:size_pop],
    n_iter=params_default[:n_iter],
    rate_crossover=params_default[:rate_crossover],
    fiteval=params_default[:fiteval],
    dgmodel=params_default[:dgmodel],
    spread_min=params_default[:spread_min],
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
        x_min,
        x_max,
        rng,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end
