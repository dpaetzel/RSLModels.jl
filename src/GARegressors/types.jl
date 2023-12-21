mutable struct GARegressor <: MMI.Probabilistic
    n_iter::Int
    size_pop::Int
    fiteval::Symbol
    dgmodel::Union{Nothing,Model}
    rng::Union{Int,AbstractRNG}
    x_min::Float64
    x_max::Float64
    # Initialization parameters.
    init_spread_min::Float64
    init_spread_max::Float64
    init_params_spread_a::Float64
    init_params_spread_b::Float64
    # Recombination parmeters.
    recomb_rate::Float64
end

params_default = Dict(
    :n_iter => 100,
    :size_pop => 32,
    :fiteval => :mae,
    :dgmodel => nothing,
    :rng => Random.default_rng(),
    :x_min => Intervals.X_MIN,
    :x_max => Intervals.X_MAX,
    # Initialization parameters.
    :init_spread_min => 0.0,
    :init_spread_max => Inf,
    :init_params_spread_a => 1.0,
    :init_params_spread_b => 1.0,
    # Recombination parameters
    :recomb_rate => 0.9,
)

function GARegressor(;
    n_iter=params_default[:n_iter],
    size_pop=params_default[:size_pop],
    fiteval=params_default[:fiteval],
    dgmodel=params_default[:dgmodel],
    rng=params_default[:rng],
    x_min=params_default[:x_min],
    x_max=params_default[:x_max],
    # Initialization parameters.
    init_spread_min=params_default[:init_spread_min],
    init_spread_max=params_default[:init_spread_max],
    init_params_spread_a=params_default[:init_params_spread_a],
    init_params_spread_b=params_default[:init_params_spread_b],
    # Recombination parameters
    recomb_rate=params_default[:recomb_rate],
)
    model = GARegressor(
        n_iter,
        size_pop,
        fiteval,
        dgmodel,
        rng,
        x_min,
        x_max,
        # Initialization parameters.
        init_spread_min,
        init_spread_max,
        init_params_spread_a,
        init_params_spread_b,
        # Recombination parameters
        recomb_rate,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end
