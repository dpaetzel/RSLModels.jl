mutable struct GARegressor <: MMI.Probabilistic
    n_iter::Int
    size_pop::Int
    fiteval::Symbol
    dgmodel::Union{Nothing,Model}
    rng::Union{Int,AbstractRNG}
    x_min::Float64
    x_max::Float64
    nmatch_min::Int
    # Initialization parameters.
    init_spread_min::Float64
    init_spread_max::Float64
    init_params_spread_a::Float64
    init_params_spread_b::Float64
    init_rate_coverage_min::Float64
    # Mutation parameters
    mutate_p_add::Float64
    mutate_p_rm::Float64
    mutate_rate_mut::Float64
    mutate_rate_std::Float64
    # Recombination parameters.
    recomb_rate::Float64
    # Selection parameters.
    select_width_window::Int
    select_lambda_window::Float64
end

params_default = Dict(
    :n_iter => 100,
    :size_pop => 32,
    :fiteval => :mae,
    :dgmodel => nothing,
    :rng => Random.default_rng(),
    :x_min => Intervals.X_MIN,
    :x_max => Intervals.X_MAX,
    :nmatch_min => 2,
    # Initialization parameters.
    :init_spread_min => 0.0,
    :init_spread_max => Inf,
    :init_params_spread_a => 1.0,
    :init_params_spread_b => 1.0,
    :init_rate_coverage_min => 0.8,
    # Mutation parameters
    :mutate_p_add => 0.05,
    :mutate_p_rm => 0.05,
    :mutate_rate_mut => 1.0,
    :mutate_rate_std => 0.05,
    # Recombination parameters
    :recomb_rate => 0.9,
    # Selection parameters.
    :select_width_window => 7,
    :select_lambda_window => 0.004,
)

function GARegressor(;
    n_iter=params_default[:n_iter],
    size_pop=params_default[:size_pop],
    fiteval=params_default[:fiteval],
    dgmodel=params_default[:dgmodel],
    rng=params_default[:rng],
    x_min=params_default[:x_min],
    x_max=params_default[:x_max],
    nmatch_min=params_default[:nmatch_min],
    # Initialization parameters.
    init_spread_min=params_default[:init_spread_min],
    init_spread_max=params_default[:init_spread_max],
    init_params_spread_a=params_default[:init_params_spread_a],
    init_params_spread_b=params_default[:init_params_spread_b],
    init_rate_coverage_min=params_default[:init_rate_coverage_min],
    # Mutation parameters
    mutate_p_add=params_default[:mutate_p_add],
    mutate_p_rm=params_default[:mutate_p_rm],
    mutate_rate_mut=params_default[:mutate_rate_mut],
    mutate_rate_std=params_default[:mutate_rate_std],
    # Recombination parameters
    recomb_rate=params_default[:recomb_rate],
    # Selection parameters.
    select_width_window=params_default[:select_width_window],
    select_lambda_window=params_default[:select_lambda_window],
)
    model = GARegressor(
        n_iter,
        size_pop,
        fiteval,
        dgmodel,
        rng,
        x_min,
        x_max,
        nmatch_min,
        # Initialization parameters.
        init_spread_min,
        init_spread_max,
        init_params_spread_a,
        init_params_spread_b,
        init_rate_coverage_min,
        # Mutation parameters
        mutate_p_add,
        mutate_p_rm,
        mutate_rate_mut,
        mutate_rate_std,
        # Recombination parameters
        recomb_rate,
        # Selection parameters.
        select_width_window,
        select_lambda_window,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

"""
    GARegressor(; <keyword arguments>)

# Arguments

- `nmatch_min::Int=$(params_default[:nmatch_min])`: Minimum number of training
  data points to be matched by any rule so that the rule is considered during
  training/prediction. The repair operator applied on all individuals deletes
  rules with less than `nmatch_min` matched training data points.
- `mutate_rate_mut::Float64=$(params_default[:mutate_rate_mut])`: Rate of condition
  parameter mutation in units of `1.0 / (2 * DX * length(genotype))`.
  `mutate_rate_mut==1.0` corresponds to on average mutating as many condition
  parameters as there are within one condition (i.e. within one rule).
- `mutate_rate_std::Float64=$(params_default[:mutate_rate_std])`: Standard
  deviation used in Gaussian condition parameter mutation in units of `x_max -
  x_min`. `mutate_rate_std==0.5` corresponds to a standard deviation of half the
  configured input space.
"""
GARegressor
