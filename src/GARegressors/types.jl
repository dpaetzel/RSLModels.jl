mutable struct GARegressor <: MMI.Probabilistic
    n_iter::Int
    size_pop::Int
    fiteval::Symbol
    dgmodel::Union{Nothing,Model}
    rng::Union{Int,AbstractRNG}
    x_min::Float64
    x_max::Float64
    nmatch_min::Int
    n_iter_earlystop::Int
    # Initialization parameters.
    init::Symbol
    init_sample_fname::String
    init_unsafe::Bool
    init_length_min::Int
    init_length_max::Int
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
    recomb::Symbol
    recomb_rate::Float64
    # Selection parameters.
    select::Symbol
    select_width_window::Int
    select_lambda_window::Float64
    select_size_tournament::Int
end

params_default = Dict(
    :n_iter => 100,
    :size_pop => 32,
    :fiteval => :NegAIC,
    :dgmodel => nothing,
    # TODO Should I use GLOBAL_RNG here?
    :rng => Random.default_rng(),
    :x_min => Intervals.X_MIN,
    :x_max => Intervals.X_MAX,
    :nmatch_min => 2,
    :n_iter_earlystop => 50,
    # Initialization parameters.
    :init => :inverse,
    :init_sample_fname => "kdata",
    :init_unsafe => false,
    :init_length_min => 1,
    :init_length_max => 30,
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
    :recomb => :spatial,
    :recomb_rate => 0.9,
    # Selection parameters.
    :select => :lengthniching,
    :select_width_window => 7,
    :select_lambda_window => 0.004,
    :select_size_tournament => 4,
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
    n_iter_earlystop=params_default[:n_iter_earlystop],
    # Initialization parameters.
    init=params_default[:init],
    init_sample_fname=params_default[:init_sample_fname],
    init_unsafe=params_default[:init_unsafe],
    init_length_min=params_default[:init_length_min],
    init_length_max=params_default[:init_length_max],
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
    recomb=params_default[:recomb],
    recomb_rate=params_default[:recomb_rate],
    # Selection parameters.
    select=params_default[:select],
    select_width_window=params_default[:select_width_window],
    select_lambda_window=params_default[:select_lambda_window],
    select_size_tournament=params_default[:select_size_tournament],
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
        n_iter_earlystop,
        # Initialization parameters.
        init,
        init_sample_fname,
        init_unsafe,
        init_length_min,
        init_length_max,
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
        recomb,
        recomb_rate,
        # Selection parameters.
        select,
        select_width_window,
        select_lambda_window,
        select_size_tournament,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

"""
    GARegressor(; <keyword arguments>)

# Arguments

- `n_iter_earlystop::Int=$(params_default[:n_iter_earlystop])`: Stop the GA
  after elitist fitness did not change for this many generations.
- `init::Symbol=$(params_default[:init])`: Which initialization method to use.
  `:inverse` means based on heuristically maximizing the density of the
  initialization parameters and requires a sample of a certain joint
  distribution (see `RSLModels.Intervals.Parameters.selectparams`). This means
  that `init_length*` and `init_length*` are used whereas `init_spread*`,
  `init_params*` and `init_rate*` are ignored during initial population
  initialization (they are set internally based on the sample). `:custom` means
  that the parameters are used as given by `init_spread*`, `init_params*` and
  `init_rate*` whereas `init_length*` are ignored. You should probably use
  `:inverse` unless you know what you're doing.
- `init_sample_fname::String=$(params_default[:init_sample_fname])`: If
  `init=:inverse` then this specifies the CSV file (or the folder of CSV files,
  see `RSLModels.Intervals.Parameters.selectparams`) that contains the joint
  distribution sample to use. See `init`.
- `init_unsafe::String=$(params_default[:init_unsafe])`: If `true`, the CSV
  file(s) of `init_sample_fname` are cashed based on the file name only. If
  `false`, they are cached based on their contents (slower). See
  the `Intervals.Parameters.selectparams` implementation for details.
- `init_length_min::Int=$(params_default[:init_length_min])`: Shortest solution
  length to use during random initialization. Note that due to the probabilistic
  nature of initialization and the way that the remaining initialization
  parameters are derived from this value, you cannot be certain that solutions
  of this length are actually generated during initialization.
- `init_length_max::Int=$(params_default[:init_length_max])`: Shortest solution
  length to use during random initialization. See `init_length_min`.
- `init_spread_min::Float64=$(params_default[:init_spread_min])`: If
  `init==:custom`, the initial population is filled with solutions drawn from a
  distribution of which this is one of the parameters (see
  `RSLModels.Intervals.draw_intervals`).

  In addition to that, this setting is also used whenever mutation adds a rule
  to a solution (even if `init == :inverse`!). Same for `init_spread_max`,
  `init_params_spread_a` and `init_params_spread_b`.
- `init_spread_max::Float64=$(params_default[:init_spread_max])`: See
  `init_spread_min`.
- `init_params_spread_a::Float64=$(params_default[:init_params_spread_a])`: See
  `init_spread_min`.
- `init_params_spread_b::Float64=$(params_default[:init_params_spread_b])`: See
  `init_spread_min`.
- `init_rate_coverage_min::Float64=$(params_default[:init_rate_coverage_min])`:
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
- `select::Int=params_default[:select]`: Which selection operator to use. One of
  `:lengthniching` (recommended) or `:tournament`.
- `select_width_window::Int=params_default[:select_width_window]`: Only relevant
  for length-niching selection.
- `select_lambda_window::Float64=params_default[:select_lambda_window]`: Only
  relevant for length-niching selection.
- `select_size_tournament::Int=params_default[:select_size_tournament]`: Only
  relevant for tournament selection.
- `recomb::Symbol=params_default[:recomb]`: Which crossover operator to use. One
  of `:spatial` and `:cutsplice`.
"""
GARegressor
