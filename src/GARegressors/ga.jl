const Genotype = AbstractVector{Intervals.Interval}

const XType::DataType = AbstractMatrix{Float64}
const YType::DataType = AbstractVector{Float64}

const draw_genotype = draw_intervals

unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

"""
Wrapper to store solutions together with the models they induce as well as their
fitness score.
"""
@auto_hash_equals struct EvaluatedGenotype
    genotype::Genotype
    phenotype::Models.Model
    fitness::Float64
end

function Base.length(g::EvaluatedGenotype)
    return Base.length(g.genotype)
end

"""
Express the given genotype, yielding a phenotype.

This is achieved by fitting a rule set model for based on the set of conditions
specified by the genotype.
"""
function express(
    genotype::Genotype,
    X::XType,
    y::YType,
    x_min,
    x_max,
    nmatch_min::Int,
)
    # TODO Consider to cache matching
    N, DX = size(X)

    conditions = deepcopy(genotype)

    dists_out = []
    idx_rm = []
    for idx in eachindex(conditions)
        y_matched = view(y, elemof(X, conditions[idx]))
        # TODO Expose this parameter as match_min or something like that
        # TODO Disable this test if `repair` is used
        # If too few training data points are matched, ignore the condition.
        if length(y_matched) < nmatch_min
            push!(idx_rm, idx)
        else
            dist_out = fit_mle(Normal, y_matched)
            push!(dists_out, dist_out)
        end
    end
    # Delete conditions that do not match enough training data points.
    deleteat!(conditions, idx_rm)

    # Inverse variance–based mixing (see drugowitsch2008b).
    vars = var.(dists_out)
    coefs_mix = 1 ./ vars
    # Fix NaN stemming from zero variance (which can occur if all training data
    # points that a rule matches have the same `y`).
    coefs_mix[.!ismissing.(vars) .&& iszero.(vars)] .= Inf

    lmodels = ConstantModel.(dists_out, coefs_mix, false)

    # Append the default rule.
    push!(conditions, maxgeneral(DX; x_min=x_min, x_max=x_max))
    push!(lmodels, ConstantModel(fit_mle(Normal, y), nextfloat(0.0), true))

    return Models.Model(conditions, lmodels; x_min=x_min, x_max=x_max)
end

"""
Evaluate the given solution genotype using the given fitness function.
"""
function evaluate(
    genotype::Genotype,
    X::XType,
    y::YType,
    x_min::Float64,
    x_max::Float64,
    ffunc::Function,
    nmatch_min::Int,
)
    phenotype = express(genotype, X, y, x_min, x_max, nmatch_min)
    fitness = ffunc(phenotype)
    return EvaluatedGenotype(genotype, phenotype, fitness)
end

function evaluate(
    solution::EvaluatedGenotype,
    X::XType,
    y::YType,
    x_min::Float64,
    x_max::Float64,
    ffunc::Function,
    nmatch_min::Int,
)
    phenotype = express(solution.genotype, X, y, x_min, x_max, nmatch_min)
    fitness = ffunc(phenotype)
    # Note that we don't return a report for `evaluate` yet. If you want to
    # implement this, consider the fact that `evaluate` typically gets
    # broadcasted over a vector of solutions/genotypes (and we should therefore
    # specialize for instead of broadcast over a vector so that reports are
    # properly concatenated instead of returning a vector of pairs).
    return EvaluatedGenotype(solution.genotype, phenotype, fitness)
end

struct GAResult
    best::EvaluatedGenotype
    pop_rest::AbstractVector{EvaluatedGenotype}
end

"""
Runs the GA based on the given config and training data.
"""
function runga end

function runga(X::XType, y::YType, config::GARegressor; verbosity::Int=0)
    # Note that this does not perform checks on the config but instead assumes
    # that it is valid.

    N, DX = size(X)

    rng = config.rng isa Int ? MersenneTwister(config.rng) : config.rng

    if config.fiteval == :mae
        ffunc = mkffunc(MAEFitness(X, y))
    elseif config.fiteval == :dissimilarity
        if config.dgmodel == nothing
            throw(
                ArgumentError(
                    "dgmodel != nothing required for fiteval == :dissimilarity",
                ),
            )
        end
        ffunc = mkffunc(DissimFitness(config.dgmodel, X))
    elseif config.fiteval == :likelihood
        ffunc = mkffunc(LikelihoodFitness(X, y))
    else
        throw(
            ArgumentError(
                "$(config.fiteval) is not a supported fitness scheme",
            ),
        )
    end

    # Cache initialization parameters.
    params_dist = if config.init == :inverse
        if verbosity > 0
            @info "Loading init distribution sample …"
        end

        # Note that if the range is very wide this may be overly expensive.
        # TODO Consider to compute `selectparam` on-demand if very many lengths
        lens = (config.init_length_min):(config.init_length_max)
        df = selectparams("kdata", lens...; verbosity=verbosity - 1)
        # TODO Make coverage configurable (but then we need to ensure that the
        # sample contains that coverage)
        # For now, we always take the highest-coverage parametrizations. Prepare
        # your sample accordingly.
        combine(
            groupby(df, [:DX, :K]),
            x -> sort(x, :rate_coverage_min)[end, :],
        )
    else
        nothing
    end

    # Build initializer.
    init1 = if config.init == :inverse
        mkinit1_inverse(config.x_min, config.x_max, params_dist)
    else
        mkinit1_custom(
            config.x_min,
            config.x_max,
            config.init_spread_min,
            config.init_spread_max,
            config.init_params_spread_a,
            config.init_params_spread_b,
            config.init_rate_coverage_min,
        )
    end

    # Initialize.
    #
    # Count number of evaluations.
    n_eval::Int = 0

    if verbosity > 0
        @info "Initializing population of size $(config.size_pop) …"
    end

    # Initialize pop, repair it, and then eval it.
    pop_uneval::Vector{Genotype}, report =
        init(rng, X, init1, config.size_pop; verbosity=verbosity - 1)
    pop_uneval[:], reports =
        unzip(repair.(Ref(rng), pop_uneval, Ref(X), Ref(config.nmatch_min)))
    pop::Vector{EvaluatedGenotype} =
        evaluate.(
            pop_uneval,
            Ref(X),
            Ref(y),
            config.x_min,
            config.x_max,
            ffunc,
            config.nmatch_min,
        )
    n_eval += length(pop)

    # Initialize elitist logging.
    idx_best::Int = fittest_idx(pop)
    best::EvaluatedGenotype = pop[idx_best]
    len_best::Int = length(best)

    # Initialize length-niching selection parameters.
    width_window::Int = 7
    lambda_window::Float64 = 0.0004
    # TODO Extract width_window and lambda_window to config

    # Bias factor for ryerkerk2020's biased window mechanism.
    bias_window::Float64 = 0

    for iter in 1:(config.n_iter)
        if verbosity > 0
            @info "Starting iteration $iter/$(config.n_iter) …"
            @info "Current best individual has length $len_best and " *
                  "$(config.fiteval) fitness $(best.fitness)."
        end

        # TODO Consider to speed this up by inplace mutating a fixed array
        offspring_uneval = []

        idx_rand = reshape(randperm(config.size_pop), 2, :)

        # Iterating over columns is faster b/c of Julia arrays using
        # column-major order.
        for (idx1, idx2) in eachcol(idx_rand)
            if rand(rng) <= config.recomb_rate
                # TODO Gotta copy in crossover if needed
                g1, g2, report = crossover(rng, pop[idx1], pop[idx2])
            else
                g1, g2 = (deepcopy(pop[idx1]), deepcopy(pop[idx2]))
            end

            g1, report = mutate(
                rng,
                g1,
                X,
                config.mutate_rate_mut,
                config.mutate_rate_std,
                config.mutate_p_add,
                config.mutate_p_rm,
                config.init_spread_min,
                config.init_spread_max,
                config.init_params_spread_a,
                config.init_params_spread_b,
                config.x_min,
                config.x_max,
            )
            g2, report = mutate(
                rng,
                g2,
                X,
                config.mutate_rate_mut,
                config.mutate_rate_std,
                config.mutate_p_add,
                config.mutate_p_rm,
                config.init_spread_min,
                config.init_spread_max,
                config.init_params_spread_a,
                config.init_params_spread_b,
                config.x_min,
                config.x_max,
            )

            g1, report = repair(rng, g1, X, config.nmatch_min)
            g2, report = repair(rng, g2, X, config.nmatch_min)

            push!(offspring_uneval, g1)
            push!(offspring_uneval, g2)
        end

        offspring =
            evaluate.(
                offspring_uneval,
                Ref(X),
                Ref(y),
                Ref(config.x_min),
                Ref(config.x_max),
                Ref(ffunc),
                Ref(config.nmatch_min),
            )
        n_eval += length(offspring)

        # (4) in ryerkerk2020.
        len_lbound =
            min(Int(ceil(len_best - width_window / 2 + bias_window)), len_best)
        len_ubound = max(
            Int(floor(len_best + width_window / 2 + bias_window)),
            len_best,
        )
        lengths = collect(len_lbound:len_ubound)

        selection, report =
            select(rng, vcat(pop, offspring), config.size_pop, lengths)
        pop[:] = selection

        idx_best = fittest_idx(pop)
        best = pop[idx_best]

        len_best_prev = len_best
        len_best = length(best)
        # (3) in ryerkerk2020.
        bias_window =
            len_best - len_best_prev +
            bias_window * exp(-lambda_window * sqrt(abs(bias_window)))
    end

    deleteat!(pop, idx_best)
    report = (; bias_window=bias_window, n_eval=n_eval)
    return GAResult(best, pop), report
end

function fittest_idx(pop::AbstractVector{EvaluatedGenotype})
    return argmax(idx -> getproperty(pop[idx], :fitness), eachindex(pop))
end
