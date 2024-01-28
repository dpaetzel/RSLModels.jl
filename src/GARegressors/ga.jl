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
    # Matching matrix from when phenotype (mixing coefficients etc.) was
    # computed.
    matches::Matrix{Bool}
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
    K = length(conditions)

    M = Matrix{Bool}(undef, N, K)
    dists_out = []
    idx_rm = []
    for idx in eachindex(conditions)
        M[:, idx] .= elemof(X, conditions[idx])
        y_matched = view(y, M[:, idx])
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
    # TODO Performance: Do not recompute this everytime, this is constant
    push!(lmodels, ConstantModel(fit_mle(Normal, y), nextfloat(0.0), true))

    return Models.Model(conditions, lmodels; x_min=x_min, x_max=x_max), M
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
    phenotype, M = express(genotype, X, y, x_min, x_max, nmatch_min)
    fitness = ffunc(phenotype)
    return EvaluatedGenotype(genotype, phenotype, fitness, M)
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
    phenotype, M = express(solution.genotype, X, y, x_min, x_max, nmatch_min)
    fitness = ffunc(phenotype)
    # Note that we don't return a report for `evaluate` yet. If you want to
    # implement this, consider the fact that `evaluate` typically gets
    # broadcasted over a vector of solutions/genotypes (and we should therefore
    # specialize for instead of broadcast over a vector so that reports are
    # properly concatenated instead of returning a vector of pairs).
    return EvaluatedGenotype(solution.genotype, phenotype, fitness, M)
end

struct GAResult
    best::EvaluatedGenotype
    pop_last::AbstractVector{EvaluatedGenotype}
end

function fitness(result::GAResult)
    return result.best.fitness
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

    if config.fiteval == :negmae
        ffunc = mkffunc(NegMAEFitness(X, y))
    elseif config.fiteval == :similarity
        if config.dgmodel == nothing
            throw(
                ArgumentError(
                    "dgmodel != nothing required for fiteval == :similarity",
                ),
            )
        end
        ffunc = mkffunc(SimFitness(config.dgmodel, X))
    elseif config.fiteval == :likelihood
        ffunc = mkffunc(LikelihoodFitness(X, y))
    elseif config.fiteval == :posterior
        ffunc = mkffunc(PosteriorFitness(X, y))
    elseif config.fiteval == :NegAIC
        ffunc = mkffunc(NegAICFitness(X, y))
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
        df = selectparams(
            config.init_sample_fname,
            lens...;
            verbosity=verbosity - 1,
            unsafe=config.init_unsafe,
        )
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

    recomb = if config.recomb == :spatial
        crossover_spatial
    elseif config.recomb == :cutsplice
        crossover_cutsplice
    else
        throw(
            ArgumentError(
                "$(config.recomb) is not a supported recombination operator",
            ),
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
    pop_uneval[:], reports = unzip(repair.(Ref(rng), pop_uneval, Ref(X)))
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
    best::EvaluatedGenotype = pop[fittest_idx(pop)]
    len_best::Int = length(best)

    # Bias factor for ryerkerk2020's biased window mechanism.
    bias_window::Float64 = 0

    # Initialize convergence logging.
    log_fitness_best = Vector{Float64}(undef, config.n_iter)
    log_fitness = Array{Float64}(undef, config.size_pop, config.n_iter)
    log_length = Array{Float64}(undef, config.size_pop, config.n_iter)
    log_select_bias = Vector{Float64}(undef, config.n_iter)
    log_select_len_lbound = Vector{Float64}(undef, config.n_iter)
    log_select_len_ubound = Vector{Float64}(undef, config.n_iter)

    iter::Int = 1
    for outer iter in 1:(config.n_iter)
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
            if rand(rng) < config.recomb_rate
                g1, g2, report = recomb(rng, pop[idx1], pop[idx2])
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

            g1, report = repair(rng, g1, X)
            g2, report = repair(rng, g2, X)

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

        pop_all = vcat(pop, offspring)

        # Update elitist.
        best_pop = pop_all[fittest_idx(pop_all)]
        if best_pop.fitness >= best.fitness && length(best_pop) <= length(best)
            best = best_pop
            len_best = length(best)
        end

        # We perform selection after updating the elitist because we want to use
        # the most recent `len_best`.
        # TODO Only compute this if lengthniching is actually used
        len_lbound, len_ubound = biasedwindow_bounds(
            len_best,
            config.select_width_window,
            bias_window,
            config.size_pop,
        )

        # TODO Refactor select mess
        selection, report = if config.select == :lengthniching
            if verbosity > 0
                @info "Selection window is " *
                      "[$len_lbound, ($len_best), $len_ubound]. " *
                      "Window bias is approximately " *
                      "$(round(bias_window; digits=2))."
            end

            select(
                rng,
                pop_all,
                config.size_pop,
                collect(len_lbound:len_ubound),
                len_best,
            )
        elseif config.select == :tournament
            select_trnmt(
                rng,
                pop_all,
                config.size_pop;
                size_trnmt=config.select_size_tournament,
            )
        else
            throw(
                ArgumentError(
                    "$(config.select) is not a supported selection operator",
                ),
            )
        end
        pop[:] = selection

        # Log convergence metrics.
        log_fitness_best[iter] = best.fitness
        log_fitness[:, iter] .= getproperty.(pop, :fitness)
        log_length[:, iter] .= length.(pop)
        log_select_bias[iter] = bias_window
        log_select_len_lbound[iter] = len_lbound
        log_select_len_ubound[iter] = len_ubound

        # TODO Only compute this if lengthniching is actually used
        # Update selection length niche.
        len_best_prev = len_best
        len_best = length(best)
        # (3) in ryerkerk2020.
        bias_window =
            len_best - len_best_prev +
            bias_window *
            exp(-config.select_lambda_window * sqrt(abs(bias_window)))

        if iter > config.n_iter_earlystop && all(
            log_fitness_best[(iter - config.n_iter_earlystop):(iter - 1)] .==
            log_fitness_best[iter],
        )
            break
        end
    end

    report = (;
        # Actual number of iterations used.
        n_iter=iter,
        bias_window=bias_window,
        n_eval=n_eval,
        # Report only the entries up to the current iteration (relevant if
        # stopping early).
        log_fitness_best=log_fitness_best[1:iter],
        log_fitness=log_fitness[:, 1:iter],
        log_length=log_length[:, 1:iter],
        log_select_bias=log_select_bias[1:iter],
        log_select_len_lbound=log_select_len_lbound[1:iter],
        log_select_len_ubound=log_select_len_ubound[1:iter],
    )
    return GAResult(best, pop), report
end

function fittest_idx(pop::AbstractVector{EvaluatedGenotype})
    return argmax(idx -> getproperty(pop[idx], :fitness), eachindex(pop))
end
