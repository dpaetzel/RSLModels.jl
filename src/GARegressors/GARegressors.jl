module GARegressors

using AutoHashEquals
using Distributions
using MLJModelInterface: MLJModelInterface
MMI = MLJModelInterface
using Random
using MLJ

using ..AbstractModels
using ..Intervals
using ..LocalModels
using ..Models
using ..Scores

include("types.jl")
include("ga.jl")
include("select.jl")
include("mutate.jl")

export GARegressor

function MMI.clean!(m::GARegressor)
    warning = ""

    # Has to be positive and divisible by 2 (due to pairwise crossover).
    if !(m.size_pop > 0 && m.size_pop % 2 == 0)
        warning *=
            "Parameter `size_pop` expected to be positive and " *
            "divisible by 2, resetting to $(params_default[:size_pop])"
        m.size_pop = params_default[:size_pop]
    end

    if !(m.n_iter > 0)
        warning *=
            "Parameter `n_iter` expected to be positive, " *
            "resetting to $(params_default[:n_iter])"
        m.n_iter = params_default[:n_iter]
    end

    if !(0 <= m.rate_crossover <= 1)
        warning *=
            "Parameter `rate_crossover` expected to be in [0, 1], " *
            "resetting to $(params_default[:rate_crossover])"
        m.rate_crossover = params_default[:rate_crossover]
    end

    if !(m.fiteval ∈ [:mae, :dissimilarity])
        warning *=
            "Parameter `fiteval` expected to be one of " *
            "{:mae, :dissimilarity}, " *
            "resetting to $(params_default[:fiteval])"
        m.fiteval = params_default[:fiteval]
    end

    if m.fiteval == :dissimilarity && !(m.dgmodel isa Models.Model)
        throw(
            DomainError(
                "In order to use dissimilarity-based fitness, you have to " *
                "provide a data-generating model",
            ),
        )
    end

    spread_min_max = (m.x_max - m.x_min) / 2
    if !(0 <= m.spread_min <= spread_min_max)
        warning *=
            "Parameter `spread_min` expected to be in [0, $spread_min_max], " *
            "resetting to $(params_default[:spread_min])"
        m.spread_min = params_default[:spread_min]
    end

    if !(0 <= m.spread_max)
        warning *=
            "Parameter `spread_max` expected to be in [0, Inf), " *
            "resetting to $(params_default[:spread_max])"
        m.spread_max = params_default[:spread_max]
    end

    if !(0 < m.params_spread_a)
        warning *=
            "Parameter `params_spread_a` expected to be in (0, Inf), " *
            "resetting to $(params_default[:params_spread_a])"
        m.params_spread_a = params_default[:params_spread_a]
    end

    if !(0 < m.params_spread_b)
        warning *=
            "Parameter `params_spread_b` expected to be in (0, Inf), " *
            "resetting to $(params_default[:params_spread_b])"
        m.params_spread_b = params_default[:params_spread_b]
    end

    return warning
end

function MMI.fit(m::GARegressor, verbosity, X, y)
    # Note that “It is not necessary for `fit` to provide type or dimension checks
    # on `X` or `y` or to call `clean!` on the model; MLJ will carry out such
    # checks.”

    fitresult = runga(m, X, y)

    cache = nothing
    # TODO Add convergence behaviour (elitist/mean fitness progression) etc. into
    # report
    report = (;)
    return fitresult, cache, report
end

function MMI.predict(m::GARegressor, fitresult, Xnew)
    # TODO Consider to allow building an ensemble here
    # TODO Refactor output_* to an output_dist function
    means = output_mean(fitresult.best.phenotype, Xnew)
    vars = output_variance(fitresult.best.phenotype, Xnew)

    return Normal.(means, sqrt.(vars))
end

MMI.input_scitype(::Type{<:GARegressor}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{<:GARegressor}) = AbstractVector{<:MMI.Continuous}

# function MMI.predict_joint(m::GARegressor, fitresult, Xnew)
#     error("Fully think this through before using it, are outputs really iid?")
#     error("Esp. check whether matching introduces some weird stuff")
#     means = output_mean(fitresult.best.model, Xnew)
#     vars = output_variance(fitresult.best.model, Xnew)

#     # Since observations are assumed to be iid, outputs are independent as well.
#     return MultivariateNormal(means, diagm(vars))
# end

# Optional, to return user-friendly form of fitted parameters:
# TODO MMI.fitted_params(model::SomeSupervisedModel, fitresult) = fitresult

# Optional, to specify default hyperparameter ranges (for use in tuning):
# TODO MMI.hyperparameter_ranges(T::Type) = Tuple(fill(nothing, length(fieldnames(T))))

# Optional, to avoid redundant calculations when re-fitting machines associated with a model:

# MMI.update(model::SomeSupervisedModel, verbosity, old_fitresult, old_cache, X, y) =
#    MMI.fit(model, verbosity, X, y)

# Required, if the model is to be registered (findable by general users):

# MMI.load_path(::Type{<:SomeSupervisedModel})    = ""
# MMI.package_name(::Type{<:SomeSupervisedModel}) = "Unknown"
# MMI.package_uuid(::Type{<:SomeSupervisedModel}) = "Unknown"

# Optional but recommended:

# MMI.package_url(::Type{<:SomeSupervisedModel})  = "unknown"
# MMI.is_pure_julia(::Type{<:SomeSupervisedModel}) = false
# MMI.package_license(::Type{<:SomeSupervisedModel}) = "unknown"

end
