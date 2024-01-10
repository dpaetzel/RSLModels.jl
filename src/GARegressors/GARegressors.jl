module GARegressors

using AutoHashEquals
using DataFrames
using Distributions
using MLJModelInterface: MLJModelInterface
MMI = MLJModelInterface
using MLUtils
using MLJ: mae
using ProgressMeter
using Random
using StatsBase

using ..AbstractModels
using ..Intervals
using ..Intervals.Parameters
using ..LocalModels
using ..Models
using ..Scores

include("types.jl")
include("ga.jl")
include("select.jl")
include("mutate.jl")
include("init.jl")
include("repair.jl")
include("recombine.jl")
include("fitness.jl")

export GARegressor

function fixparamdomain!(m, symbol, pred, desc)
    if !pred(getfield(m, symbol))
        setfield!(m, symbol, params_default[symbol])
        return "Parameter `$(string(symbol))` expected to be " *
               "$desc, resetting to $(params_default[symbol])\n"
    else
        return ""
    end
end

function MMI.clean!(m::GARegressor)
    warning = ""

    warning *= fixparamdomain!(m, :n_iter, x -> x > 0, "positive")

    warning *= fixparamdomain!(
        m,
        :size_pop,
        x -> x > 0 && x % 2 == 0,
        "positive and divisible by 2",
    )

    warning *= fixparamdomain!(
        m,
        :fiteval,
        x -> x ∈ [:mae, :dissimilarity, :likelihood],
        "one of {:mae, :dissimilarity, :likelihood}",
    )

    if m.fiteval == :dissimilarity && !(m.dgmodel isa Models.Model)
        throw(
            DomainError(
                "In order to use dissimilarity-based fitness, you have to " *
                "provide a data-generating model",
            ),
        )
    end

    warning *= fixparamdomain!(m, :nmatch_min, x -> x > 0, "positive")

    warning *= fixparamdomain!(
        m,
        :init,
        x -> x ∈ [:inverse, :custom],
        "one of {:inverse, :custom}",
    )

    warning *= fixparamdomain!(
        m,
        :init_sample_fname,
        x -> isfile(x) || isdir(x),
        "a CSV file or a folder of CSV files",
    )

    warning *= fixparamdomain!(m, :init_length_min, x -> x > 0, "positive")

    warning *= fixparamdomain!(
        m,
        :init_length_max,
        x -> x > m.init_length_min,
        "in [$(m.init_length_min), Inf]",
    )

    spread_min_max = (m.x_max - m.x_min) / 2
    warning *= fixparamdomain!(
        m,
        :init_spread_min,
        x -> 0 <= x <= spread_min_max,
        "[0, $spread_min_max]",
    )

    warning *= fixparamdomain!(
        m,
        :init_spread_max,
        x -> x > m.init_spread_min,
        "in [$(m.init_spread_min), Inf)",
    )

    warning *=
        fixparamdomain!(m, :init_params_spread_a, x -> x > 0, "in (0, Inf)")

    warning *=
        fixparamdomain!(m, :init_params_spread_b, x -> x > 0, "in (0, Inf)")

    warning *= fixparamdomain!(
        m,
        :init_rate_coverage_min,
        x -> 0.0 <= x <= 1.0,
        "in [0.0, 1.0]",
    )

    warning *=
        fixparamdomain!(m, :recomb_rate, x -> 0.0 <= x <= 1.0, "in [0.0, 1.0]")

    warning *= fixparamdomain!(
        m,
        :mutate_p_add,
        x -> 0.0 <= x <= 1.0,
        "in [0.0, 1.0]",
    )

    warning *=
        fixparamdomain!(m, :mutate_p_rm, x -> 0.0 <= x <= 1.0, "in [0.0, 1.0]")

    warning *= fixparamdomain!(
        m,
        :mutate_rate_mut,
        x -> 0.0 <= x <= 1.0,
        "in [0.0, 1.0]",
    )

    warning *= fixparamdomain!(
        m,
        :mutate_rate_std,
        x -> 0.0 <= x <= 1.0,
        "in [0.0, 1.0]",
    )

    warning *= fixparamdomain!(
        m,
        :select_width_window,
        x -> 1 <= x <= m.size_pop && x % 2 == 1,
        "in [1, $(m.size_pop)] and odd-valued",
    )

    warning *= fixparamdomain!(
        m,
        :select_lambda_window,
        x -> 0.0 <= x,
        "non-negative",
    )

    return warning
end

function MMI.fit(m::GARegressor, verbosity, X, y)
    # Note that “It is not necessary for `fit` to provide type or dimension checks
    # on `X` or `y` or to call `clean!` on the model; MLJ will carry out such
    # checks.”

    fitresult, report = runga(MMI.matrix(X), y, m, verbosity)

    cache = nothing
    # TODO Add convergence behaviour (elitist/mean fitness progression) etc. into
    # report
    return fitresult, cache, report
end

function MMI.predict(m::GARegressor, fitresult, Xnew)
    # TODO Consider to allow building an ensemble here
    Xnew_mat = MMI.matrix(Xnew)
    return output_dist(fitresult.best.phenotype, Xnew_mat)
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
