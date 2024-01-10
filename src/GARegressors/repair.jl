"""
Ensure that each rule matches at least a certain number of training data points.
"""
function repair end

function repair(
    rng::AbstractRNG,
    g::EvaluatedGenotype,
    X::XType,
    nmatch_min::Int,
)
    return repair(rng, g.genotype, X, nmatch_min)
end

function repair(rng::AbstractRNG, g::Genotype, X::XType, nmatch_min::Int)
    # TODO Consider to make this repair! and mutate g (i.e. deleteat!) directly
    g_ = deepcopy(g)

    k = nmatch_min

    idx_rm = []
    for idx in eachindex(g_)
        if count(elemof(X, g_[idx])) < k
            push!(idx_rm, idx)
        end
    end
    deleteat!(g_, idx_rm)
    n_removed = length(idx_rm)
    if n_removed > 0
        @debug "Removed $n_removed conditions due to less than $k training " *
               "data matches" operator = "repair"
    end

    report = (; n_removed=n_removed)
    return g_, report
end
