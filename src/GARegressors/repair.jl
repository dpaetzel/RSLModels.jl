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
    # TODO Consider to delete duplicate rules
    # TODO Consider to make this repair! and mutate g (i.e. deleteat!) directly
    g_ = deepcopy(g)

    # Remove rules that match less than `nmatch_min` training data points.
    idx_rm = []
    for idx in eachindex(g_)
        # TODO Cache matching matrix
        if count(elemof(X, g_[idx])) < nmatch_min
            push!(idx_rm, idx)
        end
    end
    deleteat!(g_, idx_rm)
    n_removed = length(idx_rm)
    if n_removed > 0
        @warn "Removed $n_removed conditions due to less than $nmatch_min " *
              "training data matches" operator = "repair"
    end

    report = (; n_removed=n_removed)
    return g_, report
end
