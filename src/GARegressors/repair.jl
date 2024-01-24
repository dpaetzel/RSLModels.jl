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
    # This does nothing for now.
    report = (;)
    return g, report
end
