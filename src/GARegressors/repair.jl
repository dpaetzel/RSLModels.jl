"""
Ensure that each rule matches at least a certain number of training data points.
"""
function repair end

function repair(rng::AbstractRNG, g::EvaluatedGenotype, X::XType)
    return repair(rng, g.genotype, X)
end

function repair(rng::AbstractRNG, g::Genotype, X::XType)
    # TODO Consider to delete duplicate rules
    # This does nothing for now.
    report = (;)
    return g, report
end
