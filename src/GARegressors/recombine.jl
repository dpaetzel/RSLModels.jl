function crossover end

function crossover(
    rng::AbstractRNG,
    g1::EvaluatedGenotype,
    g2::EvaluatedGenotype,
)
    return crossover(rng, g1.genotype, g2.genotype)
end

function crossover(rng::AbstractRNG, g1::Genotype, g2::Genotype)
    @warn "`crossover` not implemented yet" maxlog = 5
    report = (;)
    return deepcopy(g1), deepcopy(g2), report
end
