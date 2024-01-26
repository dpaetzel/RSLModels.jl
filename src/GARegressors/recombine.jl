function crossover_spatial end

function crossover_spatial(
    rng::AbstractRNG,
    g1::EvaluatedGenotype,
    g2::EvaluatedGenotype,
)
    return crossover_spatial(rng, g1.genotype, g2.genotype)
end

"""
Spatial crossover operator. Draw a random plane located in the input space, then
split each individual into two subsets of intervals based on that plane (based
on which side of the plane the intervals' centers lie on) and perform crossover
by recombining subsets from each side of the plane.
"""
function crossover_spatial(rng::AbstractRNG, g1::Genotype, g2::Genotype)
    g1_ = deepcopy(g1)
    g2_ = deepcopy(g2)

    # Generate a random hyperplane.
    #
    # Draw the position vector from the metavariables' locations.
    metavars = vcat(g1_, g2_)
    idx_random = rand(rng, 1:length(metavars))
    pos_plane = center(metavars[idx_random])

    # Draw a random inclination for the hyperplane (i.e. a random vector which
    # will then be seen as being normal to the hyperplane).
    v_plane = rand(rng, Float64, size(pos_plane))
    # Make sure that the length of the vector isn't 0.
    while norm(v_plane) == 0
        v_plane = rand(rng, Float64, size(pos_plane))
    end

    # Compute on which side of the hyperplane each of the metavariables'
    # locations lie.
    sides1 = [dot(v_plane, center(metavar) - pos_plane) > 0 for metavar in g1_]
    sides2 = [dot(v_plane, center(metavar) - pos_plane) > 0 for metavar in g2_]

    # Split the individuals.
    offspring1 = vcat(g1_[sides1], g2_[.!sides2])
    offspring2 = vcat(g1_[.!sides1], g2_[sides2])

    # Fix the case where all metavariables have been put into one of the two
    # offsprings.
    if isempty(offspring1)
        idx_random = rand(rng, 1:length(offspring2))
        push!(offspring1, offspring2[idx_random])
        deleteat!(offspring2, idx_random)
    end

    if isempty(offspring2)
        idx_random = rand(rng, 1:length(offspring1))
        push!(offspring2, offspring1[idx_random])
        deleteat!(offspring1, idx_random)
    end

    report = (;)
    return offspring1, offspring2, report
end
