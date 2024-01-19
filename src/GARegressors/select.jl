"""
Length niching selection operator proposed by ryerkerk2020.
"""
function select(
    rng::AbstractRNG,
    pop::AbstractVector{EvaluatedGenotype},
    n_select::Int,
    lengths::AbstractVector{Int},
)
    # TODO Maybe cache this to reduce one `O(size_pop)`
    idx_best = fittest_idx(pop)
    len_best = length(pop[idx_best])

    n_niches = length(lengths)
    @debug "select: Using $n_niches niches for population of size $(length(pop))"

    # If `size_pop` is not divisible by the number of niches, the `r` closest
    # lengths to the best solution's length each get one more space.
    r = n_select % n_niches
    lens_closest = closest(len_best - 0.1, lengths, r)

    selection = []
    size_niche = Vector{Int}(undef, n_niches)
    for (i, len) in enumerate(lengths)
        size_niche[i] = floor(Int, n_select / n_niches)
        if len in lens_closest
            size_niche[i] += 1
        end
        pop_i = [g for g in pop if length(g) == size_niche[i]]

        if isempty(pop_i)
            @debug "select: Empty niche at length $len"
            selection_i_ = []
        else
            @debug "select: Performing tournament selection within niche at " *
                   "length $len"
            selection_i_, report =
                select_trnmt_niche(rng, pop_i, size_niche[i])
        end

        append!(selection, selection_i_)
    end

    @assert sum(size_niche) == n_select "$size_niche, $n_select"

    if length(selection) < n_select
        n_filler = n_select - length(selection)
        # Since `EvaluatedGenotype` uses `auto_hash_equals`, we can safely
        # compare them.
        pop_unsel = [g for g in pop if !(g in selection)]
        selection_filler, report = select_trnmt_niche(rng, pop_unsel, n_filler)
        append!(selection, selection_filler)
    end

    @assert length(selection) == n_select "select(…, pop of length " *
                                          "$(length(pop)), $n_select, " *
                                          "$lengths) yields selection of " *
                                          "length $(length(selection))"

    report = (;)
    return selection, report
end

"""
Get the `k` values from `arr` that are closest to `x`.

Ties in distance to `x` are broken towards the smaller value.
"""
function closest(x, arr, k)
    # Sorting the array in ascending order results in ties being broken towards
    # smaller values.
    sorted_arr = sort(arr)
    perm_sort = sortperm(abs.(sorted_arr .- x))
    return sorted_arr[perm_sort[1:k]]
end

"""
The length niches given the current best individual's length, the current bias
parameter value, as well as the parameters for the biased window function.

# Arguments
- `len_best::int`:
   Length of the best-so-far individual.
- `bias::float`:
   Current value of the biased window bias factor.
- `w::int`: Window width > 0 to be used for the biased window function. The
   default of 7 seems to be sensible when looking at the original paper
   proposing this scheme (“the tested window widths appear to have little
   effect on …  performance. Wider windows will result in a greater level of
   diversity …”).
- `w_max::int`: Maximum number of niches to allow (if exceeded, remove niches
   furthest away from `len_best` until satisfied).
"""
function lens_niches(len_best::Int, bias::Float64, w::Int, w_max::Int)
    len_lb = min(ceil(len_best - w / 2 + bias), len_best)
    # Ensure that the niche window does not start below 1.
    len_lb = max(len_lb, 1)
    len_ub = max(floor(len_best + w / 2 + bias), len_best)
    lens = len_lb:len_ub
    # Return the `w_max` closest values to `len_best`.
    return closest(len_best, lens, w_max)
end

"""
Creates a tournament plan for a subpopulation and niche of the given size.

A tournament plan is a matrix with `size_tournament` columns and  `size_niche`
rows. Each column corresponds to the indexes of individuals participating in one
tournament. Further,

- each individual takes part in at least one tournament
- each individual takes part in at most two tournaments
- each tournament does not have any individual participate twice.

Warning: *This assumes but does not check whether `length(subpop) <=
n_select`!*
"""
function trnmt_plan(rng::AbstractRNG, subpop::AbstractVector, n_select::Int)
    indices = collect(eachindex(subpop))
    shuffle!(rng, indices)

    # Determine tournament size such that each individual is able to take part
    # in at least one. Note that Ryerkerk et al.'s R code calls this `K`.
    size_tournament = Int(ceil(length(subpop) / n_select))

    # Preliminary tournament plan, possibly with open spots if `size_tournament
    # * n_select != length(subpop)`. This is (as are most other arrays in this
    # function) an array of *indices* into `subpop`.
    tournaments = reshape(
        rpad_constant(
            indices,
            # Tournament plan length.
            size_tournament * n_select,
            -1,
        ),
        # Note that we have columns correspond to tournaments due to Julia's
        # column-major-ness.
        size_tournament,
        n_select,
    )

    # Track indices of the individuals that participated only once so far.
    #
    # Note that this is just an alias (but `indices` is not used for anything
    # else and we can therefore refrain from copying).
    once = indices

    # Next, fill open spots such that
    # - no individual takes part more than once in each tournament and
    # - no individual takes part in more than two tournaments.
    #
    # Iterate over all tournaments.
    for idx in eachindex(eachcol(tournaments))
        # Indices (wrt `subpop`) of the participants in the current tournament
        # so far.
        participants = filter(x -> x != -1, tournaments[:, idx])

        # Indices (wrt `subpop`) of the open spots in the current tournament.
        spots_open = findall(x -> x == -1, tournaments[:, idx])

        # Indices of as many randomly chosen individuals that only participate
        # in a single tournament so far as there are open spots in the current
        # tournament.
        #
        # We have to sort for `deleteat!` to work.
        additional = sort(
            sample(
                rng,
                eachindex(setdiff(once, participants)),
                length(spots_open);
                replace=false,
            ),
        )
        tournaments[spots_open, idx] .= once[additional]

        # The just-chosen individuals will now participate more than once and
        # their indices are thus deleted from `once`.
        deleteat!(once, additional)
    end

    return tournaments
end

"""
Tournament selection operator used by ryerkerk2020 (see [their R
code](https://github.com/ryerkerk/metameric/blob/1d85dd51474df5c54cfe075189daeec30c00331d/selection/LocalSelection_Tournament.m)).

Note that tournament size is automatically derived from the target niche size
and the number of individuals such that tournament size is minimal while each
individual takes part in at least one tournament.
"""
function select_trnmt_niche(
    rng::AbstractRNG,
    subpop::AbstractVector,
    n_select::Int;
    fitness::Function=ind -> getproperty(ind, :fitness),
)
    # If there are less individuals than we want to select (or as many as), just
    # return the individuals we have.
    if length(subpop) <= n_select
        report = (;)
        return subpop, report
    else
        tournaments = trnmt_plan(rng, subpop, n_select)

        # Perform tournaments.
        winners = [
            # `argmax(f, itr)` returns the value `v` from `itr` for which `f(v)`
            # is maximal. Since values in `tournament` are indexes, we use
            # `getfitness` to retrieve fitness values.
            argmax(idx -> fitness(subpop[idx]), tournament) for
            tournament in eachcol(tournaments)
        ]

        report = (;)
        return subpop[winners], report
    end
end

# trnmt_plan(Random.default_rng(), collect(1:11), 11; fitness=identity)
select_trnmt_niche(Random.default_rng(), collect(1:11), 11; fitness=identity)

"""
Tournament selection operator based on a user-provided tournament size.
"""
function select_trnmt(
    rng::AbstractRNG,
    pop::AbstractVector{EvaluatedGenotype},
    n_select::Int;
    size_trnmt::Int=4,
)
    selection = []
    while size(selection, 1) < n_select
        idx_trnmt = rand(eachindex(pop), size_trnmt)
        winner =
            argmax(model -> getproperty(model, :fitness), view(pop, idx_trnmt))
        push!(selection, winner)
    end

    report = (;)
    return selection, report
end

"""
Compute bounds of the biased window.
"""
function biasedwindow_bounds(len_best, width_window, bias_window, size_pop)
    # (4) in ryerkerk2020.
    len_lbound =
        min(Int(ceil(len_best - width_window / 2 + bias_window)), len_best)
    len_ubound =
        max(Int(floor(len_best + width_window / 2 + bias_window)), len_best)
    # ryerkerk2020's code says “winStart(winStart < 1 ) = 1; % Make sure the
    # window doesn't start at a length less than 1.”.
    len_lbound = max(len_lbound, 1)

    # ryerkerk2020: “In our implemented code the stretch was limited such that
    # the window can only contain a maximum of [`size_pop/2`] solution lengths.”
    #
    # ryerkerk2020's code fulfills this by repeatedly removing the niche that is
    # furthest from the current best solution's length.
    #
    # The `+1` is required because the range is inclusive.
    while len_ubound - len_lbound + 1 > size_pop / 2
        # TODO This assumes that len_lbound <= len_best (analogously for
        # len_ubound)
        if len_best - len_lbound > len_ubound - len_best
            len_lbound += 1
        else
            len_ubound -= 1
        end
    end

    return len_lbound, len_ubound
end
