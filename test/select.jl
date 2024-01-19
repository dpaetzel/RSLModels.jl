using DataFrames
# using MLJ
using RSLModels.GARegressors
using RSLModels.Intervals
# using RSLModels.Models
# using RSLModels.Transformers
using Random
import StatsBase: countmap, sample

let
    DX = 5
    X, y = rand(300, DX), rand(300)
    randsol(rng, len) = Intervals.draw_interval.(sample(rng, eachrow(X), len))
    randpop(rng, lens) = randsol.(Ref(rng), lens)
    x_min = Intervals.X_MIN
    x_max = Intervals.X_MAX
    ffunc = GARegressors.mkffunc(GARegressors.MAEFitness(X, y))
    @testset "trnmt_plan" begin
        rng = Random.Xoshiro(123)
        for size_subpop in 1:11
            for size_niche in 1:size_subpop
                tournaments = GARegressors.trnmt_plan(
                    rng,
                    collect(1:size_subpop),
                    size_niche,
                )
                # Each individual takes part in at least one tournament.
                for idx in 1:size_subpop
                    @test idx in tournaments
                end
                # Each individual occurs at most once in each tournament.
                for idx in eachindex(eachcol(tournaments))
                    @test allunique(tournaments[:, idx])
                end
                # Each individual takes part in not more than two tournaments.
                @test all(values(countmap(tournaments)) .<= 2)
            end
        end
    end

    @testset "select_trnmt_niche" begin
        rng = Random.Xoshiro(123)
        for size_subpop in 1:11
            for size_niche in 1:size_subpop
                subpop = collect(1:size_subpop)
                selection, _ = GARegressors.select_trnmt_niche(
                    rng,
                    subpop,
                    size_niche;
                    # We don't use `identity` here so that we can detect when
                    # fitness values are accidentally used as indexes.
                    fitness=x -> x + 100,
                )
                # The requested number of individuals is selected.
                @test length(selection) == size_niche
                # Only individuals from the population are selected.
                for g in selection
                    @test g in subpop
                end
                # Each individual occurs at most twice overall.
                @test all(values(countmap(selection)) .<= 2)
            end
        end
    end

    @testset "select" begin
        rng = Random.Xoshiro(123)

        let size_pop = 32
            lengths = 1:10
            nmatch_min = 2
            pop_uneval = randpop(rng, rand(rng, lengths, size_pop))
            pop =
                GARegressors.evaluate.(
                    pop_uneval,
                    Ref(X),
                    Ref(y),
                    Ref(x_min),
                    Ref(x_max),
                    Ref(ffunc),
                    Ref(nmatch_min),
                )
            for n_select in 1:size_pop
                for len_l in 1:(maximum(length.(pop)) + 1)
                    for len_u in len_l:(maximum(length.(pop)) + 1)
                        selection, _ = GARegressors.select(
                            rng,
                            pop,
                            n_select,
                            collect(len_l:len_u),
                        )
                        # Selection size is as requested.
                        @test length(selection) == n_select
                        # Selection returns only stuff that is in `pop`.
                        for g in selection
                            # Since `EvaluatedGenotype` uses `auto_hash_equals`, we can
                            # safely compare them.
                            @test g in pop
                        end
                    end
                end
            end
        end
    end

    @testset "biasedwindow_bounds" begin
        rng = Random.Xoshiro(123)

        for _ in 1:100
            len_best = rand(rng, 1:30)
            width_window = rand(rng, [5, 7, 9])
            bias_window = rand(rng) * 5 - 2.5
            size_pop = rand(rng, [8, 16, 32])
            len_lbound, len_ubound = GARegressors.biasedwindow_bounds(
                len_best,
                width_window,
                bias_window,
                size_pop,
            )

            lengths = collect(len_lbound:len_ubound)
            @test length(lengths) <= size_pop / 2

            # Check whether the vector is sorted and does not contain
            # duplicates.
            @test all(lengths .>= 1)
        end
    end

    @testset "select_trnmt" begin
        rng = Random.Xoshiro(123)

        let size_pop = 32
            lengths = 1:10
            nmatch_min = 2
            pop_uneval = randpop(rng, rand(rng, lengths, size_pop))
            pop =
                GARegressors.evaluate.(
                    pop_uneval,
                    Ref(X),
                    Ref(y),
                    Ref(x_min),
                    Ref(x_max),
                    Ref(ffunc),
                    Ref(nmatch_min),
                )
            for n_select in 1:size_pop
                for size_trnmt in 1:(size_pop + 1)
                    selection, _ = GARegressors.select_trnmt(
                        rng,
                        pop,
                        n_select;
                        size_trnmt=size_trnmt,
                    )
                    # Selection size is as requested.
                    @test length(selection) == n_select
                    # Selection returns only stuff that is in `pop`.
                    for g in selection
                        # Since `EvaluatedGenotype` uses `auto_hash_equals`, we can
                        # safely compare them.
                        @test g in pop
                    end
                end
            end
        end
    end
end
