using Infiltrator
using MLJ
using RSLModels.GARegressors
using RSLModels.Models
using RSLModels.Transformers
using Random
using StatsBase

let
    X, y = rand(300, 5), rand(300)
    ffunc = GARegressors.mkffunc(GARegressors.MAEFitness(X, y))
    config = GARegressor(;
        x_min=0.0,
        x_max=1.0,
        # Increasing this decreases the time drawing a random set
        # of conditions takes.
        spread_min=0.4,
        n_iter=5,
    )

    @testset "init: Correct population size" begin
        rng = Random.Xoshiro(123)
        for size_pop in 2:2:32
            let config = deepcopy(config)
                config.size_pop = size_pop
                config.rng = rng
                pop, _ = GARegressors.init(config, ffunc, X, y)
                @test length(pop) == config.size_pop
                # TODO Consider to check whether all training samples matched by
                # at least one rule per individual (i.e. check_matching_matrix)
            end
        end
    end

    @testset "trnmt_plan" begin
        rng = Random.Xoshiro(123)
        for size_subpop in 1:11
            for size_niche in 1:size_subpop
                tournaments = GARegressors.trnmt_plan(
                    rng,
                    collect(1:size_subpop),
                    size_niche,
                )
                # each individual at least once overall
                for idx in 1:size_subpop
                    @test idx in tournaments
                end
                # each individual at most once in each tournament
                for idx in eachindex(eachcol(tournaments))
                    @test allunique(tournaments[:, idx])
                end
                # each individual at most twice overall
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
                # the right number of individuals
                @test length(selection) == size_niche
                # only individuals from the population
                for g in selection
                    @test g in subpop
                end
                # each individual at most twice overall
                @test all(values(countmap(selection)) .<= 2)
            end
        end
    end

    @testset "select" begin
        rng = Random.Xoshiro(123)

        let config = deepcopy(config)
            config.rng = rng
            pop, _ = GARegressors.init(config, ffunc, X, y)
            pop =
                GARegressors.evaluate.(
                    pop,
                    Ref(X),
                    Ref(y),
                    config.x_min,
                    config.x_max,
                    ffunc,
                )
            for n_select in 1:length(pop)
                for len_l in 1:(maximum(length.(pop)) + 1)
                    for len_u in len_l:(maximum(length.(pop)) + 1)
                        selection, _ = GARegressors.select(
                            rng,
                            pop,
                            n_select,
                            collect(len_l:len_u),
                        )
                        # correct selection size
                        @test length(selection) == n_select
                        # selection returns only stuff that is in `pop`
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
end
