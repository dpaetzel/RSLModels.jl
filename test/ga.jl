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

    @testset "mutate_bounds" begin
        rng = Random.Xoshiro(123)

        let config = deepcopy(config)
            config.rng = rng
            pop, _ = GARegressors.init(config, ffunc, X, y)

            for g in pop
                g_ = GARegressors.mutate_bounds(rng, g, config)

                # Mutating bounds does not change length.
                @test length(g_) == length(g)

                for condition in g_
                    # Input space bounds are respected.
                    @test all(config.x_min .<= condition.lbound)
                    @test all(condition.ubound .<= config.x_max)

                    # This is rather trivial because the `Interval` constructor
                    # also checks for it.
                    @test all(condition.lbound .<= condition.ubound)

                    # Note that we allow conditions to be empty after mutation.
                    # @test !isempty(condition)
                end
            end
        end
    end

    @testset "mutate" begin
        rng = Random.Xoshiro(123)

        let config = deepcopy(config)
            config.rng = rng
            pop, _ = GARegressors.init(config, ffunc, X, y)

            for g in pop
                g_, _ = GARegressors.mutate(rng, g, X, config)

                # Mutation adds or removes at most one metavariable.
                @test length(g) - 1 <= length(g_) <= length(g) + 1
            end
        end
    end

    @testset "express" begin
        rng = Random.Xoshiro(123)

        let config = deepcopy(config)
            config.rng = rng
            pop, _ = GARegressors.init(config, ffunc, X, y)

            for g in pop
                model =
                    GARegressors.express(g, X, y, config.x_min, config.x_max)

                @test any([lm.isdefault for lm in model.local_models])
            end
        end
    end
end
