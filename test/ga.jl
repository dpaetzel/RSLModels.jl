using Infiltrator
using DataFrames
using MLJ
using RSLModels.GARegressors
using RSLModels.Models
using RSLModels.Transformers
using Random
using StatsBase

# TODO Deduplicate this file

let
    DX = 5
    X, y = rand(300, DX), rand(300)
    ffunc = GARegressors.mkffunc(GARegressors.MAEFitness(X, y))
    config = GARegressor(;
        x_min=0.0,
        x_max=1.0,
        n_iter=5,
        # Increasing this decreases the time drawing a random set
        # of conditions takes.
        init_spread_min=0.4,
    )
    model = draw_model(DX)

    @testset "init" begin
        rng = Random.Xoshiro(123)
        for size_pop in 2:2:32
            let config = deepcopy(config)
                config.size_pop = size_pop
                config.rng = rng
                pop, _ = GARegressors.init_custom(
                    rng,
                    ffunc,
                    X,
                    config.size_pop,
                    config.x_min,
                    config.x_max,
                    config.init_spread_min,
                    config.init_spread_max,
                    config.init_params_spread_a,
                    config.init_params_spread_b,
                    config.init_rate_coverage_min,
                )
                @test length(pop) == config.size_pop
            end

            let config = deepcopy(config)
                config.size_pop = size_pop
                config.rng = rng
                # `:inverse` init is hard to do here right now because test
                # somehow messes up paths and we can thus not properly set the
                # `init_sample_fname`. We thus build a minimal `DataFrame`
                # manually.
                df = DataFrame(;
                    DX=[DX, DX, DX],
                    K=[5, 10, 13],
                    a=[10.0, 20.0, 30.0],
                    b=[23.0, 22.0, 40.0],
                    spread_min=[0.2, 0.3, 0.4],
                    rate_coverage_min=[0.8, 0.8, 0.8],
                )
                pop, _ = GARegressors.init_inverse(
                    rng,
                    ffunc,
                    X,
                    config.size_pop,
                    config.x_min,
                    config.x_max,
                    df,
                )
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
            # `:inverse` init is hard to do here right now because test somehow
            # messes up paths and we can thus not properly set the
            # `init_sample_fname`.
            pop, _ = GARegressors.init_custom(
                rng,
                ffunc,
                X,
                config.size_pop,
                config.x_min,
                config.x_max,
                config.init_spread_min,
                config.init_spread_max,
                config.init_params_spread_a,
                config.init_params_spread_b,
                config.init_rate_coverage_min,
            )
            pop =
                GARegressors.evaluate.(
                    pop,
                    Ref(X),
                    Ref(y),
                    config.x_min,
                    config.x_max,
                    ffunc,
                    config.nmatch_min,
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
            # `:inverse` init is hard to do here right now because test somehow
            # messes up paths and we can thus not properly set the
            # `init_sample_fname`.
            pop, _ = GARegressors.init_custom(
                rng,
                ffunc,
                X,
                config.size_pop,
                config.x_min,
                config.x_max,
                config.init_spread_min,
                config.init_spread_max,
                config.init_params_spread_a,
                config.init_params_spread_b,
                config.init_rate_coverage_min,
            )

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

    @testset "mirror" begin
        rng = Random.Xoshiro(123)

        a = rand(rng, 100)
        for i in eachindex(a)
            @test 0.3 <= GARegressors.mirror(a[i], 0.3, 0.6) <= 0.6
        end
    end

    @testset "mutate" begin
        rng = Random.Xoshiro(123)

        let config = deepcopy(config)
            config.rng = rng
            # `:inverse` init is hard to do here right now because test somehow
            # messes up paths and we can thus not properly set the
            # `init_sample_fname`.
            pop, _ = GARegressors.init_custom(
                rng,
                ffunc,
                X,
                config.size_pop,
                config.x_min,
                config.x_max,
                config.init_spread_min,
                config.init_spread_max,
                config.init_params_spread_a,
                config.init_params_spread_b,
                config.init_rate_coverage_min,
            )

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
            # `:inverse` init is hard to do here right now because test somehow
            # messes up paths and we can thus not properly set the
            # `init_sample_fname`.
            pop, _ = GARegressors.init_custom(
                rng,
                ffunc,
                X,
                config.size_pop,
                config.x_min,
                config.x_max,
                config.init_spread_min,
                config.init_spread_max,
                config.init_params_spread_a,
                config.init_params_spread_b,
                config.init_rate_coverage_min,
            )

            for g in pop
                model = GARegressors.express(
                    g,
                    X,
                    y,
                    config.x_min,
                    config.x_max,
                    config.nmatch_min,
                )

                @test any([lm.isdefault for lm in model.local_models])
            end
        end
    end

    @testset "runga" begin
        rng = Random.Xoshiro(123)

        for fiteval in [:mae, :dissimilarity, :likelihood]
            let config = deepcopy(config)
                config.rng = rng
                config.fiteval = fiteval
                config.dgmodel = model
                # `:inverse` init is hard to do here right now because test
                # somehow messes up paths and we can thus not properly set the
                # `init_sample_fname`.
                config.init = :custom

                garesult, report = GARegressors.runga(X, y, config)

                # This may change depending on the exact GA implementation used.
                @test report.n_eval == config.size_pop * (config.n_iter + 1)
            end
        end
    end

    @testset "nmatch_min" begin
        rng = Random.Xoshiro(123)

        let config = deepcopy(config)
            config.rng = rng
            # `:inverse` init is hard to do here right now because test
            # somehow messes up paths and we can thus not properly set the
            # `init_sample_fname`.
            config.init = :custom

            for nmatch_min in [2, 5]
                config.nmatch_min = nmatch_min
                garesult, report = GARegressors.runga(X, y, config)
                for condition in garesult.best.phenotype.conditions
                    @test sum(elemof(X, condition)) >= nmatch_min
                end
            end
        end
    end
end
