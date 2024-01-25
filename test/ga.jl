using DataFrames
using MLJ
using RSLModels.GARegressors
using RSLModels.Intervals
using RSLModels.Models
using RSLModels.Transformers
using Random
using StatsBase

# TODO Deduplicate this file

let
    DX = 5
    X, y = rand(300, DX), rand(300)
    randsol(rng, len) = Intervals.draw_interval.(sample(rng, eachrow(X), len))
    randpop(rng, lens) = randsol.(Ref(rng), lens)
    x_min = Intervals.X_MIN
    x_max = Intervals.X_MAX
    ffunc = GARegressors.mkffunc(GARegressors.NegMAEFitness(X, y))
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
                init1 = GARegressors.mkinit1_custom(
                    config.x_min,
                    config.x_max,
                    config.init_spread_min,
                    config.init_spread_max,
                    config.init_params_spread_a,
                    config.init_params_spread_b,
                    config.init_rate_coverage_min,
                )
                pop, _ = GARegressors.init(rng, X, init1, config.size_pop)
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
                init1 = GARegressors.mkinit1_inverse(
                    config.x_min,
                    config.x_max,
                    df,
                )
                pop, _ = GARegressors.init(rng, X, init1, config.size_pop)
                @test length(pop) == config.size_pop
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
            init1 = GARegressors.mkinit1_custom(
                config.x_min,
                config.x_max,
                config.init_spread_min,
                config.init_spread_max,
                config.init_params_spread_a,
                config.init_params_spread_b,
                config.init_rate_coverage_min,
            )
            pop, _ = GARegressors.init(rng, X, init1, config.size_pop)

            for g in pop
                g_ = GARegressors.mutate_bounds(
                    rng,
                    g,
                    config.mutate_rate_mut,
                    config.mutate_rate_std,
                    config.x_min,
                    config.x_max,
                )

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
            init1 = GARegressors.mkinit1_custom(
                config.x_min,
                config.x_max,
                config.init_spread_min,
                config.init_spread_max,
                config.init_params_spread_a,
                config.init_params_spread_b,
                config.init_rate_coverage_min,
            )
            pop, _ = GARegressors.init(rng, X, init1, config.size_pop)

            for g in pop
                g_, _ = GARegressors.mutate(
                    rng,
                    g,
                    X,
                    config.mutate_rate_mut,
                    config.mutate_rate_std,
                    config.mutate_p_add,
                    config.mutate_p_rm,
                    config.init_spread_min,
                    config.init_spread_max,
                    config.init_params_spread_a,
                    config.init_params_spread_b,
                    config.x_min,
                    config.x_max,
                )

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
            init1 = GARegressors.mkinit1_custom(
                config.x_min,
                config.x_max,
                config.init_spread_min,
                config.init_spread_max,
                config.init_params_spread_a,
                config.init_params_spread_b,
                config.init_rate_coverage_min,
            )
            pop, _ = GARegressors.init(rng, X, init1, config.size_pop)

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

        for fiteval in [:negmae, :similarity, :likelihood, :posterior]
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
