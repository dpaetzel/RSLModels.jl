using Logging
using MLJ
using Random
using RSLModels.GARegressors
using RSLModels.Transformers

let
    Xmat, y = rand(300, 5), rand(300)
    X = MLJ.table(Xmat)
    ffunc = GARegressors.mkffunc(GARegressors.MAEFitness(Xmat, y))
    config = GARegressor(;
        x_min=0.0,
        x_max=1.0,
        n_iter=5,
        # Increasing this decreases the time drawing a random set
        # of conditions takes.
        init_spread_min=0.4,
    )

    @testset "machine" begin
        rng = Random.Xoshiro(123)

        let config = deepcopy(config)
            config.rng = rng

            Logging.with_logger(SimpleLogger(Logging.Warn)) do
                mach = machine(config, X, y)
                fit!(mach)
                return y_pred = predict(mach, X)
            end
            # Right now we only test whether all types work out etc.
            @test true
        end
    end

    @testset "pipeline" begin
        rng = Random.Xoshiro(123)

        let config = deepcopy(config)
            config.rng = rng

            # Logging.with_logger(SimpleLogger(Logging.Warn)) do
                #! format: off
                mach = machine(MinMaxScaler() |> config, X, y)
                #! format: on
            fit!(mach)
            y_pred = predict(mach, X)
            # end
            # Right now we only test whether all types work out etc.
            @test true
        end
    end
end
