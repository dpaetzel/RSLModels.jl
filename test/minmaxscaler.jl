using MLJ
using Random
using RSLModels.Transformers

@testset "minmaxscaler" begin
    for seed in 1:50
        let Xmat = rand(Random.Xoshiro(seed), 300, 5) .* 5.0 .- 2.0
            X = MLJ.table(Xmat)

            model = MinMaxScaler()
            mach = machine(model, X)
            fit!(mach)
            X_ = MLJ.transform(mach, X)
            X_ = MLJ.matrix(X_)
            @test all(0.0 .<= X_) && all(X_ .<= 1.0)
        end
    end
end
