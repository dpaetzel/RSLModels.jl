using BenchmarkTools
using Cthulhu
using Infiltrator
using JET
using MLFlowClient
using MLJ
using NPZ
using PProf
using Profile
using RSLModels.AbstractModels
using RSLModels.GARegressors
using RSLModels.MLFlowUtils

const fname = "3-18-796-30-0.9-true.data.npz"
const data = npzread(fname)
const X = data["X"]
const y = data["y"]
const X_test = data["X_test"]
const y_test = data["y_test"]
const model =
    GARegressor(; rng=42, n_iter=100, init_sample_fname="2024-01-09-16-kdata")
const mlf = getmlf()

function train(; model=model, X=X, y=y, X_test=X_test, y_test=y_test)
    garesult, report = GARegressors.runga(X, y, model; verbosity=1000)
    return garesult
    # y_test_pred = output_mean(garesult.best.phenotype, X_test)
    # return garesult, y_test_pred
end

function bmark_train(; mlf=mlf)
    name_exp = "bmark_train"
    mlfexp = getorcreateexperiment(mlf, name_exp)
    mlfrun = createrun(mlf, mlfexp)

    train()
    t = @benchmark train()
    show(t)
    for metric in [minimum, median]
        logmetric.(
            Ref(mlf),
            Ref(mlfrun),
            [
                "$(string(metric)).time",
                "$(string(metric)).mem",
                "$(string(metric)).allocs",
            ],
            [
                metric(t).time / 1000 / 1000,
                metric(t).memory / 1000 / 1000,
                metric(t).allocs,
            ],
        )
    end

    updaterun(mlf, mlfrun, "FINISHED")
    return t
end

# 2023-12-22: one Trial takes 20-30 s. See mlflow for details though.

function doprofile()
    train()

    @profile train()
    return pprof(; web=false)
    # Run `pprof --web profile.pb.gz` afterwards
end

bmark_train()

# ffunc = GARegressors.mkffunc(GARegressors.MAEFitness(X, y))

# @report_opt GARegressors.init(ffunc, X, y, model)
