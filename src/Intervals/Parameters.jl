module Parameters

using ProgressMeter
using CSV
using DataFrames
using Distributions
using ScientificTypes
using SHA
using Statistics
using StatsBase
using Tables

export histmode, selectparams

# TODO Lock this header with genkdata.jl
const csv_header::Vector{Symbol} =
    [:DX, :rate_coverage_min, :spread_min, :a, :b, :rate_coverage, :K]

"""
If the given path is a file, SHA256 hash its contents and return the resulting
string. If the given path is a directory, SHA256 hash the contents of all files
directy below it and then SHA256 these concatenated hashes again to get a single
hash for the directory contents.

Note that further directories in `path` are ignored (i.e. we go only one level
deep).

Further, note that we ignore file names and only consider contents.
"""
function hashpath(path)
    if isfile(path)
        content = read(path)
        return bytes2hex(sha256(content))
    elseif isdir(path)
        hashs = String[]
        for fname in readdir(path)
            fpath = joinpath(path, fname)
            if isfile(fpath)
                content = read(fpath)
                push!(hashs, bytes2hex(sha256(content)))
            end
        end
        return bytes2hex(sha256(reduce((*), hashs)))
    else
        error(
            "hashpath: Provided path is neither a file nor a directory: $path",
        )
    end
end

function readdata(fname; dropcensored=true, collapsemedian=false, verbosity=0)
    df = if isfile(fname)
        # Read the file into a DataFrame
        out = DataFrame(CSV.File(fname; header=csv_header))
        if verbosity > 9
            @info "Loaded CSV file $fname."
        end
        out
    elseif isdir(fname)
        combined_df = DataFrame()

        fnames = readdir(fname)
        if verbosity > 0
            @info "Loading $(length(fnames)) CSV files from directory \"$fname\" …"
        end
        @showprogress desc = "Loading CSVs …" enabled = verbosity > 9 for f in
                                                                          fnames
            fpath = joinpath(fname, f)
            if isfile(fpath) && endswith(fpath, ".csv")
                df = DataFrame(CSV.File(fpath; header=csv_header))
                combined_df = vcat(combined_df, df)
            end
        end

        if verbosity > 0
            @info "Loaded $(length(fnames)) CSV files from directory \"$fname\"."
        end
        combined_df
    else
        error("readdata: Provided path is neither a file nor a directory: $path")
    end

    # When generating the data for this, we cancel trials at some number of
    # rules to not waste time with configurations that we're not interested in.
    # We can filter these based on whether the coverage condition is fulfilled
    # (if we cancel early, the coverage condition is not yet fulfilled).
    # Filtering these out may of course result in some of the configurations
    # having less trials than others. This could be done more elegantly but
    # fully serves our purpose for now.
    if dropcensored
        df = df[df.rate_coverage .>= df.rate_coverage_min, :]
    end

    if collapsemedian
        # Collapse the distributions of the `K`s to their median right away.
        df = combine(
            groupby(df, [:DX, :rate_coverage_min, :spread_min, :a, :b]),
            :K => median => :K_median,
            :rate_coverage => median => :rate_coverage_median,
        )
    end

    # # Set scientific types.
    df[!, :DX] = coerce(df.DX, Count)
    df[!, collapsemedian ? :K_median : :K] =
        coerce(df[:, collapsemedian ? :K_median : :K], Count)
    return df
end

"""
    histmode(data; nbins::Int=ceil(nrows(data)))

Compute a (multivariate) histogram of the data and approximate the underlying
density's mode by computing the mean of the data points falling into
highest-density bin.

Returns a pair of the approximated mode and the number of data points falling
into the highest-density bin (i.e. the ones whose mean is the mode
approximation).
"""
function histmode end

function histmode(data::NamedTuple; nbins=Int(ceil(length(data[1])^(1 / 3))))
    matrix = hcat(data...)
    out, n_maxbin = histmode(matrix; nbins=nbins)
    return NamedTuple(vcat(keys(data) .=> out, :n_maxbin => n_maxbin))
end

# TODO Default number of bins is probably suboptimal
function histmode(
    matrix::AbstractMatrix{Float64};
    nbins=Int(ceil(size(matrix, 1)^(1 / 3))),
)
    @debug "Using $nbins bins per dimension."

    # TODO Consider whether Matrix/Tuple data transformations can be done better
    hst = fit(Histogram, Tuple([col for col in eachcol(matrix)]); nbins=nbins)

    # For sanity checking.
    n_maxbin = maximum(hst.weights)

    # Get the `CartesianIndex` of the bin with the most data points.
    idx_maxbin = argmax(hst.weights)

    # TODO Consider whether Matrix/Tuple data transformations can be done better
    # Get bin index for each data point. Transform matrix rows to tuples (since
    # `binindex` only accepts tuples).
    idxs =
        CartesianIndex.(
            StatsBase.binindex.(
                Ref(hst),
                [Tuple(row) for row in eachrow(matrix)],
            )
        )

    # Check which data points are in the highest density bin.
    idxs_max = idxs .== Ref(idx_maxbin)

    # Sanity check.
    @assert count(idxs_max) == n_maxbin

    # Extract data points using Bool indexing on the matrixified form of the
    # data.
    dfmax = matrix[idxs_max, :]

    @debug "Estimating mode from mean of max bin containing " *
           "$n_maxbin of $(size(matrix, 1)) data points."

    # Compute mean.
    return mean.(eachcol(dfmax)), n_maxbin
end

"""
Used internally to memoize `selectparams` results.
"""
const CACHE = Dict{Tuple{String,Tuple{Vararg{Int}}},DataFrame}()

"""
    selectparams(fname, K...; <keyword arguments>)

For each `K` given, crudely estimate sensible values for the `draw_intervals`
parameters `a`, `b` and `spread_min`.

Estimation is done by analysing a sample from the joint distribution of `K`,
`a`, `b` and `spread_min` which is assumed to be available as a CSV file
(presumably generated beforehand by `genkdata.jl`) or a directory containing CSV
files to be merged.

# Arguments

- `fname`: Name of a CSV file (or a directory of such files) as the ones
  generated by `genkdata.jl` containing samples of statistics of the condition
  set–generating process.
- `K`: Target numbers of conditions to be generate with the selected parameters.
- `verbosity::Int=0`:
"""
function selectparams(fname, K...; verbosity=0, unsafe=false)
    # For medium-sized samples it seems to be several seconds cheaper to hash
    # the CSV files and check whether we computed the corresponding `DataFrame`
    # already instead of recomputing the `DataFrame`. This comes in especially
    # handy when doing hyperparameter optimization (note that it's complicated
    # to pass the `DataFrame` as an argument to the learning algorithm right
    # now because of JSON serialization and stuff).
    hash = if !unsafe
        hashpath(fname)
    else
        fname
    end
    # Try to get the key or if it is not present, compute the do block, store
    # the key with its result and then return the result.
    get!(CACHE, (hash, K)) do
        # TODO Cache the result as a jls file based on a hash of the result of
        # readdata and K
        if verbosity > 9
            @info "Reading data from location \"$fname\" …"
        end
        df = readdata(fname; verbosity=verbosity)
        if verbosity > 9
            @info "Read data from location \"$fname\"."
        end

        K_problem = filter(K_ -> K_ ∉ df.K, K)
        if !isempty(K_problem)
            @warn "Requested K ∈ $K_problem not in data, not attempting to " *
                  "select parameters for it"
            K = filter(K_ -> K_ ∉ K_problem, K)
        end

        # Select the observations from the sample that are of interest (i.e. the
        # ones that fulfill our `K` condition).
        df_sel = subset(df, :K => K_ -> K_ .∈ Ref(K))

        df_sel_mode = combine(
            groupby(df_sel, [:DX, :rate_coverage_min, :K]),
            nrow => :count,
            [:spread_min, :a, :b] =>
                ((s, a, b) -> histmode((; spread_min=s, a=a, b=b))) =>
                    AsTable,
        )
        df_sel_mode = sort(df_sel_mode)

        df_sel_mode[!, :Beta] = Beta.(df_sel_mode.a, df_sel_mode.b)

        df_sel_mode[!, :Beta_mean] = mean.(df_sel_mode.Beta)

        df_sel_mode[!, :Beta_var] = var.(df_sel_mode.Beta)

        df_sel_mode[!, :Beta_std] = std.(df_sel_mode.Beta)

        # This is simply the formula from `Intervals.draw_spread`.
        df_sel_mode[!, :spread_mean] =
            df_sel_mode.spread_min .+
            mean.(df_sel_mode.Beta) .* (0.5 .- df_sel_mode.spread_min)

        select!(df_sel_mode, Not(:Beta))

        return df_sel_mode
    end
end

end
