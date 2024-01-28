using DataFrames
using Dates
using StatsBase

# Reduce the number of data sets considered even further.

DXs = [3, 5, 8]
Ks = [2, 4, 12]
rate_coverage_min = 0.9
n_per_d_k_coverage = 2

dname_stats = "../2024-gecco-tasks/2024-01-19T16-28-51.924-task-selection"
df_stats_orig = readstats(; prefix_fname=dname_stats)
df = df_stats_orig

df = subset(
    df,
    "params.d" => (DX -> DX .∈ Ref(DXs)),
    "stats.K" => (K -> K .∈ Ref(Ks)),
    "stats.coverage" => (c -> c .>= rate_coverage_min),
    # "stats.rate_coverage_min" => (c -> c .== rate_coverage_min),
)
display(combine(groupby(df, ["params.d", "stats.K"]), nrow))

function sampleupto(collection, n)
    if size(collection, 1) < n
        return collection
    else
        return sample(collection, n; replace=false)
    end
end

df_fnames = combine(
    groupby(df, ["params.d", "stats.K"]),
    "fname" => fname -> sampleupto(fname, 2),
)

fnames_selected = reduce(vcat, df_fnames.fname_function)
# We replace colons with dashes so scp'ing is easier.
thetime = replace(string(now()), ":" => "-")
folder_sel = "$thetime-task-selection-light"
open("$thetime-copy_selection-light.fish", "w") do f
    write(f, "#!/usr/bin/env fish\n\n")
    write(f, "# CAREFUL! THIS FILE WAS AUTOGENERATED!\n\n")
    write(f, "mkdir $folder_sel\n")
    for fname in fnames_selected
        fname_glob = replace(fname, "stats.jls" => "*")
        write(f, "cp -vu $(fname_glob) $folder_sel\n")
    end
    return write(
        f,
        "echo\n",
        "echo There should be $(length(fnames_selected)) tasks and there are (math (ls $folder_sel | wc -l) / 3).\n",
    )
end
