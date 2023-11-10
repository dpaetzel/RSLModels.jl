#!/usr/bin/env bash
#SBATCH --partition=cpu-prio
#SBATCH --mem=350G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=oc-compute03
#SBATCH --time=4-0:0:0
#SBATCH --output=output/%A-%a.txt

srun nix develop . --impure --command julia -p $SLURM_CPUS_PER_TASK  --project=. scripts/genkdata.jl 10000
