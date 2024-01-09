#!/usr/bin/env bash
#SBATCH --partition=cpu-prio
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=oc-compute03
#SBATCH --time=1:00:00
#SBATCH --array=0-1000%4
#SBATCH --output=output/%x-%A-%a.txt

srun nix develop . --impure --command julia -p $SLURM_CPUS_PER_TASK  --project=. scripts/genkdata.jl --usemmap --n-iter=2000
