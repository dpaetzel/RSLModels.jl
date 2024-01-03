#!/usr/bin/env bash
#SBATCH --partition=cpu-prio
#SBATCH --mem=20G
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=oc-compute03
#SBATCH --time=1:0:0
#SBATCH --output=output/%x-%A-%a.txt

srun nix develop --impure --command julia --project=. -e "import Pkg; Pkg.Registry.update(); Pkg.resolve(); Pkg.instantiate()"
