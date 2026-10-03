#!/bin/bash
#SBATCH -p ALL
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH -o /u6/bkassaie/DUTS/experiments/results/baselines_%x_%j.log
# usage: sbatch -J <bench>_<approach> experiments/baselines/sbatch_baseline.sh <bench> <approach> [extra args]
cd /u6/bkassaie/DUTS
PYTHONPATH=. /u6/bkassaie/.conda/envs/TableUnionNew/bin/python -m experiments.baselines.run_starmie_baselines "$@"
