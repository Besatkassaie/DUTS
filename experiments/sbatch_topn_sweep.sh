#!/bin/bash
#SBATCH -p ALL
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /u6/bkassaie/DUTS/experiments/results/topn_sweep_%x_%j.log
cd /u6/bkassaie/DUTS
PYTHONPATH=. /u6/bkassaie/.conda/envs/TableUnionNew/bin/python -m "$@"
