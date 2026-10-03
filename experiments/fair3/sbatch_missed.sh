#!/bin/bash
#SBATCH -p ALL
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /u6/bkassaie/DUTS/experiments/results/fair3_missed_%x_%j.log
cd /u6/bkassaie/DUTS
PYTHONPATH=. /u6/bkassaie/.conda/envs/TableUnionNew/bin/python -m experiments.fair3.diag_missed_solutions "$@"
