#!/bin/bash
#SBATCH -p ALL
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /u6/bkassaie/DUTS/experiments/results/fair3_eval_%x_%j.log
# usage: sbatch -J tusSmall3 experiments/fair3/sbatch_eval.sh tusSmall3
cd /u6/bkassaie/DUTS
PYTHONPATH=. /u6/bkassaie/.conda/envs/TableUnionNew/bin/python -m experiments.fair3.eval_fair3 $1 1000
