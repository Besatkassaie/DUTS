#!/bin/bash
#SBATCH -p ALL
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /u6/bkassaie/DUTS/experiments/results/wdc_retr_%x_%j.log
cd /u6/bkassaie/DUTS
PYTHONPATH=. /u6/bkassaie/.conda/envs/TableUnionNew/bin/python -m experiments.wdc.retrieval_breakdown "$@"
