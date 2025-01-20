#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --output=./log/data_preparation/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

source env/viprs2/bin/activate

python 0_data_preparation/external_sumstats/process_panukb_sumstats.py

echo "Job finished with exit code $? at: `date`"
