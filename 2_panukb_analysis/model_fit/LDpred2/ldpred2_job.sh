#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=5:00:00
#SBATCH --output=./log/model_fit/panukb_sumstats/external/LDpred2-auto/EUR/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# -----------------------------------------

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

echo "Performing model fit..."
echo "Dataset: $1"

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Extract phenotype name from the input file path by taking basename and removing .sumstats.gz extension:
phenotype=$(basename "$1" | sed 's/\.sumstats\.gz//g')
mkdir -p "data/model_fit/panukb_sumstats/external/LDpred2-auto/EUR/$phenotype/"

module load gcc/12.3 r/4.3.1
export R_LIBS=2_panukb_analysis/model_fit/LDpred2/R_ldpred2_env

SECONDS=0

# Use /usr/bin/time to track the computational resources used by the R script:
/usr/bin/time -o "data/model_fit/panukb_sumstats/external/LDpred2-auto/EUR/$phenotype/LDpred2-auto.prof" \
  -v Rscript "2_panukb_analysis/model_fit/LDpred2/fit_ldpred2_auto.R" "$1"

MINUTES=$(echo "scale=2; $SECONDS/60" | bc)

echo "Job finished with exit code $? at: `date`"
echo "Duration (minutes): $MINUTES"
