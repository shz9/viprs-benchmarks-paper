#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=00:40:00
#SBATCH --output=./log/analysis/total_runtime_benchmarks/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

# Take the cross validation fold as an argument (default to `fold_1`):
cv_fold=${1:-"fold_1"}

# Activate the virtual environment:
source env/viprs-old/bin/activate

mkdir -p data/benchmark_results/total_runtime/

# Call the benchmarking script:
/usr/bin/time -v viprs_fit -l "data/ld/eur/old_format/ukbb_50k_windowed/chr_*/" \
                          -s "data/sumstats/benchmark_sumstats/$cv_fold/chr_*.PHENO1.glm.linear" \
                          --output-file "data/model_fit/benchmark_sumstats/$cv_fold/old_viprs" \
                          --sumstats-format "plink" 2> "data/benchmark_results/total_runtime/old_viprs_${cv_fold}.txt"

# ------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
