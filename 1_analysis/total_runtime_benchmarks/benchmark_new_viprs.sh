#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=00:30:00
#SBATCH --output=./log/analysis/total_runtime_benchmarks/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# Take the cross validation fold as an argument (default to `fold_1`):
cv_fold=${1:-"fold_1"}

# Activate the virtual environment:
source env/viprs-old/bin/activate

mkdir -p data/benchmark_results/total/

# Call the benchmarking script:
/usr/bin/time -v viprs_fit -l "data/ld/eur/converted/ukbb_50k_windowed/int8/chr_*/" \
                          -s "data/sumstats/benchmark_sumstats/chr_*" \
                          --output-file "data/model_fit/benchmarks/$cv_fold/new_viprs" \
                          --sumstats-format "plink" 2> "data/benchmark_results/total/new_viprs_fold_${cv_fold}.txt"