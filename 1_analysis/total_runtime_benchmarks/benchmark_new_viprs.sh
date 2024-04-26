#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=00:15:00
#SBATCH --output=./log/analysis/total_runtime_benchmarks/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

# Take the cross validation fold as an argument (default to `fold_1`):
cv_fold=${1:-"fold_1"}
ld_dtype=${2:-"int8"}
threads=${3:-1}
jobs=${4:-1}

model_id="l${ld_dtype}_t${threads}_j${jobs}"

# Activate the virtual environment:
source env/viprs/bin/activate

mkdir -p "data/benchmark_results/total_runtime/$cv_fold/"

# Call the benchmarking script:
/usr/bin/time -v viprs_fit -l "data/ld/eur/converted/ukbb_50k_windowed/$ld_dtype/chr_*/" \
                          -s "data/sumstats/benchmark_sumstats/train/$cv_fold/chr_*.PHENO1.glm.linear" \
                          --output-dir "data/model_fit/benchmark_sumstats/$cv_fold/new_viprs/" \
                          --output-file-prefix "$model_id" \
                          --threads "$threads" \
                          --n-jobs "$jobs" \
                          --output-profiler-metrics \
                          --sumstats-format "plink" 2> "data/benchmark_results/total_runtime/$cv_fold/new_viprs/$model_id.txt"

# Perform evaluation using GWAS summary statistics from independent test set:
python 1_analysis/total_runtime_benchmarks/sumstats_evaluate.py \
        --fit-files "data/model_fit/benchmark_sumstats/$cv_fold/new_viprs/$model_id*.fit*" \
        --test-sumstats "data/sumstats/benchmark_sumstats/test/height_test_$cv_fold.csv.gz" \
        --output-dir "data/benchmark_sumstats/prediction/$cv_fold/new_viprs/$model_id.csv"

# ------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
