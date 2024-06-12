#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=01:00:00
#SBATCH --output=./log/analysis/total_runtime_benchmarks/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

# Exit on error:
set -e

# Take the cross validation fold as an argument (default to `fold_1`):
cv_fold=${1:-"fold_1"}
ld_dtype=${2:-"int8"}
threads=${3:-1}
jobs=${4:-1}
low_mem=${5:-"false"}
dequantize=${6:-"false"}

model_id="l${ld_dtype}_m${low_mem}_q${dequantize}_t${threads}_j${jobs}"

# Activate the virtual environment:
source env/viprs/bin/activate

mkdir -p "data/benchmark_results/total_runtime/$cv_fold/new_viprs_bma/"

# Parse optional parameters:
extra_params=()

# If low_mem, add an option --use-symmetric-ld:
if [ "$low_mem" = "false" ]; then
  extra_params+=(--use-symmetric-ld)
fi

if [ "$dequantize" = "true" ]; then
  extra_params+=(--dequantize-on-the-fly)
fi

echo "==================== Copy the input data to local storage ===================="
# To minimize variation due to network latency, copy the input data to local storage:

# Create temporary directory specific to this job:

tmp_dir="$SLURM_TMPDIR/$SLURM_JOBID"
mkdir -p "$tmp_dir"
mkdir -p "$tmp_dir/ld"
mkdir -p "$tmp_dir/sumstats"

# Copy the LD data to tmp_dir:
cp -r "data/ld/eur/converted/ukbb_50k_windowed/$ld_dtype/"* "$tmp_dir/ld/"

# Copy the summary statistics to tmp_dir:
cp -r "data/sumstats/benchmark_sumstats/train/$cv_fold/"* "$tmp_dir/sumstats/"


echo "=========================== Model fit ==========================="

# Call the benchmarking script:
/usr/bin/time -o "data/benchmark_results/total_runtime/$cv_fold/new_viprs_bma/$model_id.txt" \
              -v viprs_fit -l "$tmp_dir/ld/chr_*/" \
                          -s "$tmp_dir/sumstats/chr_*.PHENO1.glm.linear" \
                          --output-dir "data/model_fit/benchmark_sumstats/$cv_fold/new_viprs_bma/" \
                          --output-file-prefix "$model_id" \
                          --threads "$threads" \
                          --n-jobs "$jobs" \
                          --hyp-search "BMA" \
                          --pi-steps 30 \
                          --output-profiler-metrics \
                          --sumstats-format "plink" \
                          "${extra_params[@]}"

echo "=========================== Model Evaluation ==========================="

# Perform evaluation using GWAS summary statistics from independent test set:
# Use float32 LD panel by default for evaluating the test set:
python 1_analysis/total_runtime_benchmarks/sumstats_evaluate.py \
        --fit-files "data/model_fit/benchmark_sumstats/$cv_fold/new_viprs_bma/$model_id*.fit*" \
        --test-ld-panel "data/ld/eur/converted/ukbb_50k_windowed/float32/chr_*/" \
        --test-sumstats "data/sumstats/benchmark_sumstats/test/height_test_$cv_fold.csv.gz" \
        --output-file "data/benchmark_results/prediction/$cv_fold/new_viprs_bma/$model_id.csv"

# ------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
