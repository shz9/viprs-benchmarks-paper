#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=00:45:00
#SBATCH --output=./log/analysis/total_runtime_benchmarks/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

# Take the cross validation fold as an argument (default to `fold_1`):
cv_fold=${1:-"fold_1"}

# Activate the virtual environment:
source env/viprs-old/bin/activate

mkdir -p "data/benchmark_results/total_runtime/$cv_fold/"

echo "==================== Copy the input data to local storage ===================="
# To minimize variation due to network latency, copy the input data to local storage:

# Create temporary directory specific to this job:

tmp_dir="$SLURM_TMPDIR/$SLURM_JOBID"
mkdir -p "$tmp_dir"
mkdir -p "$tmp_dir/ld"
mkdir -p "$tmp_dir/sumstats"

# Copy the LD data to tmp_dir:
cp -r "data/ld/eur/old_format/ukbb_50k_windowed/"* "$tmp_dir/ld/"

# Copy the summary statistics to tmp_dir:
cp -r "data/sumstats/benchmark_sumstats/train/$cv_fold/"* "$tmp_dir/sumstats/"


echo "=========================== Model fit ==========================="

# Call the benchmarking script:
/usr/bin/time -o "data/benchmark_results/total_runtime/$cv_fold/old_viprs.txt" \
              -v python3 1_analysis/total_runtime_benchmarks/updated_viprs_fit_v004.py \
                          -l "$tmp_dir/ld/chr_*/" \
                          -s "$tmp_dir/sumstats/chr_*.PHENO1.glm.linear" \
                          --output-file "data/model_fit/benchmark_sumstats/$cv_fold/old_viprs" \
                          --sumstats-format "plink"

# Perform evaluation using GWAS summary statistics from independent test set:
deactivate
source env/viprs/bin/activate

python 1_analysis/total_runtime_benchmarks/sumstats_evaluate.py \
        --fit-files "data/model_fit/benchmark_sumstats/$cv_fold/old_viprs*.fit*" \
        --test-ld-panel "data/ld/eur/converted/ukbb_50k_windowed/float32/chr_*/" \
        --test-sumstats "data/sumstats/benchmark_sumstats/test/height_test_$cv_fold.csv.gz" \
        --output-file "data/benchmark_results/prediction/$cv_fold/old_viprs.csv"

# ------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
