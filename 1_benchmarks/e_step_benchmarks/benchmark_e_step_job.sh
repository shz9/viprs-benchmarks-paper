#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=10:00:00
#SBATCH --output=./log/analysis/e_step_benchmarks/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

# ------------------------------------------------------------
# Inputs:

CHROM=${1:-22}  # Chromosome number (default 22)

# ------------------------------------------------------------
# Time the E-Step of the old implementation of VIPRS:

# Activate the virtual environment:
source env/viprs-old/bin/activate

# Call the benchmarking script:
python3 1_benchmarks/e_step_benchmarks/benchmark_e_step_old_viprs.py \
      --temp-dir temp \
      --output-dir data/benchmark_results/e_step/old_viprs/ \
      --ld-panel "data/ld/eur/old_format/ukbb_50k_windowed/chr_$CHROM" \
      --sumstats "data/sumstats/benchmark_sumstats/train/fold_1/chr_$CHROM.PHENO1.glm.linear" \
      --sumstats-format plink \
      --file-prefix "chr_${CHROM}_"

# Deactivate the virtual environment:
deactivate

# ------------------------------------------------------------
# Time the E-Step of the new implementation of VIPRS:

# Activate the virtual environment:
source env/viprs/bin/activate

# Test with new LD format / keep float precision same:
python3 1_benchmarks/e_step_benchmarks/benchmark_e_step.py \
      --temp-dir temp \
      --output-dir data/benchmark_results/e_step/new_viprs/ \
      --model all \
      --ld-panel "data/ld/eur/converted/ukbb_50k_windowed/int8/chr_$CHROM" \
      --sumstats "data/sumstats/benchmark_sumstats/train/fold_1/chr_$CHROM.PHENO1.glm.linear" \
      --sumstats-format plink \
      --implementation cpp \
      --threads "2,4,8" \
      --file-prefix "chr_${CHROM}_" \
      --float-precision "float64"

# Test with new LD format / update float precision to float32:
python3 1_benchmarks/e_step_benchmarks/benchmark_e_step.py \
      --temp-dir temp \
      --output-dir data/benchmark_results/e_step/new_viprs/ \
      --model all \
      --ld-panel "data/ld/eur/converted/ukbb_50k_windowed/int8/chr_$CHROM" \
      --sumstats "data/sumstats/benchmark_sumstats/train/fold_1/chr_$CHROM.PHENO1.glm.linear" \
      --sumstats-format plink \
      --implementation cpp \
      --threads "2,4,8" \
      --file-prefix "chr_${CHROM}_"

# Test with float precision float32 and low memory mode:
python3 1_benchmarks/e_step_benchmarks/benchmark_e_step.py \
      --temp-dir temp \
      --output-dir data/benchmark_results/e_step/new_viprs/ \
      --model all \
      --ld-panel "data/ld/eur/converted/ukbb_50k_windowed/int8/chr_$CHROM" \
      --sumstats "data/sumstats/benchmark_sumstats/train/fold_1/chr_$CHROM.PHENO1.glm.linear" \
      --sumstats-format plink \
      --implementation cpp \
      --threads "2,4,8" \
      --file-prefix "chr_${CHROM}_" \
      --low-memory

# Test with float precision float32 and low memory mode + dequantize on the fly:
python3 1_benchmarks/e_step_benchmarks/benchmark_e_step.py \
      --temp-dir temp \
      --output-dir data/benchmark_results/e_step/new_viprs/ \
      --model all \
      --ld-panel "data/ld/eur/converted/ukbb_50k_windowed/int8/chr_$CHROM" \
      --sumstats "data/sumstats/benchmark_sumstats/train/fold_1/chr_$CHROM.PHENO1.glm.linear" \
      --sumstats-format plink \
      --implementation cpp \
      --threads "2,4,8" \
      --file-prefix "chr_${CHROM}_" \
      --low-memory \
      --dequantize-on-the-fly

# ------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"

