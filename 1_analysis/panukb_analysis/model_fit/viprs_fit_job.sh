#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=./log/model_fit/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL


echo "Job started at: `date`"

sumstats_file=${1:-"data/sumstats/panukb_sumstats/EUR/50.sumstats.gz"}
ld_dir=${2:-"data/ld/hq_imputed_variants_hm3/EUR/chr_*"}
output_dir=${3:-"data/model_fit/panukb_sumstats/hq_imputed_variants_hm3/EUR/50/"}
threads=${4:-2}

source "env/viprs/bin/activate"

viprs_fit -l "$ld_dir" \
         -s "$sumstats_file" \
         --sumstats-format "magenpy" \
         --output-dir "$output_dir" \
         --hyp-search "BMA" \
         --pi-steps 30 \
         --dequantize-on-the-fly \
         --output-profiler-metrics \
         --threads "$threads"

echo "Job finished with exit code $? at: `date`"
