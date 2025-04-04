#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH --time=04:00:00
#SBATCH --output=./log/model_fit/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL


echo "Job started at: `date`"

sumstats_file=${1:-"data/sumstats/panukb_sumstats/EUR/50.sumstats.gz"}
ld_dir=${2:-"data/ld_xarray/hq_imputed_variants_hm3/EUR/block/int8/chr_*"}
output_dir=${3:-"data/model_fit/panukb_sumstats/hq_imputed_variants_hm3/block_int8_mi/EUR/50/"}
hyp_search=${4:-"EM"}
grid_search_mode=${5:-"pathwise"}
threads=${6:-4}

# Parse optional parameters:
extra_params=()

# If hyp_search is set to "GS", then specify the pi steps flag (--pi-steps) to be 20:
if [ "$hyp_search" == "GS" ]; then
  extra_params+=("--pi-steps")
  extra_params+=("20")
fi

# If the hyp_search is set to "GS" and the grid search mode is "pathwise", then add
# pathwise_ to the output file prefix:

if [ "$hyp_search" == "GS" ] && [ "$grid_search_mode" == "pathwise" ]; then
  extra_params+=("--output-file-prefix")
  extra_params+=("pathwise_")
fi

source "env/viprs/bin/activate"

viprs_fit -l "$ld_dir" \
         -s "$sumstats_file" \
         --sumstats-format "magenpy" \
         --output-dir "$output_dir" \
         --lambda-min 0. \
         --hyp-search "$hyp_search" \
         --grid-search-mode "$grid_search_mode" \
         --dequantize-on-the-fly \
         --output-profiler-metrics \
         --threads "$threads" \
         "${extra_params[@]}"

echo "Job finished with exit code $? at: `date`"
