#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=./log/model_fit/panukb_sumstats/external/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

##############################################
# Variables: need to be fixed
sumstats_file="$1"               # GWAS summary statistics
ld_reference=${2:-"HapMap3"}
threads=${3:-8}

##############################################

export OMP_NUM_THREADS=$threads # Revise the threads

# Obtain the phenotype name:
phenotype=$(basename "$sumstats_file" | sed 's/\.sumstats\.gz//g')

source env/viprs/bin/activate

# Track start time:
prep_start_time=$(date +%s)

python 2_panukb_analysis/model_fit/SBayesRC/transform_sumstats.py -s "$sumstats_file" \
        --output-dir "$(dirname "$sumstats_file")" \

# Obtain the transformed sumstats file by replacing the .sumstats.gz extension with .ma:
sumstats_file_updated=$(echo "$sumstats_file" | sed 's/\.sumstats\.gz/.ma/g')

# Determine the ld_folder path depending on ld_reference:
if [ "$ld_reference" == "HapMap3" ]; then
  ld_folder="2_panukb_analysis/model_fit/SBayesRC/data/ld/hapmap3/ukbEUR_HM3/"
elif [ "$ld_reference" == "HapMap3-hq" ]; then
  ld_folder="2_panukb_analysis/model_fit/SBayesRC/data/ld/hapmap3_block/ld_data/"
elif [ "$ld_reference" == "HapMap3-hq-4cM" ]; then
  ld_folder="2_panukb_analysis/model_fit/SBayesRC/data/ld/hapmap3_block_4cM/ld_data/"
elif [ "$ld_reference" == "HapMap3-x-4cM" ]; then
  ld_folder="2_panukb_analysis/model_fit/SBayesRC/data/ld/hapmap3x_block_4cM/ld_data/"
elif [ "$ld_reference" == "7m" ]; then
  ld_folder="2_panukb_analysis/model_fit/SBayesRC/data/ld/imputed/ukbEUR_Imputed/"
else
  echo "Invalid ld_reference: $ld_reference"
  exit 1
fi

output_prefix="data/model_fit/panukb_sumstats/external/SBayesRC-${ld_reference}/EUR/${phenotype}/SBayesRC-${ld_reference}"
mkdir -p $(dirname "$output_prefix") || true

prep_end_time=$(date +%s)

# Package into a function:
run_sbayesrc_commands() {

  # Step 1: Impute the summary statistics:
  apptainer run 2_panukb_analysis/model_fit/SBayesRC/sbayesrc_env/sbayesrc_latest.sif \
        --impute-summary \
        --ldm-eigen $ld_folder \
        --gwas-summary "$sumstats_file_updated" \
        --out "$output_prefix" \
        --threads "$threads"

  # Step 2: Perform inference
  apptainer run 2_panukb_analysis/model_fit/SBayesRC/sbayesrc_env/sbayesrc_latest.sif \
                  --sbayes RC \
                  --ldm-eigen "$ld_folder" \
                  --gwas-summary "${output_prefix}.imputed.ma" \
                  --out "$output_prefix" \
                  --thread "$threads"

}

inference_start_time=$(date +%s)

# Export the function + required variables:
export -f run_sbayesrc_commands
export sumstats_file_updated
export output_prefix
export ld_folder
export threads

# Run the function and track its performance with time:
/usr/bin/time -o "${output_prefix}.prof" \
        -v bash -c "run_sbayesrc_commands"

inference_end_time=$(date +%s)

python 2_panukb_analysis/model_fit/SBayesRC/transform_output.py --ld-snp-info "$ld_folder/snp.info" \
        -o "${output_prefix}.txt"

# Delete the transformed sumstats file:
rm "$sumstats_file_updated"

# Write detailed profiling information to a tab-separated file:
echo -e "DataPrep_Time\tFit_time\tTotal_WallClockTime" > "${output_prefix}_detailed.prof"
echo -e "$((prep_end_time - prep_start_time))\t$((inference_end_time - inference_start_time))\t$((inference_end_time - prep_start_time))" >> "${output_prefix}_detailed.prof"

echo "Job finished with exit code $? at: `date`"
