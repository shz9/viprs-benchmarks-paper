#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=60:00:00
#SBATCH --output=./log/data_preparation/ld_mat/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

. global_config.sh

module load plink
source "env/viprs/bin/activate"

LD_EST=${1:-"windowed"}  # LD estimator (default "windowed")
BEDFILE=${2:-"data/ukbb_qc_genotypes/chr_22"}  # Bed file (default 22)
POP=${3:-"EUR"}  # Population (default "EUR")
KEEPFILE=${4:-"data/keep_files/ukbb_qc_individuals_${POP}.keep"}  # Keep file for individuals
EXTRACTFILE=${5:-"data/keep_files/hq_imputed_variants_hm3.txt"}  # Extract file for SNPs
VARIANT_SET=$(basename $EXTRACTFILE .txt)
DTYPE=${6:-"int8"}  # Data type for the LD matrix
BACKEND=${7:-"plink"}  # Backend for the LD computation

# If backend is plink, set default output directory to data/ld/
# else, set it to data/ld_xarray/:

if [[ "${BACKEND}" == "plink" ]]
then
  default_output_dir="data/ld"
else
  default_output_dir="data/ld_xarray"
fi

OUTPUTDIR=${8:-"${default_output_dir}/${VARIANT_SET}/${POP}/${LD_EST}/${DTYPE}"}


echo "Computing the LD matrix for:"
echo "> Chromosome: $(basename $BEDFILE)"
echo "> LD estimator: ${LD_EST}"
echo "> Population: ${POP}"
echo "> Keep file: $(basename $KEEPFILE)"
echo "> Extract file: $(basename $EXTRACTFILE)"
echo "> Storage data type: ${DTYPE}"

SECONDS=0

mkdir -p "$OUTPUTDIR"
module load plink

# Parse optional parameters:
extra_params=()

# If the estimator is windowed, add the window size to the parameters:
# else, add the path to the LDetect blocks for the block estimator:
if [[ "${LD_EST}" == "windowed" ]]
then
  extra_params+=("--ld-window-cm" "3")
else
  extra_params+=("--ld-blocks" "data/ldetect_data/${POP}_blocks.bed")
fi

magenpy_ld \
    --bfile "$BEDFILE" \
    --keep "$KEEPFILE" \
    --extract "$EXTRACTFILE" \
    --min-mac 20 \
    --backend "$BACKEND" \
    --estimator "${LD_EST}" \
    --storage-dtype "${DTYPE}" \
    --compressor "zstd" \
    --compression-level 9 \
    --compute-spectral-properties \
    --genome-build "GRCh37" \
    --metadata "Biobank=UK Biobank,Ancestry=${POP}" \
    --output-dir "$OUTPUTDIR"  \
    "${extra_params[@]}"

MINUTES=$(echo "scale=2; $SECONDS/60" | bc)

echo "Job finished with exit code $? at: `date`"
echo "Duration (minutes): $MINUTES"
