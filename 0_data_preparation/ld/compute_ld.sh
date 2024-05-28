#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=2:00:00
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
KEEPFILE=${4-"data/keep_files/ukbb_qc_individuals_${POP}.keep"}  # Keep file for individuals
EXTRACTFILE=${5-"data/keep_files/hq_imputed_variants_hm3.txt"}  # Extract file for SNPs
VARIANT_SET=$(basename $EXTRACTFILE .txt)
OUTPUTDIR=${6-"data/ld/${VARIANT_SET}/${POP}"}

echo "Computing the LD matrix for:"
echo "> Chromosome: $(basename $BEDFILE)"
echo "> LD estimator: ${LD_EST}"
echo "> Population: ${POP}"
echo "> Keep file: $(basename $KEEPFILE)"
echo "> Extract file: $(basename $EXTRACTFILE)"

SECONDS=0

mkdir -p "$OUTPUTDIR"
module load plink2

magenpy_ld \
    --bfile "$BEDFILE" \
    --keep "$KEEPFILE" \
    --extract "$EXTRACTFILE" \
    --min-mac 20 \
    --backend plink \
    --estimator windowed \
    --ld-window-cm 3 \
    --storage-dtype "int8" \
    --compressor "zstd" \
    --compression-level 9 \
    --genome-build "GRCh37" \
    --metadata "Biobank=UK Biobank,Ancestry=${POP},Date=$(date +%B%Y)" \
    --output-dir "$OUTPUTDIR"

MINUTES=$(echo "scale=2; $SECONDS/60" | bc)

echo "Job finished with exit code $? at: `date`"
echo "Duration (minutes): $MINUTES"
