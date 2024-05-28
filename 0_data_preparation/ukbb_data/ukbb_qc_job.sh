#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=08:00:00
#SBATCH --output=./log/data_preparation/genotypes/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

. global_config.sh

module load plink

cd "$WORKING_DIR" || exit

CHR=${1:-22}  # Chromosome number (default 22)
snp_extract_file=${2-"data/keep_files/hq_imputed_variants.txt"}
ind_keep_file=${3-"data/keep_files/ukbb_qc_individuals_all.keep"}
output_dir=${4-"data/ukbb_qc_genotypes"}

mkdir -p "$output_dir"

plink2 --bgen "$UKBB_GENOTYPE_DIR/ukb22828_c${CHR}_b0_v3.bgen" ref-first \
      --sample "$UKBB_GENOTYPE_DIR/ukb22828_c${CHR}_b0_v3.sample" \
      --make-bed \
      --allow-no-sex \
      --keep "$ind_keep_file" \
      --extract "$snp_extract_file" \
      --mac "$MIN_MAC" \
      --max-alleles 2 \
      --hard-call-threshold "$HARDCALL_THRES" \
      --out "$output_dir/chr_${CHR}"

module load nixpkgs/16.09
module load plink/1.9b_4.1-x86_64
# Update the SNP cM position using the HapMap3 genetic map:
plink --bfile "$output_dir/chr_${CHR}" \
      --cm-map "$GENETIC_MAP_DIR/genetic_map_chr@_combined_b37.txt" \
      --make-just-bim \
      --out "$output_dir/chr_${CHR}"

rm -r "$output_dir"/*~

echo "Job finished with exit code $? at: `date`"
