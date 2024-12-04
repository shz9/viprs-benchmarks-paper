#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=16
#SBATCH --mem=30GB
#SBATCH --time=05:00:00
#SBATCH --output=./log/score/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL


echo "Job started at: `date`"

fit_file=${1:-"data/model_fit/panukb_sumstats/hq_imputed_variants_hm3/EUR/50/VIPRS_EM.fit.gz"}
output_file=${2:-"data/score/panukb_sumstats/hq_imputed_variants_hm3/EUR/50"}
genotype_dir=${3:-"data/ukbb_qc_genotypes/chr_*"}

source "env/viprs/bin/activate"

viprs_score -f "$fit_file" \
            --bfile "$genotype_dir" \
            --output-file "$output_file" \
            --backend "plink" \
            --compress

echo "Job finished with exit code $? at: `date`"
