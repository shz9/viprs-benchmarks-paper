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
output_file=${2:-"data/score/cartagene/panukb_sumstats/hq_imputed_variants_hm3/EUR/50"}
genotype_dir=${3:-"$HOME/projects/ctb-sgravel/cartagene/research/flagship_project/processed_data/flagship_array_imputed_w_cag_and_topmed_r3/PGEN"}

source "env/viprs/bin/activate"
module load plink

# Create a temporary directory to store updated score files by joining SLURM_TMPDIR with the job ID:
tmp_dir="$SLURM_TMPDIR/$SLURM_JOBID"
mkdir -p "$tmp_dir/updated_fit_files/"
mkdir -p "$tmp_dir/score_files/"

# Update the fit file variant IDs to match the .pgen file IDs:
python 2_panukb_analysis/score/cartagene_score/map_fit_file_ids.py --fit-file "$fit_file" \
                    --output-dir "$tmp_dir/updated_fit_files/"

echo "Updated fit files created at: $tmp_dir/updated_fit_files/"

# Loop over chromosomes and perform linear scoring on .pgen files using plink2:

for chrom in {1..22}
do
    echo "> Scoring chromosome $chrom..."
    plink2 --pgen "$genotype_dir/chr${chrom}.imputed_metaminimac_w_CaG_TOPMed_r3.pgen" \
           --pvar "$genotype_dir/chr${chrom}.imputed_metaminimac_w_CaG_TOPMed_r3.pvar" \
           --psam "$genotype_dir/chr${chrom}.imputed_metaminimac_w_CaG_TOPMed_r3.psam" \
           --score "$tmp_dir/updated_fit_files/chr${chrom}.txt" 1 2 header-read cols=+scoresums \
           --score-col-nums 3 \
           --out "$tmp_dir/score_files/chr_${chrom}"
done

echo "Scoring done; Score files per chromosome are at: $tmp_dir/score_files/"

# Create the output directory for the output file:
mkdir -p "$(dirname $output_file)"

# Concatenate the score files:
python 2_panukb_analysis/score/combine_score_files.py --input-dir "$tmp_dir/score_files/" \
                    --output-file "$output_file"


echo "Job finished with exit code $? at: `date`"
