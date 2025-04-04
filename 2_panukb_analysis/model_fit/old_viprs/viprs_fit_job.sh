#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=01:00:00
#SBATCH --output=./log/model_fit/panukb_sumstats/external/VIPRS_v0.0.4/EUR/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL


echo "Job started at: `date`"

sumstats_file=${1:-"data/sumstats/panukb_sumstats/EUR/50.sumstats.gz"}
phenotype=$(basename "$sumstats_file" | sed 's/\.sumstats\.gz//g')

ld_dir=${2:-"data/ld/eur/old_format/ukbb_50k_windowed/chr_*"}
output_prefix=${3:-"data/model_fit/panukb_sumstats/external/VIPRS_v0.0.4/EUR/$phenotype/VIPRS_v0.0.4"}

mkdir -p "$(dirname "$output_prefix")" || true

source "env/viprs-old/bin/activate"

start_time=$(date +%s)

/usr/bin/time -o "${output_prefix}.prof" -v viprs_fit -l "$ld_dir" \
         -s "$sumstats_file" \
         --sumstats-format "magenpy" \
         --output-file "$output_prefix" \
         --compress

end_time=$(date +%s)

echo -e "Total_WallClockTime" > "${output_prefix}_detailed.prof"
echo -e "$((end_time - start_time))" >> "${output_prefix}_detailed.prof"

echo "Job finished with exit code $? at: `date`"
