#!/bin/bash

sumstats_origin=${1:-"panukb_sumstats"}
genotype_data=${2:-"ukbb"}
# Extract unique phenotypes for which we have PRS scores:
phenotypes=($(find data/score/"$genotype_data"/"$sumstats_origin"/hq_imputed_variant* -type d -regex '.*/[0-9]+$' | sed 's/.*\///' | sort -u))

mkdir -p "./log/evaluate"

# Loop over the phenotypes:
for pheno in "${phenotypes[@]}"
do

  # Check if the phenotype file exists under data/phenotypes/${genotype_data}/${pheno}.txt
  if [ ! -f "data/phenotypes/${genotype_data}/${pheno}.txt" ]
  then
    echo "Phenotype file not found: data/phenotypes/${genotype_data}/${pheno}.txt"
    continue
  fi

  echo "Submitting job for: $pheno"

  mkdir -p "./log/evaluate/${genotype_data}/${sumstats_origin}"

  sbatch -J "${genotype_data}/${sumstats_origin}/${pheno}" \
         1_analysis/panukb_analysis/evaluation/evaluate_job.sh "$sumstats_origin" "$pheno" "$genotype_data"

done
