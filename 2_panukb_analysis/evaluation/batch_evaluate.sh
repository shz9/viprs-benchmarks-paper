#!/bin/bash

scoring_dir=${1:-"ukbb/panukb_sumstats/"}
phenotype_dir=${2:-"ukbb"}
# Extract unique phenotypes for which we have PRS scores:
phenotypes=($(find "data/score/$scoring_dir" -type d -regextype posix-extended -regex '.*/[0-9]+$' | sed 's/.*\///' | sort -u))

mkdir -p "./log/evaluate"

# Loop over the phenotypes:
for pheno in "${phenotypes[@]}"
do

  # Check if the phenotype file exists under data/phenotypes/${phenotype_dir}/${pheno}.txt
  if [ ! -f "data/phenotypes/${phenotype_dir}/${pheno}.txt" ]
  then
    echo "Phenotype file not found: data/phenotypes/${phenotype_dir}/${pheno}.txt"
    continue
  fi

  echo "Submitting job for: $pheno"

  mkdir -p "./log/evaluate/${scoring_dir}/${phenotype_dir}"

  sbatch -J "${scoring_dir}/${phenotype_dir}/${pheno}" \
         2_panukb_analysis/evaluation/evaluate_job.sh \
                          "$scoring_dir" \
                          "data/phenotypes/${phenotype_dir}/${pheno}.txt"

done
