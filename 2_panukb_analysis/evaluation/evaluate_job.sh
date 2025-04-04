#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem=12GB
#SBATCH --time=02:00:00
#SBATCH --output=./log/evaluate/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# Loop over the scoring files:

scoring_dir=${1:-"ukbb/panukb_sumstats"}
pheno_file=${2:-"data/phenotypes/ukbb/50.txt"}

# Extract the phenotype name from the phenotype file:
pheno=$(basename "$pheno_file" | sed 's/\.txt//g')
# Extract the cohort name from the phenotype file parent directory:
cohort=$(basename "$(dirname "$pheno_file")")

# Activate the virtual environment:
source "env/viprs/bin/activate"

# Find all files in all subdirectories of the scoring directory that end with .prs.gz:
find "data/score/${scoring_dir}" -type f -path "*/${pheno}/*.prs.gz" | while read score_file
do

  # To obtain the filename for the evaluation file, replace data/score with data/evaluation:
  eval_file=$(echo "$score_file" | sed 's/data\/score/data\/evaluation/g' | sed 's/\.prs\.gz//g')
  mkdir -p "$eval_file"

  for test_pop_keep_file in "data/keep_files/${cohort}_qc_individuals_"*.keep
  do

    echo "Testing: $test_pop_keep_file"

    # Extract population name from keep file:
    test_pop="${test_pop_keep_file##*_}"
    test_pop="${test_pop%.keep}"

    viprs_evaluate --prs-file "$score_file" \
                   --phenotype-file "$pheno_file" \
                   --keep "$test_pop_keep_file" \
                   --covariates-file "data/covariates/covars_${cohort}.txt" \
                   --output-file "${eval_file}/${test_pop}"
  done

done
