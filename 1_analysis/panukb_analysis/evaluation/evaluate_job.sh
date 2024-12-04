#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem=10GB
#SBATCH --time=01:00:00
#SBATCH --output=./log/evaluate/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# Loop over the scoring files:

sumstats_origin=${1:-"panukb_sumstats"}
pheno=${2:-"50"}
genotype_data=${3:-"ukbb"}

# Activate the virtual environment:
source "env/viprs/bin/activate"

for score_file in data/score/"${genotype_data}"/"${sumstats_origin}"/*/*/*/"${pheno}"/VIPRS_EM.prs.gz
do

  # To obtain the filename for the evaluation file, replace data/score with data/evaluation:
  eval_file=$(echo "$score_file" | sed 's/data\/score/data\/evaluation/g' | sed 's/\VIPRS_EM.prs\.gz//g')
  mkdir -p "$(dirname $eval_file)"

  for test_pop_keep_file in "data/keep_files/${genotype_data}_qc_individuals_"*.keep
  do

    echo "Testing: $test_pop_keep_file"

    # Extract population name from keep file:
    test_pop="${test_pop_keep_file##*_}"
    test_pop="${test_pop%.keep}"

    viprs_evaluate --prs-file "$score_file" \
                   --phenotype-file "data/phenotypes/${genotype_data}/${pheno}.txt" \
                   --keep "$test_pop_keep_file" \
                   --covariates-file "data/covariates/covars_${genotype_data}.txt" \
                   --output-file "${eval_file}${test_pop}"
  done

done
