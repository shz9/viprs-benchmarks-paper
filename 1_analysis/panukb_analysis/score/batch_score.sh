#!/bin/bash

ignore_already_scored=${1:-"true"}
genotype_data=${2:-"ukbb"}

for fit_file in data/model_fit/*_sumstat*/hq_imputed_variant*/*/*/*/VIPRS_EM.fit.gz
do

  # To obtain the score_file, replace data/model_fit/ with data/score/${genotype_data}/ and replace .fit.gz with nothing
  score_file=$(echo "$fit_file" | sed "s|data/model_fit|data/score/${genotype_data}|g" | sed 's/\.fit\.gz//g')

  # Check if the score file exists and the ignore_already_scored flag is set to true:
  if [ -f "${score_file}.prs.gz" ] && [ "$ignore_already_scored" == "true" ]
  then
    # echo "Score file already exists: $score_file"
    continue
  fi

  # For the logging file path, take the directory name of the scoring file and replace data/score with log/score:
  log_path=$(dirname $score_file | sed 's/data\/score/log\/score/g')

  # For the job name, replace log/score with nothing:
  job_name=$(echo $log_path | sed 's/log\/score//g')

  mkdir -p "log_path"

  echo "Scoring $fit_file"

  sbatch -J "${job_name}" \
         1_analysis/panukb_analysis/score/score_job.sh "$fit_file" \
         "$score_file" \
         "data/${genotype_data}_qc_genotypes/chr_*"

done
