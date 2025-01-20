#!/bin/bash

fit_files_dir=${1:-"panukb_sumstats"}
genotype_data=${2:-"ukbb"}
ignore_already_scored=${3:-"true"}

find "data/model_fit/${fit_files_dir}" -type f -path "*.fit.gz" | while read fit_file
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

  # Create the parent directory of the log file:
  mkdir -p "$(dirname $log_path)"

  echo "Scoring $fit_file"

  sbatch -J "${job_name}" \
         2_panukb_analysis/score/score_job.sh "$fit_file" \
         "$score_file" \
         "data/${genotype_data}_qc_genotypes/chr_*"

done
