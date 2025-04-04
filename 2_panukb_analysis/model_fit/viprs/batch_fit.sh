#!/bin/bash

sumstats_to_fit=${1:-"panukb_sumstats"}
variantset_to_fit=${2:-"hq_imputed_variants_hm3"}
ld_est=${3:-"block"}
ld_dtype=${4:-"int16"}
hyp_search=${5:-"EM"}
grid_search_mode=${6:-"pathwise"}

declare -A time_dict
time_dict["hq_imputed_variants_hm3_VIPRS_EM"]="01:00:00"
time_dict["hq_imputed_variants_hm3_VIPRS_GS"]="01:00:00"
time_dict["hq_imputed_variants_hm3_VIPRS_GS_pathwise"]="01:00:00"

time_dict["hq_imputed_variants_maf001_VIPRS_EM"]="03:00:00"
time_dict["hq_imputed_variants_maf001_VIPRS_GS"]="09:00:00"
time_dict["hq_imputed_variants_maf001_VIPRS_GS_pathwise"]="07:00:00"

time_dict["hq_imputed_variants_VIPRS_EM"]="05:00:00"
time_dict["hq_imputed_variants_VIPRS_GS"]="15:00:00"
time_dict["hq_imputed_variants_VIPRS_GS_pathwise"]="12:00:00"

declare -A threads_dict
threads_dict["hq_imputed_variants_hm3"]=4
threads_dict["hq_imputed_variants_maf001"]=8
threads_dict["hq_imputed_variants"]=8

shopt -s nullglob

# Loop over the summary statistics files:
for sumstats_file in data/sumstats/"$sumstats_to_fit"/EUR/*.sumstats.gz
do
  # Extract the population name from the file path:
  pop=$(basename "$(dirname "$sumstats_file")")
  # Extract the phenotype code from the file name (exclude .sumstats.gz):
  pheno=$(basename "$sumstats_file" .sumstats.gz)
  # Extract the sumstats origin from the file path:
  sumstats_origin=$(basename "$(dirname "$(dirname "$sumstats_file")")")

  # Loop over the available LD matrices for this population:
  for ld_dir in data/ld_x*/"$variantset_to_fit"/"$pop"/"${ld_est}"/"${ld_dtype}"
  do
    # Extract the variant set name from the file path (hq_imputed_variants...):
    variant_set=$(basename "$(dirname "$(dirname "$(dirname "$ld_dir")")")")

    # Extract LD matrix characteristics:
    # From the parent directory, extract the LD matrix type (windowed, block):
    ld_dtype=$(basename "$ld_dir")

    # Extract the LD estimator type (windowed, block):
    ld_estimator=$(basename "$(dirname "$ld_dir")")

    # If ld_xarray is present in ld_dir, add _mi, otherwise, add nothing:
    ld_est_data=$(echo "$ld_dir" | grep -q "ld_xarray" && echo "_mi" || echo "")

    ld_info="${ld_estimator}_${ld_dtype}${ld_est_data}"

    mkdir -p "./log/model_fit/${sumstats_origin}/${variant_set}/${ld_info}/${pop}/"

    log_suffix="VIPRS_${hyp_search}"

    if [ "$hyp_search" == "GS" ] && [ "$grid_search_mode" == "pathwise" ]; then
      log_suffix="${log_suffix}_pathwise"
    fi

    # If the LD type is int16 and the variant set is hq_imputed_variants, then set memory usage to
    # 36GB, otherwise set it to 30GB:
    if [ "$ld_dtype" == "int16" ] && [ "$variant_set" == "hq_imputed_variants" ]; then
      mem="36GB"
    else
      mem="30GB"
    fi

    echo "Submitting job for: $sumstats_origin | $variant_set | $pop | $pheno"
    echo "Using ${threads_dict[$variant_set]} threads and time limit: ${time_dict[${variant_set}_${log_suffix}]}"

    sbatch --time "${time_dict[${variant_set}_${log_suffix}]}" --mem "$mem" \
            -J "${sumstats_origin}/${variant_set}/${ld_info}/${pop}/${pheno}${log_suffix}" \
            2_panukb_analysis/model_fit/viprs/viprs_fit_job.sh \
                    "$sumstats_file" \
                    "$ld_dir/chr_*" \
                    "data/model_fit/${sumstats_origin}/${variant_set}/${ld_info}/${pop}/${pheno}/" \
                    "$hyp_search" \
                    "$grid_search_mode" \
                    "${threads_dict[$variant_set]}"

  done

done