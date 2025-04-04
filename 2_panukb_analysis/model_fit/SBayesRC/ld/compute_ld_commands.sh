#!/bin/bash

mkdir -p ./log/data_preparation/sbayesr_ld/

# Compute LD matrices for European samples using default LDetect boundaries
sbatch 2_panukb_analysis/model_fit/SBayesRC/ld/compute_ld.sh

# Compute LD matrices for European samples using fused LDetect boundaries (min. 4cM)
sbatch 2_panukb_analysis/model_fit/SBayesRC/ld/compute_ld.sh \
        "data/keep_files/hq_imputed_variants_hm3.txt" \
        "data/ldetect_data/EUR_blocks_4cM.bed" \
        "2_panukb_analysis/model_fit/SBayesRC/data/ld/hapmap3_block_4cM"

sbatch 2_panukb_analysis/model_fit/SBayesRC/ld/compute_ld.sh \
        "data/keep_files/hq_imputed_variants_hm3_sbayes_intersect.txt" \
        "data/ldetect_data/EUR_blocks_4cM.bed" \
        "2_panukb_analysis/model_fit/SBayesRC/data/ld/hapmap3x_block_4cM"