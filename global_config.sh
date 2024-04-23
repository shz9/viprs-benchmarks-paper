#!/bin/bash
# This script contains paths/general settings for running
# the analyses for this paper.

# ------------------------ Genotype Data Constants/Filters ------------------------

# Minimum/Maximum allele frequency and count:
MIN_MAC=20

# Missingness rate filters:
MIND=0.05  # Maximum missingness rate for individuals
GENO=0.05  # Maximum missingness rate for SNPs

# Hardy-Weinberg Equilibrium test cutoff:
HWE_CUTOFF=1e-10

# Hard call threshold:
HARDCALL_THRES=0.1

# ------------------------ Paths and Directories ------------------------
# The path to the viprs-paper home directory:
# Derived from the location of the config script. You may
# hardcode it here if you wish.
WORKING_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# The path to the UKBB genotype data:
# (See data_preparation/ukbb_qc_job.sh for how this path is used)
UKBB_GENOTYPE_DIR="/lustre03/project/6004777/projects/uk_biobank/imputed_data/full_UKBB/v3_bgen12"

# The path to the UKBB phenotype data:
# (See data_preparation/prepare_real_phenotypes.py for how this path is used)
UKBB_PHENOTYPE_DIR="/lustre03/project/6004777/projects/uk_biobank/raw"

# The path to the 1000G genetic map:
# (See data_preparation/ukbb_qc_job.sh for how this path is used)
GENETIC_MAP_DIR="$HOME/projects/ctb-sgravel/data/genetic_maps/1000GP_Phase3"
