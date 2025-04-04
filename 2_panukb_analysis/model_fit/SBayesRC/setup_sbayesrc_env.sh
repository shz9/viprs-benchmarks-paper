#!/bin/bash

SBAYESRC_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading required input data for SBayesRC..."

mkdir -p "$SBAYESRC_PATH/data/ld/" || true

# Download the HapMap3 LD data:
wget https://sbayes.pctgplots.cloud.edu.au/data/SBayesRC/resources/v2.0/LD/HapMap3/ukbEUR_HM3.zip -O "$SBAYESRC_PATH/data/ld/ukbEUR_HM3.zip"
mkdir -p "$SBAYESRC_PATH/data/ld/hapmap3/" || true
unzip "$SBAYESRC_PATH/data/ld/ukbEUR_HM3.zip" -d "$SBAYESRC_PATH/data/ld/hapmap3/"

# Download the imputed LD data:
wget https://sbayes.pctgplots.cloud.edu.au/data/SBayesRC/resources/v2.0/LD/Imputed/ukbEUR_Imputed.zip -O "$SBAYESRC_PATH/data/ld/ukbEUR_Imputed.zip"
mkdir -p "$SBAYESRC_PATH/data/ld/imputed/" || true
unzip "$SBAYESRC_PATH/data/ld/ukbEUR_Imputed.zip" -d "$SBAYESRC_PATH/data/ld/imputed/"

# ---------------------------------------------------------------------

echo "Setting up the environment for SBayesRC..."

# Setup the R environment:
# module load gcc/12.3 r/4.3.1
# mkdir -p "$SBAYESRC_PATH/R_sbayesrc_env" || true

# export R_LIBS="$SBAYESRC_PATH/R_sbayesrc_env"

# R -e 'install.packages(c("Rcpp", "data.table", "stringi", "BH",  "RcppEigen"), repos="https://cloud.r-project.org/")'
# export CXXFLAGS="-std=c++14"
# R -e 'install.packages("https://github.com/zhilizheng/SBayesRC/releases/download/v0.2.6/SBayesRC_0.2.6.tar.gz", repos=NULL, type="source")'

module load apptainer
mkdir -p "$SBAYESRC_PATH/sbayesrc_env" || true
apptainer pull --dir "$SBAYESRC_PATH/sbayesrc_env" docker://zhiliz/sbayesrc

echo "Done!"