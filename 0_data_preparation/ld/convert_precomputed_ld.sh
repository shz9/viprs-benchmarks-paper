#!/bin/bash

# Convert the LD matrices using the provided script.

# First, load the python environment where the newer version of magenpy is installed:

source env/viprs/bin/activate

# Take the data type as an argument (with default set to int8):
data_type=("int8" "int16" "float32" "float64")

for dtype in "${data_type[@]}"
do
  for chrom in {1..22}
  do
    echo "Converting LD matrix for chromosome: $chrom..."
    python3 0_data_preparation/convert_old_ld_matrices.py \
      --old-matrix-path "data/ld/eur/old_format/ukbb_50k_windowed/chr_$chrom/" \
      --new-path "data/ld/eur/converted/ukbb_50k_windowed/$dtype/chr_$chrom/" \
      --dtype "$dtype"
  done
done

deactivate
