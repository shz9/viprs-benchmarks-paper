#!/bin/bash

# Set the URL and target directory
URL="https://zenodo.org/records/14270953/files/benchmark_sumstats.tar.gz?download=1"
TARGET_DIR="data/sumstats"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Change to the target directory
cd "$TARGET_DIR"

# Download the file
# Use wget with -O to specify output filename and remove any query parameters
wget -O "benchmark_sumstats.tar.gz" "$URL"

# Extract the tar.gz file
tar -xzvf "benchmark_sumstats.tar.gz"

# Optional: Remove the downloaded archive after extraction
# Uncomment the next line if you want to delete the archive after extracting
# rm "benchmark_sumstats.tar.gz"

echo "Download and extraction complete."
