#!/bin/bash
#SBATCH --account=ctb-sgravel
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=15:00:00
#SBATCH --output=./log/model_fit/panukb_sumstats/external/PRScs/EUR/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

echo "Performing model fit..."
echo "Model: PRScs"

PRSCS_PATH="2_panukb_analysis/model_fit/PRScs/"

source "$PRSCS_PATH/PRScs_env/bin/activate"
prscs_bin="$PRSCS_PATH/bin/PRScs/PRScs.py"
LD_PANEL_PATH="$PRSCS_PATH/data/ld/ldblk_ukbb_eur"

# Inputs:
sumstats_file=${1:-"data/sumstats/panukb_sumstats/EUR/50.sumstats.gz"}
phenotype=$(basename "$sumstats_file" | sed 's/\.sumstats\.gz//g')

output_prefix="data/model_fit/panukb_sumstats/external/PRScs/EUR/${phenotype}/PRScs"

mkdir -p "$(dirname "$output_prefix")" || true

run_prscs_commands() {

  mkdir -p "$SLURM_TMPDIR/transformed_sumstats/"
  mkdir -p "$SLURM_TMPDIR/fit_files/"

  export MKL_NUM_THREADS=8
  export NUMEXPR_NUM_THREADS=8
  export OMP_NUM_THREADS=8

  source "$PRSCS_PATH/PRScs_env/bin/activate"

  # Transform the summary statistics:
  python "$PRSCS_PATH/transform_sumstats.py" -s "$sumstats_file" \
         -o "$SLURM_TMPDIR/transformed_sumstats/"

  # Extract the sample size from the file outputted by transform_sumstats:
  N=$(head "$SLURM_TMPDIR/transformed_sumstats/N.txt")

  for chrom in $(seq 1 22)
  do
    python "$prscs_bin" --ref_dir "$LD_PANEL_PATH" \
           --bim_prefix "data/ukbb_qc_genotypes/chr_$chrom" \
           --sst_file "$SLURM_TMPDIR/transformed_sumstats/chr_${chrom}.prscs.ss" \
           --out_dir "$SLURM_TMPDIR/fit_files/chr_$chrom" \
           --n_gwas "$N" \
           --chrom "$chrom"
  done

}

# Export the function + required variables:
export -f run_prscs_commands
export sumstats_file
export prscs_bin
export PRSCS_PATH
export SLURM_TMPDIR
export LD_PANEL_PATH

start_time=$(date +%s)

/usr/bin/time -o "${output_prefix}.prof" \
        -v bash -c "run_prscs_commands"

end_time=$(date +%s)

echo "Transforming output of PRScs..."

# Combine the outputs of PRScs into a single file:
cat "$SLURM_TMPDIR/fit_files/chr_"*".txt" > "${output_prefix}.fit"

# and add the header: CHR\tSNP\tPOS\tA1\tA2\tBETA
sed -i '1s/^/CHR\tSNP\tPOS\tA1\tA2\tBETA\n/' "${output_prefix}.fit"

# Gzip the output file:
gzip -f "${output_prefix}.fit"

# Output the runtime:

echo -e "Total_WallClockTime" > "${output_prefix}_detailed.prof"
echo -e "$((end_time - start_time))" >> "${output_prefix}_detailed.prof"


echo "Job finished with exit code $? at: `date`"
