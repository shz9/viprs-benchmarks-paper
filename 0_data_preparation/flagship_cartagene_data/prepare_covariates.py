import pandas as pd
import os

covar_df = pd.read_csv("~/projects/ctb-sgravel/cartagene/research/flagship_project/processed_data/"
                       "flagship_GWAS_covariates/GWAS_EUR_covariates.tsv", sep="\t")
covar_df = covar_df.drop(columns=['Age2', 'Array']).dropna()
covar_df.to_csv("data/covariates/covars_cartagene.txt", sep="\t", index=False, header=False, na_rep='NA')

# =============================================================================
# Copy the file defining European samples:

os.system("cp ~/projects/ctb-sgravel/cartagene/research/flagship_project/processed_data/"
          "association_testing/European_Flagship_cluster_7-14.tsv data/keep_files/cartagene_qc_individuals_EUR.keep")

