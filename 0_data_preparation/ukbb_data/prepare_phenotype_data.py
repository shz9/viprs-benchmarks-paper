import pandas as pd
import numpy as np
from magenpy.utils.system_utils import makedir
from magenpy.stats.transforms.phenotype import detect_outliers

pheno_df = pd.read_csv("data/sumstats/panukb_sumstats/subset_pheno_manifest.csv")
unique_phenocode = list(pheno_df['phenocode'].unique())

# -----------------------------------------------------
# First of all, we need to determine which columns to load from the big tabular data:

# Read just the columns:
df = pd.read_csv("/lustre03/project/6008063/neurohub/UKB/Tabular/current.csv", nrows=0)

# Construct a dataframe with the column IDs (excluding sample IDs):
col_df = pd.DataFrame({'measurement_id': df.columns[1:]})
# Get the trait ID only (ignoring the visit/instances codes):
col_df['trait_id'] = col_df['measurement_id'].str.split('-').str[0]
# Filter to trait IDs in the unique phenocode list and drop duplicates (keeping first instance):
final_cols = col_df.loc[col_df['trait_id'].isin(
    list(map(str, unique_phenocode))
)].drop_duplicates(subset=['trait_id'])

# -----------------------------------------------------

# Read the relevant columns from the tabular data:
df = pd.read_csv("/lustre03/project/6008063/neurohub/UKB/Tabular/current.csv",
                 usecols=['eid'] + final_cols['measurement_id'].tolist())
df.columns = [col.split('-')[0] for col in df.columns]

makedir("data/phenotypes/")

# Output the data for each phenotype separately:
for pheno in df.columns[1:]:
    print(f"Processing phenotype: {pheno}")
    sub_df = df[['eid', 'eid', pheno]].copy()
    sub_df.columns = ['FID', 'IID', pheno]
    sub_df[pheno] = np.where(detect_outliers(sub_df[pheno]), np.nan, sub_df[pheno])
    sub_df.to_csv(f"data/phenotypes/{pheno}.txt", sep="\t", header=False, index=False)
