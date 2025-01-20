import pandas as pd
import glob
import os

h2_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1AeeADtT0U1AukliiNyiVzVRdLYPkTbruQSk38DeutU8/export?gid=1136849106&format=csv")
pheno_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1AeeADtT0U1AukliiNyiVzVRdLYPkTbruQSk38DeutU8/export?gid=1450719288&format=csv")

keep_lifestyle_continuous = ['Morning/evening person (chronotype)',
                             'Current tobacco smoking',
                             'Skin colour']

subset_df = h2_df.loc[h2_df['qcflags.pass_all'] &
                      (h2_df['N_ancestry_QC_pass'] > 2) &
                      (h2_df['pheno_sex'] == 'both_sexes')]

merged_df = pheno_df.merge(subset_df)

# Filter to only keep continuous / biomarkers / categorical variables:
merged_df = merged_df.loc[merged_df.trait_type.isin(
    ['continuous', 'biomarkers']
)]

# Filter to only keep digit codes that can be extracted directly from UKB:
merged_df = merged_df.loc[merged_df.phenocode.str.isdigit()]

# Filter to remove some of the continuous variables:
merged_df = merged_df.loc[(merged_df['trait_type'] != 'continuous') |
                          ~(merged_df['category'].fillna('').str.contains('Touchscreen > Lifestyle and environment') &
                          ~merged_df['description'].isin(keep_lifestyle_continuous))]

os.system("mkdir -p data/sumstats/panukb_sumstats")
merged_df.to_csv("data/sumstats/panukb_sumstats/subset_pheno_manifest.csv", index=False)

sumstats_files = list(merged_df['aws_link'].unique())

# Download the sumstats:
for f in sumstats_files:
    os.system("wget -P data/sumstats/panukb_sumstats/ " + f)
