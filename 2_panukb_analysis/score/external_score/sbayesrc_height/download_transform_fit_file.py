import pandas as pd
import numpy as np
from magenpy.utils.system_utils import makedir
from magenpy.parsers.plink_parsers import parse_bim_file

bim_df = pd.concat([parse_bim_file("data/ukbb_qc_genotypes/chr_{}.bim".format(chrom)) for chrom in range(1, 23)])

fit_df = pd.read_csv("https://sbayes.pctgplots.cloud.edu.au/data/SBayesRC/share/v1.0/PGS/HT.txt.gz",
                     sep="\t")
fit_df.rename(columns={'beta': 'BETA'}, inplace=True)

matched_variants = fit_df.merge(bim_df, on='SNP')
matched_variants['matched_a1'] = matched_variants['A1_x'] == matched_variants['A1_y']
matched_variants['flipped_a1'] = matched_variants['A1_x'] == matched_variants['A2']
matched_variants = matched_variants.loc[matched_variants['matched_a1'] | matched_variants['matched_a1']]

matched_variants.rename(columns={'A1_x': 'A1'}, inplace=True)
# Extract the reference allele information:
matched_variants['A2_new'] = np.where(matched_variants['flipped_a1'], matched_variants['A1_y'], matched_variants['A2'])
matched_variants.drop(columns=['A2'], inplace=True)
matched_variants.rename(columns={'A2_new': 'A2'}, inplace=True)
matched_variants['PIP'] = np.nan
matched_variants['VAR_BETA'] = np.nan

makedir("data/model_fit/external/SBayesRC/EUR/50/")

# Output the file:
matched_variants[['CHR', 'SNP', 'POS', 'A1', 'A2', 'BETA', 'PIP', 'VAR_BETA']].to_csv(
    "data/model_fit/external/SBayesRC/EUR/50/SBayesRC_7m.fit.gz", sep="\t", index=False
)
