import pandas as pd
import sys

sumstats_file = sys.argv[1]

sumstats_df = pd.read_csv(sumstats_file, sep="\t")
sumstats_df.rename(columns={
    'rsID': 'SNP',
    'CHROM': 'CHR',
    'POS_b37': 'POS',
    'REF': 'A2',
    'ALT': 'A1',
    'POOLED_ALT_AF': 'MAF',
    'EFFECT_SIZE': 'BETA',
    'pvalue': 'PVAL'
}, inplace=True)

sumstats_df[['CHR', 'SNP', 'POS', 'A2', 'A1', 'MAF', 'N', 'BETA', 'SE', 'PVAL']].to_csv(
    sumstats_file, sep="\t", index=False
)
