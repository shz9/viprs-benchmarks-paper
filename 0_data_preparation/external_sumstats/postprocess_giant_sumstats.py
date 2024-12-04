import pandas as pd
import sys

sumstats_file = sys.argv[1]

sumstats_df = pd.read_csv(sumstats_file, sep="\t")
sumstats_df.rename(columns={
    'RSID': 'SNP',
    'OTHER_ALLELE': 'A2',
    'EFFECT_ALLELE': 'A1',
    'EFFECT_ALLELE_FREQ': 'MAF',
    'P': 'PVAL'
}, inplace=True)

sumstats_df[['CHR', 'SNP', 'POS', 'A2', 'A1', 'MAF', 'N', 'BETA', 'SE', 'PVAL']].to_csv(
    sumstats_file, sep="\t", index=False
)
