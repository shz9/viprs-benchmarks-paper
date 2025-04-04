import pandas as pd
import numpy as np
import argparse
from magenpy.utils.model_utils import merge_snp_tables

parser = argparse.ArgumentParser(description='Transform output of SBayesRC')

parser.add_argument('-o', '--output', dest='output_file', type=str, required=True,
                    help='The path to the SBayesRC output files reside')
parser.add_argument('--ld-snp-info', dest='ld_snp_info', type=str, required=True,
                    help='The LD SNP info file')
args = parser.parse_args()

# Read the SNP effects file:
snp_effect_df = pd.read_csv(args.output_file, sep="\t")
snp_effect_df.drop(columns=['BETAlast', 'SE'], inplace=True)

# Read the LD SNP info file:
ld_snp_info = pd.read_csv(args.ld_snp_info, sep="\t")
merged_tab = snp_effect_df.merge(ld_snp_info[['Chrom', 'ID', 'PhysPos', 'A1', 'A2']],
                                 left_on=['SNP', 'A1'], right_on=['ID', 'A1'])
merged_tab.rename(columns={'Chrom': 'CHR', 'PhysPos': 'POS'}, inplace=True)

merged_tab = merged_tab[['CHR', 'SNP', 'POS', 'A1', 'A2', 'BETA', 'PIP']]
merged_tab.to_csv(args.output_file.replace('.txt', '.fit.gz'), sep="\t", index=False)
