import pandas as pd
import os.path as osp
import argparse


parser = argparse.ArgumentParser(description='Transform summary statistics to PRScs format')

parser.add_argument('-s', '--sumstats', dest='ss_file', type=str, required=True,
                    help='The summary statistics files')
parser.add_argument('-o', '--output', dest='output_dir', type=str, required=True,
                    help='The directory where the PRScs output files reside')
args = parser.parse_args()

print(f"> Transforming summary statistics file: {args.ss_file}")
# Read the sumstats file:
ss_df = pd.read_csv(args.ss_file, sep="\t")

for chrom in ss_df['CHR'].unique():
    ss_df_chrom = ss_df[ss_df['CHR'] == chrom]
    ss_df_chrom = ss_df_chrom[['SNP', 'A1', 'A2', 'BETA', 'SE']]
    ss_df_chrom.to_csv(osp.join(args.output_dir, f'chr_{chrom}.prscs.ss'), sep="\t", index=False)

n = int(ss_df['N'].max())

# Write the sample size to file:
with open(osp.join(args.output_dir, 'N.txt'), 'w') as f:
    f.write(str(n))

