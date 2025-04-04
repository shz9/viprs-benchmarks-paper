import pandas as pd
import argparse
import os.path as osp
import numpy as np
from scipy import stats


parser = argparse.ArgumentParser(description='Transform summary statistics to COJO format')

parser.add_argument('-s', '--sumstats', dest='ss_file', type=str, required=True,
                    help='The summary statistics files')
parser.add_argument('--output-dir', dest='output_dir', type=str, required=False,
                    help='The output directory')
args = parser.parse_args()

print(f"> Transforming summary statistics file: {args.ss_file}")
# Read the sumstats file:
ss_df = pd.read_csv(args.ss_file, sep="\t")
# Compute p-value:
ss_df['p'] = 2.*stats.norm.sf(np.abs(ss_df['BETA']/ss_df['SE']))
ss_df = ss_df[['SNP', 'A1', 'A2', 'MAF', 'BETA', 'SE', 'p', 'N']]

# Write the results to the same directory:
ss_df.columns = ['SNP', 'A1', 'A2', 'freq', 'b', 'se', 'p', 'N']
ss_df['N'] = ss_df['N'].astype(int)

if args.output_dir is None:
    new_f_name = args.ss_file.replace(".sumstats.gz", ".ma")
else:
    new_f_name = osp.join(args.output_dir, osp.basename(args.ss_file).replace(".sumstats.gz", ".ma"))

ss_df.to_csv(new_f_name, sep="\t", index=False)
