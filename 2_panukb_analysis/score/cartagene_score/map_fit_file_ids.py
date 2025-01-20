import pandas as pd
import os.path as osp
import argparse
from magenpy.utils.model_utils import merge_snp_tables


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Map fit file variant rsIDs to cartagene variant IDs')
    parser.add_argument('--fit-file', type=str, dest='fit_file',
                        help='Path to the fit file',
                        required=True)
    parser.add_argument('--rsid-map', type=str, dest='rsid_map',
                        default='metadata/rsid_map_cartagene.csv.gz')
    parser.add_argument('--output-dir', type=str, dest='output_dir', required=True)

    args = parser.parse_args()

    # Load the rsID map:
    rsid_map = pd.read_csv(args.rsid_map)
    fit_df = pd.read_csv(args.fit_file, sep='\t')

    # Merge the two tables:
    merged_df = merge_snp_tables(rsid_map, fit_df, return_ref_indices=True)
    # Add the CARTaGENE ID back to the merged table:
    merged_df['ID'] = rsid_map.iloc[merged_df.REF_IDX.values]['ID'].values

    # Form the final table:
    merged_df = merged_df[['CHR', 'ID', 'A1', 'BETA']]
    merged_df.rename(columns={'ID': 'SNP'}, inplace=True)

    # Split the merged table by chromosome and save:
    for chrom in range(1, 23):
        chrom_df = merged_df.loc[merged_df['CHR'] == chrom, ['SNP', 'A1', 'BETA']]
        chrom_df.to_csv(osp.join(args.output_dir, f'chr{chrom}.txt'), index=False, sep="\t")
