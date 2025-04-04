import pandas as pd
import numpy as np
import glob
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare a .ma file for building LD matrices with SBayesRC.')

    parser.add_argument('--bim-file', dest='bim_file', required=True,
                        help='Path to the .bim file containing the SNP information. Can be a wildcard expression.')
    parser.add_argument('--extract', dest='extract',
                        help='Path to a file containing the list of SNPs to extract')

    parser.add_argument('--output-file', dest='output_file', required=True,
                        help='Path to save the processed .ma file')

    args = parser.parse_args()

    # Read the SNP information from the .bim file
    snp_info = []

    for f in glob.glob(args.bim_file):
        snp_info.append(
            pd.read_csv(f,
                        sep=r'\s+',
                        names=['CHR', 'SNP', 'cM', 'POS', 'A1', 'A2'],
                        dtype={
                            'CHR': int,
                            'SNP': str,
                            'cM': np.float32,
                            'POS': np.int32,
                            'A1': str,
                            'A2': str
                        })
        )

    snp_info = pd.concat(snp_info)

    # If the extract file is provided, read it and merge with snp_info:
    if args.extract:
        extract = pd.read_csv(args.extract, sep='\t', header=None, names=['SNP'])
        snp_info = snp_info.merge(extract, on='SNP', how='inner')

    # Output the resulting .ma file:

    snp_info.to_csv(args.output_file, sep='\t', index=False)
