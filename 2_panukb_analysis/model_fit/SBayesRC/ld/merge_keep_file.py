import argparse
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Merge plink's .fam file with and keep file.")
    parser.add_argument('--fam-file', dest="fam_file", help="Path to the plink's .fam file.", required=True)
    parser.add_argument('--keep', dest="keep_file", help="Path to the keep file.", required=True)
    parser.add_argument('--output-file', dest="output_file",
                        help="Path to the output file.", required=True)

    args = parser.parse_args()

    fam_df = pd.read_csv(args.fam_file, sep=r'\s+', header=None,
                         usecols=[0, 1], names=['FID', 'IID'],
                         dtype={'FID': str, 'IID': str})

    keep_df = pd.read_csv(args.keep_file, sep=r'\s+', header=None,
                          names=['FID', 'IID'],
                          dtype={'FID': str, 'IID': str})

    merged_df = fam_df.merge(keep_df, on=['FID', 'IID'])

    merged_df.to_csv(args.output_file, sep="\t", header=False, index=False)
