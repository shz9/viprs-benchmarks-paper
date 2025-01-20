import pandas as pd
import os.path as osp
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combine per-chromosome score files')
    parser.add_argument('--input-dir', type=str, dest='input_dir', required=True)
    parser.add_argument('--output-file', type=str, dest='output_file', required=True)

    args = parser.parse_args()

    score_df = None

    for chrom in range(1, 23):

        chrom_df = pd.read_csv(osp.join(args.input_dir, f'chr_{chrom}.sscore'),
                               sep=r'\s+',
                               names=['FID', 'IID', 'PRS'],
                               skiprows=1,
                               usecols=[0, 1, 5])

        # If score_df is None, assign chrom_df to it.
        # Otherwise, merge them based on FID/IID and add the PRS columns.
        if score_df is None:
            score_df = chrom_df
        else:
            score_df = score_df.merge(chrom_df, on=['FID', 'IID'], suffixes=('', f'_{chrom}'))
            score_df['PRS'] += score_df[f'PRS_{chrom}'].values
            score_df.drop(columns=[f'PRS_{chrom}'], inplace=True)

    score_df.to_csv(args.output_file + ".prs.gz", index=False, sep="\t")
