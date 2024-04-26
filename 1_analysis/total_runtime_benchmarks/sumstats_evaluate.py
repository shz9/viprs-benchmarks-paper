import magenpy as mgp
from magenpy.utils.system_utils import makedir
from viprs.eval.pseudo_metrics import pseudo_r2
import pandas as pd
import os.path as osp
import glob
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the performance of a PRS model using '
                                                 'GWAS summary statistics.')

    parser.add_argument('-f', '--fit-files', dest='fit_files', type=str, required=True,
                        help='The path to the file(s) with the output parameter estimates from VIPRS. '
                             'You may use a wildcard here if fit files are stored per-chromosome (e.g. "prs/chr_*.fit")')
    parser.add_argument('--test-sumstats', dest='test_sumstats', type=str, required=True,
                        help='The path to the test GWAS summary statistics. You may use a wildcard.')
    parser.add_argument('--test-sumstats-format', dest='test_sumstats_format', type=str,
                        default='magenpy',
                        help='The format of the test GWAS summary statistics.')
    parser.add_argument('--output-file', dest='output_file', type=str, required=True,
                        help='The output file where to store the evaluation results.')

    args = parser.parse_args()

    gdl = mgp.GWADataLoader(sumstats_files=args.test_sumstats,
                            sumstats_format=args.test_sumstats_format)

    prs_betas = []

    for f in glob.glob(args.fit_files):
        prs_betas.append(pd.read_csv(f, sep="\t"))

    prs_betas = pd.concat(prs_betas)

    r2 = pseudo_r2(gdl, prs_betas)

    output_res = pd.DataFrame({'pseudo_R2': [r2]})
    makedir(osp.dirname(args.output_file))
    output_res.to_csv(args.output_file, sep="\t", index=False)
