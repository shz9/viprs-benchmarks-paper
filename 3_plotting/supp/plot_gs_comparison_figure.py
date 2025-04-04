import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from magenpy.utils.system_utils import makedir
import glob
import magenpy as mgp
import numpy as np
import pandas as pd
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import *


def main(iargs):

    prof_metrics = extract_aggregate_performance_metrics(model='*VIPRS_GS',
                                                         variant_set='hq_imputed_variants_hm3')
    prof_metrics['Model'] = prof_metrics['Model'].map({
        'VIPRS_GS': 'VIPRS-GS',
        'pathwise_VIPRS_GS': 'Pathwise VIPRS-GS'
    })
    prof_metrics['Total_FitTime'] /= 60


    eval_df = extract_aggregate_evaluation_metrics(ld_estimator='block_int8_mi',
                                                   variant_set='hq_imputed_variants_hm3',
                                                   model='*VIPRS_GS')
    eval_df['Test cohort'] = eval_df['Test_cohort'].map(
            {'ukbb': 'UKB', 'cartagene': 'CARTaGENE'}
    ) + '-' + eval_df['Test_pop']

    eval_df = eval_df.loc[~eval_df['Test cohort'].isin(['UKB-EUR', 'UKB-all'])]

    eval_df = pivot_evaluation_df(eval_df, columns=['Model'])

    # Plot the figure:
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))

    # Plot the first panel with accuracy metrics:
    sns.scatterplot(data=eval_df, x='Incremental_R2_VIPRS_GS',
                    y='Incremental_R2_pathwise_VIPRS_GS', ax=ax[0])
    # Add the diagonal line:
    ax[0].plot([0, 0.3], [0, 0.3], color='grey', linestyle='--')
    ax[0].set_xlabel('Incremental $R^2$ (VIPRS-GS)')
    ax[0].set_ylabel('Incremental $R^2$ (Pathwise VIPRS-GS)')
    ax[0].set_title("(a) Accuracy comparison")

    # Plot the second panel with the performance metrics:
    sns.barplot(data=prof_metrics, x='Model', y='Total_FitTime', hue='Model', ax=ax[1])
    ax[1].set_ylabel('Total inference time (minutes)')
    ax[1].set_xlabel('Model')
    ax[1].set_title("(b) Runtime comparison")

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'VIPRS_GS_comparison.{iargs.extension}'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot grid search comparison figures')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='figures/supp_figures/',
                        help='The directory where to store the figure panels.')
    parser.add_argument('--extension', dest='extension', type=str,
                        default='eps',
                        help='The extension for the output figure files.')
    args = parser.parse_args()

    # Create the output directory if it does not exist:
    makedir(args.output_dir)
    makedir(osp.join(args.output_dir, 'figure_data'))

    sns.set_context("paper", font_scale=1.5)

    main(args)
