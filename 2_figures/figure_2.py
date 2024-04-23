import argparse
import glob
import os.path as osp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import extract_performance_statistics


model_versions = {
    'old_viprs': 'v0.0.4',
    'new_viprs': 'v0.1'
}


def extract_data_panel_b():
    """
    Plot panel B of Figure 2.
    :return: The extracted and pre-processed data for panel B
    """

    # Extract total time metrics:
    total_stats = []

    for f in glob.glob("data/benchmark_results/total/*viprs_fold_*.txt"):
        total_perf = extract_performance_statistics(f)
        total_perf['Model'] = model_versions["_".join(osp.basename(f).split('_')[:2])]
        total_stats.append(total_perf)

    total_df = pd.DataFrame(total_stats)

    pass


def extract_data_panel_c():
    """
    Plot panel C of Figure 2.
    :return: The extracted and pre-processed data for panel C
    """
    pass


def extract_data_panel_d():
    """
    Plot panel D of Figure 2.
    :return: The extracted and pre-processed data for panel D
    """
    pass


def plot_panel_a(iargs):
    """
    Plot panel A of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = pd.read_csv(osp.join(iargs.output_dir, 'figure_data', 'ld_matrices.csv'))
    # Compute normalized storage per 1m SNPs:
    df['NormalizedStorage'] = (df['TotalStorage'] / df['n_snps'])*1e6

    # Sort the dataframe by normalized storage:
    df = df.sort_values('NormalizedStorage')

    sns.barplot(data=df,
                y='LDSource',
                x='NormalizedStorage',
                color='skyblue')
    plt.xlabel("Storage (GB) per 1m variants")
    plt.ylabel("LD Matrix Resource")

    plt.savefig(osp.join(iargs.output_dir, f'panel_a.{iargs.extension}'),
                bbox_inches='tight')


def plot_panel_b(iargs):
    """
    Plot panel B of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_data_panel_b()

    # The metrics to compare between new vs. old VIPRS:
    metrics = ['Total Time (minutes)', 'Peak Memory (GB)', 'Avg Time per E-Step (ms)']

    fig, axs = plt.subplots(ncols=3, figsize=(15, 10))

    for i, metric in enumerate(metrics):
        sns.barplot(x='Model', y=metric, data=df, ax=axs[i])
        axs[i].set_title(metric)

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panel_b.{iargs.extension}'))


def plot_panel_c(iargs):
    """
    Plot panel C of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """
    pass


def plot_panel_d(iargs):
    """
    Plot panel D of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot the panels of Figure 2.')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='3_figures/figure_2/',
                        help='The directory where to store the figure panels.')
    parser.add_argument('--extension', dest='extension', type=str,
                        default='eps',
                        help='The extension for the output figure files.')
    args = parser.parse_args()

    # Set seaborn context:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=2)

    plot_panel_a(args)
    plot_panel_b(args)
    plot_panel_c(args)
    plot_panel_d(args)
