import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from magenpy.utils.system_utils import makedir
import sys
import os.path as osp
import pandas as pd
import numpy as np
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import *


def plot_relative_improvement(iargs):
    """
    Plot panel C of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_relative_improvement_data()

    # Save the figure data:
    df.to_csv(osp.join(iargs.output_dir, 'figure_data', 'e_step_improvements.csv'), index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Change', y='Improvement', data=df, color='salmon')

    # Change the angle for the x-labels to 45 degrees:
    plt.xticks(rotation=45, ha='right', size='small')

    plt.title("Relative runtime-per-iteration (v0.0.4 / v0.1)\n(Chromosome 1; 92206 variants)")
    plt.ylabel("Relative runtime-per-iteration\n(Ratio of medians)", size='small')
    plt.xlabel("Incremental changes (left -> right)")

    plt.savefig(osp.join(iargs.output_dir, f'relative_improvement_e_step.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_accuracy_by_ld_mode(iargs):
    """
    Plot panel D of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_accuracy_metrics(ld_datatype=None, ld_mode=None, threads=None)

    # Save the figure data:
    df.to_csv(osp.join(iargs.output_dir, 'figure_data', 'accuracy_by_ld_mode.csv'), index=False)

    # Create two sub-panels (2 columns, 1 row) with the left sub-panel showing
    # symmetric LD mode and the right sub-panel showing triangular LD mode:

    fig, axs = plt.subplots(ncols=2, figsize=(15, 6), sharey=True,
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    # First show the accuracy of v0.0.4 as its own bar:
    sns.barplot(x='Model', y='R-Squared', data=df.loc[df.Model == 'v0.0.4'], color='skyblue', ax=axs[0])

    # Then show the accuracy of v0.1 as a function of threads:
    sns.barplot(x='LD Data Type', y='R-Squared',
                data=df.loc[(df.Model == 'v0.1') & (df['LD Mode'] == 'Symmetric LD')],
                order=['float64', 'float32', 'int16', 'int8'],
                hue='Threads', ax=axs[0])

    # Then show the data for triangular LD:
    sns.barplot(x='LD Data Type', y='R-Squared',
                data=df.loc[(df.Model == 'v0.1') & (df['LD Mode'] == 'Triangular LD')],
                order=['float64', 'float32', 'int16', 'int8'],
                hue='Threads', ax=axs[1])

    # Change the angle for the x-labels to 45 degrees:
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45)

    axs[0].set_title("Symmetric LD")
    axs[1].set_title("Triangular LD")

    axs[0].set_xlabel("Model / LD Data Type")
    axs[1].set_xlabel("LD Data Type")

    axs[0].set_ylabel("Prediction R-Squared")

    plt.savefig(osp.join(iargs.output_dir, f'accuracy_by_ld_mode.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_e_step_runtime_all_models(iargs):
    """
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_e_step_stats(model=None, threads=None, aggregate=False)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 6), sharey=True,
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    for i, model in enumerate(df.Model.unique()):

        df_model = df.loc[df.Model == model]

        df_model_old = df_model.loc[df_model['ModelVersion'] == 'v0.0.4']
        df_model_old = df_model_old.groupby(['Model', 'ModelVersion', 'Chromosome', 'n_snps', 'Threads']).agg(
            {'TimePerIteration': 'mean'}
        ).reset_index()
        df_model_old['Threads'] = 'v0.0.4'

        sns.lineplot(data=df_model_old,
                     x='n_snps', y='TimePerIteration',
                     hue='Threads',
                     linewidth=3,
                     marker='o',
                     palette={'v0.0.4': 'skyblue'},
                     markersize=7,
                     ax=axs[i])

        df_model_new = df_model.loc[(df_model['ModelVersion'] == 'v0.1')]
        df_model_new = df_model_new.groupby(['Model', 'ModelVersion', 'Chromosome', 'n_snps', 'Threads']).agg(
            {'TimePerIteration': 'mean'}
        ).reset_index()

        sns.lineplot(data=df_model_new,
                     x='n_snps', y='TimePerIteration', hue='Threads',
                     linewidth=3,
                     marker='o',
                     markersize=7,
                     ax=axs[i])
        if i == 0:
            axs[i].set_ylabel("Time per Iteration (s)")
        else:
            axs[i].set_ylabel(None)

        if i == 1:
            axs[i].set_xlabel("Variants per Chromosome")
        else:
            axs[i].set_xlabel(None)

        if model == 'VIPRSGrid':
            model = 'VIPRSGrid(J=100)'
        axs[i].set_title(model, pad=10)
        axs[i].set_yscale('log')

    # Create a shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Version / Threads', loc='center right', bbox_to_anchor=(1.08, 0.5))

    # Remove individual legends from each subplot
    for ax in axs:
        ax.legend().set_visible(False)

    plt.savefig(osp.join(iargs.output_dir, f'runtime_per_iteration_all_models.{iargs.extension}'),
                bbox_inches="tight")
    plt.close()


def plot_speed_metrics_intel_cluster(iargs):
    """
    Raw data is provided by experiments from Chirayu Anant Haryan.
    :param iargs:
    """

    sns.set_context("paper", font_scale=1.5)

    df = pd.DataFrame(
        np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                  [1.830035, 1.771177, 1.782653, 1.806345, 1.705882353, 1.309524, 1.257143, 1.727273],
                  [3.398605, 3.320087, 2.979073, 3.318547, 2.843137255, 2.291667, 1.173333, 2.375],
                  [5.402454, 5.726396, 4.939991, 6.024378, 4.53125, 3.666667, 3.52, 1.117647],
                  [8.053627, 8.036762, 7.935028, 8.255128, 6.444444444, 5, 1.795918, 1.055556],
                  [9.513928, 9.298972, 9.502708, 9.1346, 6.904761905, 3.793103, 1.660377, 1.266667]
                  ]),
        columns=[1399124, 769198, 496421, 233381, 93917, 61017, 34047, 16558],
        index=[1, 2, 4, 8, 16, 32]
    )

    melt_df = df.reset_index().melt(id_vars='index',
                                    var_name='Variants per Chromosome',
                                    value_name='Speedup')
    melt_df = melt_df.rename(columns={'index': 'Threads'})

    # Process the speedup data for the DQF variant of the algorithm:
    df_dqf = pd.DataFrame(
        np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1.932602, 1.888424, 1.838572, 1.952205, 1.890981, 1.896391, 1.930183, 1.870795],
             [3.683155, 3.606639, 3.303344, 3.700784, 3.555424, 3.587851, 3.200823, 3.096638],
             [7.194678, 6.554392, 5.828078, 6.817131, 6.223478, 6.218108, 5.307272, 4.26792],
             [12.29023, 11.69549, 11.05558, 11.73058, 9.204516, 9.231979, 6.80883, 5.293304],
             [22.23656, 18.41054, 18.37324, 14.03472, 12.33516, 11.67186, 7.392569, 5.376654]]
        ),
        columns=[1399124, 769198, 496421, 233381, 93917, 61017, 34047, 16558],
        index=[1, 2, 4, 8, 16, 32]
    )

    melt_df_dqf = df_dqf.reset_index().melt(id_vars='index',
                                            var_name='Variants per Chromosome',
                                            value_name='Speedup')
    melt_df_dqf = melt_df_dqf.rename(columns={'index': 'Threads'})

    # Plotting the data
    threads = [1, 2, 4, 8, 16, 32]
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6), sharey=True, sharex=True)

    palette = sns.color_palette("rocket_r", df.shape[1])

    for i, mdf in enumerate([melt_df, melt_df_dqf]):
        # Line plot for speedup vs threads
        sns.lineplot(data=mdf, x='Threads', y='Speedup', hue='Variants per Chromosome', marker='o',
                     palette=palette, ax=axs[i], lw=1.5)

        # Plot ideal line (for comparison)
        axs[i].plot(threads, threads, label='Ideal', color='#45C4B0', linestyle='--', lw=1.75)

        # Customize plot
        axs[i].set_xlabel('Threads')
        axs[i].set_ylabel('Speedup')

        # Adjust x-ticks to show only number of threads
        axs[i].set_xticks(threads)
        # Turn off the legend:
        axs[i].get_legend().set_visible(False)

    axs[0].set_title(r'$\bf{(a)}$' + ' Multi-threading with Triangular LD (default)', loc='left')
    axs[1].set_title(r'$\bf{(b)}$' + ' Multi-threading with Triangular LD + DQF', loc='left')

    # Create custom legend outside the plot (right side)
    handles, labels = axs[0].get_legend_handles_labels()

    # Humanize the labels
    n_snps_labels = {
        '1399124': 'CHR1 | MAF>20 | 1.4m',
        '769198': 'CHR9 | MAF>20 | 769k',
        '496421': 'CHR15 | MAF>20 | 496k',
        '233381': 'CHR22 | MAF>20 | 233k',
        '93917': 'CHR1 | HM3+ | 94k',
        '61017': 'CHR10 | HM3+ | 61k',
        '34047': 'CHR18 | HM3+ | 34k',
        '16558': 'CHR22 | HM3+ | 17k'
    }

    new_labels = [n_snps_labels[label] if label.replace('.', '').isdigit() else label for label in labels]

    # Create new legend with humanized labels, placed outside the plot
    fig.legend(handles, new_labels, title='Chromosome | LD Panel | # Variants',
               bbox_to_anchor=(1., 0.5), loc='center left', fontsize='x-small',
               title_fontsize='x-small', frameon=False)

    # Layout adjustments
    plt.tight_layout()

    plt.savefig(osp.join(iargs.output_dir, f'runtime_per_iteration_speedup_intel.{iargs.extension}'),
                bbox_inches="tight")
    plt.close()


def plot_parallelism_vs_resource_util_18m(iargs):
    """
    Raw data is provided by experiments from Chirayu Anant Haryan.
    :param iargs:
    :return:
    """

    sns.set_context("paper", font_scale=1)

    df = pd.DataFrame(
        {'Threads': {0: 1, 1: 42, 2: 14, 3: 14},
         'Processes': {0: 1, 1: 1, 2: 9, 3: 3},
         'Wallclock_Time_m': {0: 236.80949999999999,
                              1: 19.0965,
                              2: 9.158333333333333,
                              3: 16.94066666666667},
         'peak_memory_gb': {0: 13.153373718261719,
                            1: 13.146907806396484,
                            2: 72.20704650878906,
                            3: 29.925189971923828},
         'Resources': {0: '1 | 1', 1: '1 | 42', 2: '9 | 14', 3: '3 | 14'}}
    )

    # Create two subplots that share the same x-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharex=True)

    order = ['1 | 1', '1 | 42', '3 | 14', '9 | 14']

    # Plot the second barplot for wallclock_time in the second subplot (ax2)
    sns.barplot(x='Resources', y='Wallclock_Time_m',
                hue='Resources',
                order=order,
                hue_order=order,
                palette='BuPu',
                data=df, ax=ax1)
    ax1.set_title(r'$\bf{(a)}$' + ' Wallclock Time (minutes)', loc='left')  # Label for the right y-axis
    ax1.tick_params(axis='y')
    ax1.set_ylabel('')
    ax1.set_xlabel('')
    add_labels_to_bars(ax1, rotation=90., orientation='vertical')

    sns.barplot(x='Resources', y='peak_memory_gb', data=df, ax=ax2,
                hue='Resources',
                palette='BuPu',
                hue_order=order,
                order=order)
    ax2.set_title(r'$\bf{(b)}$' + ' Peak Memory (GB)', loc='left')  # Label for the left y-axis
    ax2.tick_params(axis='y')
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    add_labels_to_bars(ax2, rotation=90., orientation='vertical')

    # Set the x-axis label (common for both subplots)
    fig.supxlabel('Parallelism Resources (# Processes | # Threads)')

    # Add a title for the entire figure
    fig.suptitle('Resource utilization with parallelism at 18m variants')

    # Adjust layout for better spacing between the subplots
    plt.tight_layout()

    plt.savefig(osp.join(iargs.output_dir, f'resource_util_18m.{iargs.extension}'),
                bbox_inches="tight")
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot benchmarking supplementary figures.')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='figures/supp_figures/',
                        help='The directory where to store the figure panels.')
    parser.add_argument('--extension', dest='extension', type=str,
                        default='eps',
                        help='The extension for the output figure files.')
    args = parser.parse_args()

    # Set seaborn context:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=2)

    # Create the output directory if it does not exist:
    makedir(args.output_dir)
    makedir(osp.join(args.output_dir, 'figure_data'))

    plot_relative_improvement(args)
    plot_accuracy_by_ld_mode(args)
    plot_e_step_runtime_all_models(args)
    plot_speed_metrics_intel_cluster(args)
    plot_parallelism_vs_resource_util_18m(args)
