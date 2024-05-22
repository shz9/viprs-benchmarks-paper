import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from magenpy.utils.system_utils import makedir
from utils import *


def plot_panel_a(iargs):
    """
    Plot panel A of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = pd.read_csv(osp.join(iargs.output_dir, 'figure_data', 'ld_matrices.csv'),
                     comment='#')
    # Compute normalized storage per 1m SNPs:
    df['NormalizedStorage'] = (df['Storage size (GB)'] / df['Number of variants'])*1e6

    # Sort the dataframe by normalized storage:
    df = df.sort_values('NormalizedStorage')

    plt.figure(figsize=(5, 8))

    colormap = {}
    for r in df['Resource'].unique():
        if 'VIPRS' in r:
            colormap[r] = ['skyblue', 'salmon']['VIPRS(v0.1)' in r]
        else:
            colormap[r] = 'lightgray'

    ax = sns.barplot(data=df,
                     y='Resource',
                     x='NormalizedStorage',
                     hue='Resource',
                     palette=colormap)
    add_labels_to_bars(ax, rotation=0., orientation='horizontal', units='GB')
    plt.xlabel("Storage (GB) per 1m variants")
    plt.ylabel("LD Matrix Resource")

    plt.savefig(osp.join(iargs.output_dir, f'panel_a.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_panel_b(iargs):
    """
    Plot panel B of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    prof_data = extract_profiler_data(aggregate=True)
    prediction = extract_accuracy_metrics()
    e_step = extract_e_step_stats(model='VIPRS')
    total_runtime = extract_total_runtime_stats()

    # Save the figure data:
    prof_data.to_csv(osp.join(iargs.output_dir, 'figure_data', 'profiler_data.csv'), index=False)

    fig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[1, :])

    # Plot the total time:
    sns.barplot(x='Model', y='Total_WallClockTime',
                data=prof_data,
                ax=ax1,
                hue='Model',
                order=['v0.0.4', 'v0.1'],
                palette=model_colors,
                legend=False)
    ax1.set_xlabel(None)
    ax1.set_ylabel(None)
    ax1.set_yticks(np.arange(0, 10, 2))
    ax1.set_title('Wallclock Time (m)', pad=10)

    # Plot the peak memory:
    sns.barplot(x='Model', y='Peak_Memory_GB',
                data=total_runtime, ax=ax2,
                hue='Model',
                order=['v0.0.4', 'v0.1'],
                palette=model_colors,
                legend=False)
    ax2.set_xlabel(None)
    ax2.set_ylabel(None)
    ax2.set_yticks(np.arange(0, 3., .5))
    ax2.set_title('Peak Memory (GB)', pad=10)

    # Plot prediction accuracy:
    sns.barplot(x='Model', y='R-Squared',
                data=prediction,
                ax=ax3,
                hue='Model',
                order=['v0.0.4', 'v0.1'],
                palette=model_colors,
                legend=False)
    ax3.set_xlabel(None)
    ax3.set_ylabel(None)
    ax3.set_yticks(np.arange(0, 0.35, 0.05))
    ax3.set_title('Prediction R-Squared', pad=10)

    # Plot the avg time per E-Step and per chromosome:
    sns.lineplot(x='n_snps', y='TimePerIteration',
                 hue='ModelVersion',
                 data=e_step,
                 ax=ax4,
                 linewidth=3,
                 legend=False,
                 marker='o',
                 markersize=7,
                 palette=model_colors)
    ax4.set_yscale('log')
    ax4.set_xlabel('Variants per Chromosome')
    ax4.set_ylabel('Time per Iteration (s)')

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panel_b.{iargs.extension}'))
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot the panels of Figure 2.')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='figures/figure_2/',
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

    plot_panel_a(args)
    plot_panel_b(args)
