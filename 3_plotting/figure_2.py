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
    df = df.sort_values('NormalizedStorage', ascending=False)

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
    add_improvement_annotation(ax, perc_above_bar=-0.11, text_offset=0.3, orientation='horizontal')
    plt.xlabel("Storage (GB) per 1m variants")
    plt.ylabel("LD Matrix Resource")
    plt.title(r"$\bf{(a)}$ LD Storage Requirements", pad=10, loc='left')

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

    fig = plt.figure(figsize=(9, 8))
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
    ax1.set_yticks(np.arange(0, 8, 1))
    ax1.set_ylim(0, 8.)
    ax1.set_title(r'$\bf{(b)}$' + '\nWallclock Time (m)', pad=10, loc='left')
    add_improvement_annotation(ax1, perc_above_bar=.2)

    # Plot the peak memory:
    sns.barplot(x='Model', y='Peak_Memory_GB',
                data=total_runtime, ax=ax2,
                hue='Model',
                order=['v0.0.4', 'v0.1'],
                palette=model_colors,
                legend=False)
    ax2.set_xlabel(None)
    ax2.set_ylabel(None)
    ax2.set_yticks(np.arange(0, 3.5, .5))
    ax2.set_ylim(0, 3.5)
    ax2.set_title(r'$\bf{(c)}$' + '\nPeak Memory (GB)', pad=10, loc='left')
    add_improvement_annotation(ax2, perc_above_bar=.1)

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
    ax3.set_title(r'$\bf{(d)}$' + '\nPrediction' + r' $R^2$', pad=10, loc='left')

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
    ax4.set_title(r"$\bf{(e)}$ Runtime per coordinate ascent step", loc='left')

    def compute_arrow_start_end(data, model1_name='v0.0.4', model2_name='v0.1'):
        """
        This function computes the start and end points for the two-sided arrow
        showing the improvement between the two models at the specified midpoint.
        It also returns the average improvement across all points.
        """

        # Filter data for the given models
        model1_data = data[data['ModelVersion'] == model1_name]
        model2_data = data[data['ModelVersion'] == model2_name]

        # Sort and find unique values of 'n_snps'
        unique_n_snps = np.sort(data['n_snps'].unique())

        # Compute the median of n_snps and find the next largest available value in the sorted list
        mid_n_snps = np.median(data['n_snps'])

        # Find the index of the closest value in unique_n_snps that is greater than or equal to the median
        mid_n_snps = unique_n_snps[unique_n_snps >= mid_n_snps][0]

        # Find the 'TimePerIteration' values at the midpoint for both models
        model1_at_mid = model1_data.loc[model1_data['n_snps'] == mid_n_snps, 'TimePerIteration'].values[0]
        model2_at_mid = model2_data.loc[model2_data['n_snps'] == mid_n_snps, 'TimePerIteration'].values[0]

        # Compute the start and end coordinates for the arrow
        arrow_start = (mid_n_snps, model1_at_mid)
        arrow_end = (mid_n_snps, model2_at_mid)

        # Merge the two datasets on 'n_snps' to align them
        merged_data = pd.merge(model1_data[['n_snps', 'TimePerIteration']],
                               model2_data[['n_snps', 'TimePerIteration']],
                               on='n_snps',
                               suffixes=('_Model1', '_Model2'))

        # Calculate the improvement for each data point
        merged_data['Improvement'] = merged_data['TimePerIteration_Model1'] / merged_data['TimePerIteration_Model2']

        # Calculate the average improvement
        avg_improvement = merged_data['Improvement'].mean()

        return arrow_start, arrow_end, avg_improvement

    # Call the helper function to compute the arrow's start and end points, and average improvement
    arrow_start, arrow_end, avg_improvement = compute_arrow_start_end(e_step)

    # Add a two-sided arrow to show the average improvement
    ax4.annotate('', xy=arrow_end, xytext=arrow_start,
                 arrowprops=dict(facecolor='grey', edgecolor='grey', arrowstyle='<->', lw=1.5))

    # Compute the mid-y point for the arrow while accounting for the log-scale:
    arrow_mid_y = 10**((np.log10(arrow_start[1]) + np.log10(arrow_end[1])) / 2)

    # Annotating the average improvement value
    ax4.annotate(f'{avg_improvement:.0f}X',
                 xy=(arrow_start[0], arrow_mid_y),
                 xytext=(arrow_start[0]*1.01, arrow_mid_y),  # Move annotation to the right
                 ha='left', va='center', color='black', fontweight='bold')

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
    sns.set_context("paper", font_scale=1.8)

    # Create the output directory if it does not exist:
    makedir(args.output_dir)
    makedir(osp.join(args.output_dir, 'figure_data'))

    plot_panel_a(args)
    plot_panel_b(args)
