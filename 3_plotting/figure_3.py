import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from magenpy.utils.system_utils import makedir
from utils import *


def plot_panel_a(iargs):
    """
    Plot panel a of Figure 3.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_e_step_stats(model='VIPRS', threads=None, aggregate=False)
    df = df.loc[df['ModelVersion'] == 'v0.1']
    df = df.groupby(['Model', 'ModelVersion', 'Chromosome', 'n_snps', 'Threads']).agg(
        {'TimePerIteration': 'mean'}
    ).reset_index()

    plt.figure(figsize=(8, 5))
    g = sns.lineplot(data=df,
                     x='n_snps', y='TimePerIteration', hue='Threads',
                     linewidth=3,
                     marker='o',
                     markersize=7)
    plt.ylabel("Time per Iteration (s)")
    plt.xlabel("Variants per Chromosome")
    g.set_title('A',
                pad=10, loc='left')
    g.legend(title="Threads", bbox_to_anchor=(1, 1), loc='upper left', fontsize='smaller',
             title_fontsize='smaller')

    # Add an inset figure that shows the speedup across a subset of the chromosomes:

    # Create inset axes - moved to top left
    axins = g.inset_axes([0.1, 0.62, 0.25, 0.35])  # Moved to left side

    # Set the color and width of the border
    for _, spine in axins.spines.items():
        spine.set_edgecolor('lightgray')  # Set the border color (e.g., black)
        spine.set_linewidth(1.7)  # Set the border line width (e.g., 2)

    # Calculate speedup for selected chromosomes
    df_subset = df.copy()

    # Calculate speedup relative to single thread
    baseline_times = df_subset[df_subset['Threads'] == 1].set_index(['Chromosome'])['TimePerIteration']
    df_subset['Speedup'] = df_subset.apply(
        lambda row: baseline_times[row['Chromosome']] / row['TimePerIteration'],
        axis=1
    )

    # Average the speedup across chromosomes:
    df_subset = df_subset.groupby(['Threads', 'n_snps']).agg(
        {'Speedup': 'mean'}
    ).reset_index()

    # Plot speedup in inset
    sns.lineplot(data=df_subset,
                 x='Threads',
                 y='Speedup',
                 marker='o',
                 color='#45C4B0',
                 markersize=5,
                 err_style='band',
                 legend=False,
                 ax=axins)

    # Format inset
    axins.grid(which='both', axis='both', color='lightgray', linestyle='--', linewidth=1)
    axins.set_xticks([1, 2, 4, 8])
    axins.set_xticklabels(['1', '2', '4', '8'], fontsize='xx-small')
    axins.set_yticks([1, 2, 4, 6, 8])
    axins.set_yticklabels(['1', '2', '4', '6', '8'], fontsize='xx-small')
    axins.set_xlabel('Threads', fontsize='xx-small')
    axins.set_ylabel('Speedup', fontsize='xx-small')
    axins.set_xlim(.5, 8.5)
    axins.set_ylim(.5, 8.5)

    axins.plot([1, 2, 4, 8], [1, 2, 4, 8], label='Ideal', ls='--', color='#13678A')
    axins.legend(fontsize='xx-small', loc='upper left')

    plt.savefig(osp.join(iargs.output_dir, f'panel_a_1.{iargs.extension}'), bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------------

    df = extract_profiler_data(threads=None, jobs=None)
    df = df.loc[df['Model'] == 'v0.1']

    df.to_csv(osp.join(iargs.output_dir, 'figure_data', 'panel_d_processes.csv'), index=False)

    # Generate a grouped barplot that shows the improvement in total runtime
    # for difference processes (x-axis) and number of threads (`hue`):
    plt.figure(figsize=(6.5, 5))
    g = sns.barplot(x='Processes', y='Total_WallClockTime', data=df, hue='Threads')
    plt.ylabel("Wallclock Time (m)")
    plt.xlabel("Processes")
    g.set_title('B', loc='left')

    plt.savefig(osp.join(iargs.output_dir, f'panel_a_2.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_panel_b(iargs):

    prof_df_no_dequantize = extract_profiler_data(ld_datatype=None, ld_mode=None, dequantize=False)
    prof_df_no_dequantize['LD Mode'] = prof_df_no_dequantize['LD Mode'].str.replace(' LD', '')
    prof_df_dequantize = extract_profiler_data(ld_datatype=None, ld_mode=None, dequantize=True)
    prof_df_dequantize['LD Mode'] = prof_df_dequantize['LD Mode'].str.replace(' LD', '')
    prof_df_dequantize['LD Mode'] += '+DQF'
    prof_df = pd.concat([prof_df_no_dequantize, prof_df_dequantize])

    # mean_load_time_old_viprs = prof_df.loc[prof_df.Model == 'v0.0.4', 'Load_time'].mean()

    total_df_no_dequantize = extract_total_runtime_stats(ld_mode=None, dequantize=False)
    total_df_no_dequantize['LD Mode'] = total_df_no_dequantize['LD Mode'].str.replace(' LD', '')
    total_df_dequantize = extract_total_runtime_stats(ld_mode=None, dequantize=True)
    total_df_dequantize['LD Mode'] = total_df_dequantize['LD Mode'].str.replace(' LD', '')
    total_df_dequantize['LD Mode'] += '+DQF'
    total_df = pd.concat([total_df_no_dequantize, total_df_dequantize])

    e_step_df_no_dequantize = extract_e_step_stats(ld_mode=None, chrom=1, model='VIPRS',
                                                   threads=None, aggregate=False, dequantize=False)
    e_step_df_no_dequantize['LD Mode'] = e_step_df_no_dequantize['Low_memory'].map(
        {True: 'Triangular', False: 'Symmetric'}
    )
    e_step_df_dequantize = extract_e_step_stats(ld_mode=None, chrom=1, model='VIPRS',
                                                threads=None, aggregate=False, dequantize=True)
    e_step_df_dequantize['LD Mode'] = e_step_df_dequantize['Low_memory'].map(
        {True: 'Triangular+DQF', False: 'Symmetric+DQF'}
    )
    e_step_df = pd.concat([e_step_df_no_dequantize, e_step_df_dequantize])

    total_df = total_df.loc[total_df.Model == 'v0.1']
    prof_df = prof_df.loc[prof_df.Model == 'v0.1']
    e_step_df = e_step_df.loc[e_step_df.ModelVersion == 'v0.1']

    fig, axs = plt.subplots(ncols=4, figsize=(15, 6))

    # Plot the peak memory:
    sns.barplot(x='LD Mode', y='Peak_Memory_GB',
                data=total_df,
                ax=axs[0],
                hue='LD Mode',
                order=sorted(total_df['LD Mode'].unique()),
                palette='Set2',
                legend=False)
    axs[0].set_ylabel('Peak Memory (GB)')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)
    axs[0].set_title('C', loc='left')

    # Plot the total time:
    sns.barplot(x='LD Mode', y='Total_WallClockTime',
                data=prof_df.loc[prof_df['LD Data Type'] == 'int8'],
                ax=axs[1],
                hue='LD Mode',
                order=sorted(prof_df['LD Mode'].unique()),
                palette='Set2',
                legend=False)
    axs[1].set_ylabel('Wallclock Time (m)')
    # Rotate the x-axis tick labels:
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)
    axs[1].set_title('D', loc='left')

    sns.barplot(x='LD Mode', y='TimePerIteration',
                order=sorted(e_step_df['LD Mode'].unique()),
                hue='Threads',
                data=e_step_df,
                ax=axs[2])
    axs[2].set_ylabel('Time per Iteration (s)')
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=90)
    axs[2].legend(title='Threads', prop={'size': 10}, title_fontsize=12)
    axs[2].set_title('E', loc='left')

    sub_prof_df = prof_df.loc[~((prof_df['LD Mode'] == 'Triangular+DQF')
                              & (prof_df['LD Data Type'].isin(['float32', 'float64'])))]

    sns.barplot(data=sub_prof_df,
                x='LD Mode', y='Load_time', hue='LD Data Type',
                ax=axs[3],
                order=sorted(prof_df['LD Mode'].unique()),
                hue_order=['float64', 'float32', 'int16', 'int8'],
                palette='Paired')
    # axs[3].axhline(mean_load_time_old_viprs, c='grey', label='v0.0.4 Load Time', ls='--')
    axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=90)
    axs[3].set_ylabel("LD Matrix Load Time (s)")
    axs[3].legend(prop={'size': 12})
    axs[3].set_title(r'F', loc='left')

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panel_b.{iargs.extension}'))
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot the panels of Figure 3.')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='figures/figure_3/',
                        help='The directory where to store the figure panels.')
    parser.add_argument('--extension', dest='extension', type=str,
                        default='eps',
                        help='The extension for the output figure files.')
    args = parser.parse_args()

    # Set seaborn context:
    link_font("Helvetica")
    sns.set_theme(style="whitegrid", font="Helvetica")
    sns.set_context("paper", font_scale=2)

    # Create the output directory if it does not exist:
    makedir(args.output_dir)
    makedir(osp.join(args.output_dir, 'figure_data'))

    plot_panel_a(args)
    plot_panel_b(args)
