import argparse
import glob
import os.path as osp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from magenpy.utils.system_utils import makedir
from utils import extract_performance_statistics, add_labels_to_bars


model_versions = {
    'old_viprs': 'v0.0.4',
    'new_viprs': 'v0.1'
}

model_colors = {
    'v0.0.4': 'skyblue',
    'v0.1': 'salmon'
}


def extract_total_runtime_stats(ld_datatype='int8', ld_mode='Triangular LD', dequantize=False, threads=1, jobs=1):

    if ld_mode is None:
        ld_mode = '*'
    else:
        ld_mode = str(ld_mode == 'Triangular LD').lower()

    if dequantize is None:
        dequantize = '*'
    else:
        dequantize = str(dequantize).lower()

    ld_datatype = ld_datatype or '*'
    threads = threads or '*'
    jobs = jobs or '*'

    total_files = [
        f"data/benchmark_results/total_runtime/fold_*/new_viprs/l{ld_datatype}_m{ld_mode}_q{dequantize}_t{threads}_j{jobs}.txt",
        "data/benchmark_results/total_runtime/fold_*/old_viprs.txt",
    ]

    stats = []

    for pattern in total_files:
        for f in glob.glob(pattern):
            total_perf = extract_performance_statistics(f)
            if 'new_viprs' in f:
                total_perf['Model'] = model_versions['new_viprs']

                if 'mtrue' in f:
                    total_perf['LD Mode'] = 'Triangular LD'
                else:
                    total_perf['LD Mode'] = 'Symmetric LD'

                total_perf['Dequantize'] = 'qtrue' in f
                total_perf['Fold'] = int(osp.basename(osp.dirname(osp.dirname(f))).replace('fold_', ''))

                fname = osp.basename(f).replace('.txt', '')
                total_perf['Processes'] = int(fname.split('_')[4].replace('j', ''))
                total_perf['Threads'] = int(fname.split('_')[3].replace('t', ''))
                total_perf['LD Data Type'] = fname.split('_')[0][1:]


            else:
                total_perf['Model'] = model_versions['old_viprs']
                total_perf['LD Mode'] = 'Symmetric LD'
                total_perf['Dequantize'] = False
                total_perf['Processes'] = 1
                total_perf['Threads'] = 1
                total_perf['LD Data Type'] = 'float64'
                total_perf['Fold'] = int(osp.basename(osp.dirname(f)).replace('fold_', ''))

            stats.append(total_perf)

    return pd.DataFrame(stats)


def extract_accuracy_metrics(ld_datatype='int8', ld_mode='Triangular LD', dequantize=False, threads=1, jobs=1):

    if ld_mode is None:
        ld_mode = '*'
    else:
        ld_mode = str(ld_mode == 'Triangular LD').lower()

    if dequantize is None:
        dequantize = '*'
    else:
        dequantize = str(dequantize).lower()

    ld_datatype = ld_datatype or '*'
    threads = threads or '*'
    jobs = jobs or '*'

    pred_files = [
        f"data/benchmark_results/prediction/fold_*/new_viprs/l{ld_datatype}_m{ld_mode}_q{dequantize}_t{threads}_j{jobs}.csv",
        "data/benchmark_results/prediction/fold_*/old_viprs.csv",
    ]

    preds = []

    import ast

    for pattern in pred_files:
        for f in glob.glob(pattern):
            df = pd.read_csv(f)

            pred = {
                'R-Squared': ast.literal_eval(df.pseudo_R2[0])[0]
            }

            if 'new_viprs' in f:
                pred['Model'] = model_versions['new_viprs']

                if 'mtrue' in f:
                    pred['LD Mode'] = 'Triangular LD'
                else:
                    pred['LD Mode'] = 'Symmetric LD'

                fname = osp.basename(f).replace('.csv', '')
                pred['LD Data Type'] = fname.split('_')[0][1:]
                pred['Processes'] = int(fname.split('_')[4].replace('j', ''))
                pred['Threads'] = int(fname.split('_')[3].replace('t', ''))
                pred['Fold'] = int(osp.basename(osp.dirname(osp.dirname(f))).replace('fold_', ''))

                pred['Dequantize'] = 'qtrue' in f

            else:
                pred['Model'] = model_versions['old_viprs']
                pred['LD Mode'] = 'Symmetric LD'
                pred['Dequantize'] = False
                pred['LD Data Type'] = 'float64'
                pred['Processes'] = 1
                pred['Threads'] = 1
                pred['Fold'] = int(osp.basename(osp.dirname(f)).replace('fold_', ''))

            preds.append(pred)

    return pd.DataFrame(preds)


def extract_e_step_stats(chrom=None,
                         float_precision='float32',
                         threads=1,
                         ld_mode='Triangular LD',
                         axpy_implementation='Manual',
                         dequantize=False,
                         model=None,
                         aggregate=True):

    chrom = chrom or '*'
    float_precision = float_precision or '*'
    threads = threads or '*'

    if dequantize is None:
        dequantize = '*'

    if ld_mode is None:
        ld_mode = '*'
    else:
        ld_mode = ld_mode == 'Triangular LD'

    path = (f"data/benchmark_results/e_step/new_viprs/chr_{chrom}_*_model"
            f"ALL_lm{ld_mode}_dq{dequantize}_pr{float_precision}_threads{threads}.csv")

    e_step_stats = []
    # Extract data for new viprs:
    for f in glob.glob(path):
        df = pd.read_csv(f)
        df = df.loc[df['axpy_implementation'] == axpy_implementation]
        if model is not None:
            df = df.loc[df['Model'] == model]
        df['ModelVersion'] = model_versions[f.split('/')[3]]
        e_step_stats.append(df)

    # Extract data for old viprs:
    for f in glob.glob(f"data/benchmark_results/e_step/old_viprs/chr_{chrom}_*.csv"):
        df = pd.read_csv(f)
        if model is not None:
            df = df.loc[df['Model'] == model]
        df['ModelVersion'] = model_versions[f.split('/')[3]]
        e_step_stats.append(df)

    e_step_df = pd.concat(e_step_stats)

    if aggregate:
        e_step_df = e_step_df.groupby(['Model', 'ModelVersion', 'Chromosome', 'n_snps']).agg(
            {'TimePerIteration': 'mean'}
        ).reset_index()

    return e_step_df


def extract_data_panel_c(chrom=1):
    """
    Plot panel C of Figure 2.
    :return: The extracted and pre-processed data for panel C
    """

    # Extract E-Step performance metrics for old viprs:
    old_viprs_df = pd.read_csv(f"data/benchmark_results/e_step/old_viprs/chr_{chrom}_timing_results.csv")
    old_viprs_df = old_viprs_df.loc[(old_viprs_df['Model'] == 'VIPRS')]
    # Extract our reference runtime:
    mean_time_old_viprs = old_viprs_df['TimePerIteration'].median()

    # --------------------------------------------------
    # Extract E-Step performance metrics for new viprs:

    data = []

    # First, let's extract metrics for base version that only changed data structures:
    data_improv_df = pd.read_csv(
        f"data/benchmark_results/e_step/new_viprs/chr_{chrom}_timing_results_impcpp_modelALL_lmFalse_dqFalse_prfloat64_threads1.csv"
    )

    data_improv_df = data_improv_df.loc[(data_improv_df['Model'] == 'VIPRS') &
                                        (data_improv_df['axpy_implementation'] == 'Manual')]

    data.append({
        'Change': 'LD data layout',
        'Improvement': mean_time_old_viprs / data_improv_df['TimePerIteration'].median()
    })

    # Second, let's extract metrics for the version that changed the data layout and float precision:
    data_improv_df = pd.read_csv(
        f"data/benchmark_results/e_step/new_viprs/chr_{chrom}_timing_results_impcpp_modelALL_lmFalse_dqFalse_prfloat32_threads1.csv"
    )

    data_improv_df = data_improv_df.loc[(data_improv_df['Model'] == 'VIPRS') &
                                        (data_improv_df['axpy_implementation'] == 'Manual')]

    data.append({
        'Change': '+ Float precision: float32',
        'Improvement': mean_time_old_viprs / data_improv_df['TimePerIteration'].median()
    })

    # Fifth, show improvement with multithreading (2 threads)

    data_improv_df = pd.read_csv(
        f"data/benchmark_results/e_step/new_viprs/chr_{chrom}_timing_results_impcpp_modelALL_lmFalse_dqFalse_prfloat32_threads2.csv"
    )

    data_improv_df = data_improv_df.loc[(data_improv_df['Model'] == 'VIPRS') &
                                        (data_improv_df['axpy_implementation'] == 'Manual')]

    data.append({
        'Change': '+ Threads: 2',
        'Improvement': mean_time_old_viprs / data_improv_df['TimePerIteration'].median()
    })

    # Fifth, show improvement with multithreading (4 threads)

    data_improv_df = pd.read_csv(
        f"data/benchmark_results/e_step/new_viprs/chr_{chrom}_timing_results_impcpp_modelALL_lmFalse_dqFalse_prfloat32_threads4.csv"
    )

    data_improv_df = data_improv_df.loc[(data_improv_df['Model'] == 'VIPRS') &
                                        (data_improv_df['axpy_implementation'] == 'Manual')]

    data.append({
        'Change': '+ Threads: 4',
        'Improvement': mean_time_old_viprs / data_improv_df['TimePerIteration'].median()
    })

    return pd.DataFrame(data)


def extract_profiler_data(ld_datatype='int8',
                          ld_mode='Triangular LD',
                          dequantize=False,
                          threads=1,
                          jobs=1,
                          aggregate=True):

    if ld_mode is None:
        ld_mode = '*'
    else:
        ld_mode = str(ld_mode == 'Triangular LD').lower()

    if dequantize is None:
        dequantize = '*'
    else:
        dequantize = str(dequantize).lower()

    ld_datatype = ld_datatype or '*'
    threads = threads or '*'
    jobs = jobs or '*'

    new_viprs_files = glob.glob(f"data/model_fit/benchmark_sumstats/fold_*/new_viprs/l{ld_datatype}_m{ld_mode}_q{dequantize}_t{threads}_j{jobs}VIPRS*.prof")
    old_viprs_files = glob.glob("data/model_fit/benchmark_sumstats/fold_*/old_viprs.prof")

    data = []

    for f in old_viprs_files + new_viprs_files:

        df = pd.read_csv(f, sep="\t")

        if aggregate:
            df = pd.DataFrame({
                'Fit_time': [df['Fit_time'].sum()],
                'Load_time': [df['Load_time'].sum()],
                'Total_WallClockTime': [df['Total_WallClockTime'][0] / 60]
            })

        if 'new_viprs' in f:
            df['Model'] = model_versions['new_viprs']

            fname = osp.basename(f).replace('.txt', '')
            df['Threads'] = int(fname.split('_')[3].replace('t', ''))
            df['LD Data Type'] = fname.split('_')[0][1:]
            df['Processes'] = int(fname.split('_')[4].replace('j', '').replace('VIPRS', ''))
            df['Fold'] = int(osp.basename(osp.dirname(osp.dirname(f))).replace('fold_', ''))

            if 'mtrue' in f:
                df['LD Mode'] = 'Triangular LD'
            else:
                df['LD Mode'] = 'Symmetric LD'

            df['Dequantize'] = 'qtrue' in f

        else:
            df['Model'] = model_versions['old_viprs']
            df['Threads'] = 1
            df['Processes'] = 1
            df['LD Mode'] = 'Symmetric LD'
            df['LD Data Type'] = 'float64'
            df['Dequantize'] = False
            df['Fold'] = int(osp.basename(osp.dirname(f)).replace('fold_', ''))

        data.append(df)

    if len(data) < 1:
        raise Exception("Found no data for the benchmarking of the models.")

    return pd.concat(data)


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

    plt.savefig(osp.join(iargs.output_dir, f'panel_a_1.{iargs.extension}'), bbox_inches="tight")
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


def plot_panel_c(iargs):
    """
    Plot panel C of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_data_panel_c()

    # Save the figure data:
    df.to_csv(osp.join(iargs.output_dir, 'figure_data', 'e_step_improvements.csv'), index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Change', y='Improvement', data=df, color='salmon')

    # Change the angle for the x-labels to 45 degrees:
    plt.xticks(rotation=45, ha='right')

    plt.ylabel("Median improvement over v0.0.4")
    plt.xlabel("Incremental changes (left -> right)")
    plt.title("Fold runtime improvements in Coordinate Ascent (E-Step)")

    plt.savefig(osp.join(iargs.output_dir, f'panel_c.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_panel_d(iargs):
    """
    Plot panel D of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_e_step_stats(model='VIPRS', threads=None, aggregate=False)
    df = df.loc[df['ModelVersion'] == 'v0.1']
    df = df.groupby(['Model', 'ModelVersion', 'Chromosome', 'n_snps', 'Threads']).agg(
        {'TimePerIteration': 'mean'}
    ).reset_index()

    plt.figure(figsize=(8.5, 5))
    sns.lineplot(data=df,
                 x='n_snps', y='TimePerIteration', hue='Threads',
                 linewidth=3,
                 marker='o',
                 markersize=7)
    plt.ylabel("Time per Iteration (s)")
    plt.xlabel("Variants per Chromosome")
    plt.title("Multithreading across SNPs:\nRuntime improvements with Parallel Coordinate Ascent")

    plt.savefig(osp.join(iargs.output_dir, f'panel_d_1.{iargs.extension}'), bbox_inches="tight")
    plt.close()

    df = extract_profiler_data(threads=1, jobs=None)
    df = df.loc[df['Model'] == 'v0.1']

    df.to_csv(osp.join(iargs.output_dir, 'figure_data', 'panel_d_processes.csv'), index=False)

    # Generate a grouped barplot that shows the improvement in total runtime
    # for difference processes (x-axis) and number of threads (`hue`):
    plt.figure(figsize=(8.5, 5))
    sns.barplot(x='Processes', y='Total_WallClockTime', data=df, color='salmon')
    plt.ylabel("Wallclock Time (m)")
    plt.xlabel("Processes")
    plt.title("Parallelism across Chromosomes")

    plt.savefig(osp.join(iargs.output_dir, f'panel_d_2.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_panel_e(iargs):
    """
    Plot panel E of Figure 2.
    :param iargs:
    :return:
    """

    df = extract_accuracy_metrics(ld_datatype=None, ld_mode=None, threads=None)

    mean_acc_old = df[df['Model'] == 'v0.0.4']['R-Squared'].mean()

    # Create the grouped barplot
    df_new = df[df['Model'] == 'v0.1']
    g = sns.catplot(kind='bar', x='LD Data Type', y='R-Squared', hue='Threads', data=df_new,
                    col='LD Mode',
                    palette={1: '#FFB2A8', 2: '#FF99A3', 4: '#FF8C7A'})

    # Loop over the subplots and add the mean accuracy for the old model as
    # a dashed line:
    for ax in g.axes.flat:
        ax.axhline(mean_acc_old, ls='--', color='black', label='v0.0.4')

    # Customize the plot
    #plt.xlabel('Model / LD Data Type')
    plt.ylabel('Pseudo R-Squared')
    plt.suptitle('Prediction accuracy of VIPRS models on Standing Height')
    #plt.legend()

    plt.savefig(osp.join(iargs.output_dir, f'panel_e.{iargs.extension}'))
    plt.close()


def plot_ld_mode_panel(iargs):

    prof_df_no_dequantize = extract_profiler_data(ld_datatype=None, ld_mode=None, dequantize=False)
    prof_df_no_dequantize['LD Mode'] = prof_df_no_dequantize['LD Mode'].str.replace(' LD', '')
    prof_df_dequantize = extract_profiler_data(ld_datatype=None, ld_mode=None, dequantize=True)
    prof_df_dequantize['LD Mode'] = prof_df_dequantize['LD Mode'].str.replace(' LD', '')
    prof_df_dequantize['LD Mode'] += '+DQF'
    prof_df = pd.concat([prof_df_no_dequantize, prof_df_dequantize])

    mean_load_time_old_viprs = prof_df.loc[prof_df.Model == 'v0.0.4', 'Load_time'].mean()

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

    fig, axs = plt.subplots(ncols=4, figsize=(14, 6))

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

    sns.barplot(x='LD Mode', y='TimePerIteration',
                order=sorted(e_step_df['LD Mode'].unique()),
                hue='Threads',
                data=e_step_df,
                ax=axs[2])
    axs[2].set_ylabel('Time per Iteration (s)')
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=90)
    axs[2].legend(title='Threads', prop={'size': 8}, title_fontsize=10)

    sns.barplot(data=prof_df, x='LD Mode', y='Load_time', hue='LD Data Type',
                ax=axs[3],
                order=sorted(prof_df['LD Mode'].unique()),
                hue_order=['float64', 'float32', 'int16', 'int8'],
                palette='Paired')
    axs[3].axhline(mean_load_time_old_viprs, c='grey', label='v0.0.4 Load Time', ls='--')
    axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=90)
    axs[3].set_ylabel("LD Matrix Load Time (s)")
    axs[3].legend(prop={'size': 8})

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panel_ld_mode.{iargs.extension}'))
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

    plot_panel_a(args)
    plot_panel_b(args)
    #plot_panel_c(args)
    plot_panel_d(args)
    #plot_panel_e(args)
    plot_ld_mode_panel(args)
