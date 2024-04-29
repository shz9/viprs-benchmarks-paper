import argparse
import glob
import os.path as osp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from magenpy.utils.system_utils import makedir
from utils import extract_performance_statistics


model_versions = {
    'old_viprs': 'v0.0.4',
    'new_viprs': 'v0.1'
}

model_colors = {
    'v0.0.4': 'skyblue',
    'v0.1': 'salmon'
}


def extract_data_panel_b():
    """
    Extract and transform data for panel B of Figure 2.
    :return: The extracted and pre-processed data for panel B
    """

    # --------------------------------------------------------
    # Extract total time metrics:
    total_stats = []

    total_files = [
        "data/benchmark_results/total_runtime/fold_*/old_viprs.txt",
        "data/benchmark_results/total_runtime/fold_*/new_viprs/lint8_t1_j1.txt"
    ]

    for pattern in total_files:
        for f in glob.glob(pattern):
            total_perf = extract_performance_statistics(f)
            if 'new_viprs' in f:
                total_perf['Model'] = model_versions['new_viprs']
            else:
                total_perf['Model'] = model_versions['old_viprs']

            total_stats.append(total_perf)

    total_df = pd.DataFrame(total_stats)

    # --------------------------------------------------------

    # Extract E-Step time metrics:
    e_step_stats = []

    # Extract data for new viprs:
    for f in glob.glob("data/benchmark_results/e_step/new_viprs/chr_*_modelALL_lmTrue_dqFalse_prfloat32_threads1.csv"):
        df = pd.read_csv(f)
        df = df.loc[(df['Model'] == 'VIPRS') & (df['axpy_implementation'] == 'BLAS')]
        df['Model'] = model_versions[f.split('/')[3]]
        e_step_stats.append(df)

    # Extract data for old viprs:
    for f in glob.glob("data/benchmark_results/e_step/old_viprs/chr_*.csv"):
        df = pd.read_csv(f)
        df = df.loc[(df['Model'] == 'VIPRS')]
        df['Model'] = model_versions[f.split('/')[3]]
        e_step_stats.append(df)

    e_step_df = pd.concat(e_step_stats)
    e_step_df = e_step_df.groupby(['Model', 'Chromosome']).agg({'TimePerIteration': 'mean'}).reset_index()

    return {
        'total_runtime': total_df,
        'e_step': e_step_df
    }


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
        'Change': 'Float precision: float32',
        'Improvement': mean_time_old_viprs / data_improv_df['TimePerIteration'].median()
    })

    # Fourth, extract data for the low_memory version:
    data_improv_df = pd.read_csv(
        f"data/benchmark_results/e_step/new_viprs/chr_{chrom}_timing_results_impcpp_modelALL_lmTrue_dqFalse_prfloat32_threads1.csv"
    )

    data_improv_df = data_improv_df.loc[(data_improv_df['Model'] == 'VIPRS') &
                                        (data_improv_df['axpy_implementation'] == 'Manual')]

    data.append({
        'Change': 'Low memory',
        'Improvement': mean_time_old_viprs / data_improv_df['TimePerIteration'].median()
    })

    # Fifth, show improvement with multithreading (2 threads)

    data_improv_df = pd.read_csv(
        f"data/benchmark_results/e_step/new_viprs/chr_{chrom}_timing_results_impcpp_modelALL_lmFalse_dqFalse_prfloat32_threads2.csv"
    )

    data_improv_df = data_improv_df.loc[(data_improv_df['Model'] == 'VIPRS') &
                                        (data_improv_df['axpy_implementation'] == 'Manual')]

    data.append({
        'Change': 'Coordinate Ascent Threads: 2',
        'Improvement': mean_time_old_viprs / data_improv_df['TimePerIteration'].median()
    })

    # Fifth, show improvement with multithreading (4 threads)

    data_improv_df = pd.read_csv(
        f"data/benchmark_results/e_step/new_viprs/chr_{chrom}_timing_results_impcpp_modelALL_lmFalse_dqFalse_prfloat32_threads4.csv"
    )

    data_improv_df = data_improv_df.loc[(data_improv_df['Model'] == 'VIPRS') &
                                        (data_improv_df['axpy_implementation'] == 'Manual')]

    data.append({
        'Change': 'Coordinate Ascent Threads: 4',
        'Improvement': mean_time_old_viprs / data_improv_df['TimePerIteration'].median()
    })

    return pd.DataFrame(data)


def extract_data_panel_d():
    """
    Plot panel D of Figure 2.
    :return: The extracted and pre-processed data for panel D
    """

    # Extract total runtime information for the new viprs:

    new_viprs_files = glob.glob("data/benchmark_results/total_runtime/fold_*/new_viprs/lint8_t*_j*.txt")
    old_viprs_files = glob.glob("data/benchmark_results/total_runtime/fold_*/old_viprs.txt")

    old_viprs_data = []

    for f in old_viprs_files:
        stats = extract_performance_statistics(f)
        old_viprs_data.append(stats['Wallclock_Time'])

    mean_runtime_old = np.mean(old_viprs_data)

    data = []

    for f in new_viprs_files:

        stats = extract_performance_statistics(f)

        fname = osp.basename(f).replace('.txt', '')
        stats['Threads'] = int(fname.split('_')[1].replace('t', ''))
        stats['Jobs'] = int(fname.split('_')[2].replace('j', ''))

        data.append({
            'Threads': int(fname.split('_')[1].replace('t', '')),
            'Processes': int(fname.split('_')[2].replace('j', '')),
            'Improvement in Total Runtime': mean_runtime_old / stats['Wallclock_Time']
        })

    return pd.DataFrame(data)


def extract_data_panel_e():
    """

    :return:
    """

    import ast

    new_viprs_files = glob.glob("data/benchmark_results/prediction/fold_*/new_viprs/*_j1.csv")
    old_viprs_files = glob.glob("data/benchmark_results/prediction/fold_*/old_viprs.csv")

    data = []

    for f in old_viprs_files + new_viprs_files:

        df = pd.read_csv(f)

        pred = {
            'R-Squared': ast.literal_eval(df.pseudo_R2[0])[0]
        }

        if 'new_viprs' in f:
            pred['Model'] = model_versions['new_viprs']

            fname = osp.basename(f).replace('.txt', '')
            pred['Threads'] = int(fname.split('_')[1].replace('t', ''))
            pred['LD'] = fname.split('_')[0][1:]
        else:
            pred['Model'] = model_versions['old_viprs']
            pred['Threads'] = 1
            pred['LD'] = 'float64'

        data.append(pred)

    return pd.DataFrame(data)


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

    sns.barplot(data=df,
                y='Resource',
                x='NormalizedStorage',
                hue='Resource',
                palette={r: ['skyblue', 'salmon']['VIPRS(v0.1)' in r] for r in df['Resource'].unique()})
    plt.xlabel("Storage (GB) per 1m variants")
    plt.ylabel("LD Matrix Resource")

    plt.savefig(osp.join(iargs.output_dir, f'panel_a.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_panel_b(iargs):
    """
    Plot panel B of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    timing_data = extract_data_panel_b()

    # Save the figure data:
    timing_data['total_runtime'].to_csv(osp.join(iargs.output_dir, 'figure_data', 'total_runtime.csv'), index=False)
    timing_data['e_step'].to_csv(osp.join(iargs.output_dir, 'figure_data', 'e_step.csv'), index=False)

    fig, axs = plt.subplots(nrows=3, figsize=(10, 14))

    # Plot the total time:
    sns.barplot(x='Model', y='Wallclock_Time',
                data=timing_data['total_runtime'], ax=axs[0],
                hue='Model',
                palette=model_colors,
                legend=False)
    axs[0].set_ylabel('Total Time (minutes)')

    # Plot the peak memory:
    sns.barplot(x='Model', y='Peak_Memory_GB',
                data=timing_data['total_runtime'], ax=axs[1],
                hue='Model',
                palette=model_colors,
                legend=False)
    axs[1].set_ylabel('Peak Memory (GB)')

    # Plot the avg time per E-Step and per chromosome:
    sns.barplot(x='Chromosome', y='TimePerIteration',
                hue='Model', data=timing_data['e_step'],
                ax=axs[2],
                legend=False,
                palette=model_colors)
    axs[2].set_ylabel('Avg Time per E-Step (s)')

    plt.suptitle("Computational Efficiency of VIPRS Models")

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

    plt.ylabel("Fold improvement over VIPRS v0.0.4")
    plt.xlabel("Optimization")
    plt.title("Improvements in E-Step Runtime")

    plt.savefig(osp.join(iargs.output_dir, f'panel_c.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_panel_d(iargs):
    """
    Plot panel D of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_data_panel_d()

    # Generate a grouped barplot that shows the improvement in total runtime
    # for difference processes (x-axis) and number of threads (`hue`):

    sns.barplot(x='Processes', y='Improvement in Total Runtime',
                hue='Threads', data=df,
                palette={1: '#FFB2A8', 2: '#FF99A3', 4: '#FF8C7A'})

    plt.xlabel("Processes")
    plt.ylabel("Fold improvement over VIPRS v0.0.4")
    plt.title("Improvement in Total Runtime with Parallelism")

    plt.savefig(osp.join(iargs.output_dir, f'panel_d.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_panel_e(iargs):
    """
    Plot panel E of Figure 2.
    :param iargs:
    :return:
    """

    df = extract_data_panel_e()

    df_B = df[df['Model'] == 'v0.0.4']
    sns.barplot(x='Model', y='R-Squared', data=df_B, color='skyblue')

    # Create the grouped barplot
    df_A = df[df['Model'] == 'v0.1']
    sns.barplot(x='LD', y='R-Squared', hue='Threads', data=df_A,
                palette={1: '#FFB2A8', 2: '#FF99A3', 4: '#FF8C7A'})

    # Customize the plot
    plt.xlabel('Model / LD Data Type')
    plt.ylabel('Pseudo R-Squared')
    plt.title('Prediction accuracy of VIPRS models on Standing Height')
    plt.legend()

    plt.savefig(osp.join(iargs.output_dir, f'panel_e.{iargs.extension}'), bbox_inches="tight")
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
    plot_panel_c(args)
    plot_panel_d(args)
    plot_panel_e(args)
