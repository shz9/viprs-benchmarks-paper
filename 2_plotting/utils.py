import re
import pandas as pd
import numpy as np
import glob
import os.path as osp




def extract_performance_statistics(time_file):
    """
    Extract relevant performance statistics from the output of
    /usr/bin/time -v command.

    Primarily, this function, returns the following metrics:
    - Maximum resident set size
    - Wall clock time
    - CPU utilization.

    :param time_file: A path to the file containing the output of /usr/bin/time -v
    :return: A dictionary of parsed performance metrics
    """

    with open(time_file, 'r') as file:
        data = file.read()

    # Extract wall-clock time
    wall_clock_time = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\d*):(\d*\.\d*)', data)
    if wall_clock_time:
        minutes = int(wall_clock_time.group(1))
        seconds = float(wall_clock_time.group(2))
        wall_clock_time = minutes + seconds / 60  # Convert to minutes
    else:
        wall_clock_time = None

    # Extract CPU utilization
    cpu_utilization = re.search(r'Percent of CPU this job got:\s*(\d*)%', data)
    if cpu_utilization:
        cpu_utilization = int(cpu_utilization.group(1))
    else:
        cpu_utilization = None

    # Extract memory utilization
    memory_utilization = re.search(r'Maximum resident set size \(kbytes\):\s*(\d*)', data)
    if memory_utilization:
        memory_utilization = int(memory_utilization.group(1)) / 1024**2  # Convert from KB to GB
    else:
        memory_utilization = None

    return {
        'Wallclock_Time': wall_clock_time,
        'CPU_Util': cpu_utilization,
        'Peak_Memory_GB': memory_utilization
    }


def add_labels_to_bars(g, rotation=90, fontsize='smaller', units=None, orientation='vertical'):
    """
    This function takes a barplot and adds labels above each bar with its value.
    """

    from seaborn.axisgrid import FacetGrid

    if isinstance(g, FacetGrid):
        axes = g.axes.flatten()
    else:
        axes = [g]

    for ax in axes:

        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        scale = ax.get_yaxis().get_scale()

        for p in ax.patches:

            if scale == 'linear':

                height = p.get_height() - y_min
                width = p.get_width() - x_min

                if orientation == 'vertical':
                    value = height
                    x = p.get_x() + p.get_width() / 2
                    if height > 0.5 * y_max:
                        y = y_min + height / 2
                        on_top = False
                    else:
                        y = y_min + height * 1.05
                        on_top = True
                else:  # horizontal barplot
                    value = width
                    y = p.get_y() + p.get_height() / 2
                    if width > 0.5 * x_max:
                        x = x_min + width / 2
                        on_top = False
                    else:
                        x = x_min + width * 1.05
                        on_top = True

            else:

                height = np.log10(p.get_height()) - np.log10(y_min)
                width = np.log10(p.get_width()) - np.log10(x_min)

                if orientation == 'vertical':
                    value = 10 ** height
                    x = p.get_x() + p.get_width() / 2
                    if height > 0.5 * np.log10(y_max):
                        y = 10 ** (np.log10(y_min) + height / 2)
                        on_top = False
                    else:
                        y = 10 ** (np.log10(y_min) + height * 1.05)
                        on_top = True
                else:  # horizontal barplot
                    value = 10 ** width
                    y = p.get_y() + p.get_height() / 2
                    if width > 0.5 * np.log10(x_max):
                        x = 10 ** (np.log10(x_min) + width / 2)
                        on_top = False
                    else:
                        x = 10 ** (np.log10(x_min) + width * 1.05)
                        on_top = True

            label = f'{value:.3f}'
            if units:
                label += f' {units}'

            if orientation == 'horizontal':
                ha = 'left' if on_top else 'center'
                va = 'center'
            else:  # orientation is 'vertical'
                ha = 'center'
                va = 'bottom' if on_top else 'center'

            ax.text(x,
                    y,
                    label,
                    color='black',
                    fontsize=fontsize,
                    rotation=rotation,
                    ha=ha,
                    va=va)


model_versions = {
    'old_viprs': 'v0.0.4',
    'new_viprs': 'v0.1'
}

model_colors = {
    'v0.0.4': 'skyblue',
    'v0.1': 'salmon'
}


def extract_total_runtime_stats(ld_datatype='int8',
                                ld_mode='Triangular LD',
                                dequantize=False,
                                threads=1,
                                jobs=1):

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
        f"data/benchmark_results/total_runtime/fold_*/"
        f"new_viprs/l{ld_datatype}_m{ld_mode}_q{dequantize}_t{threads}_j{jobs}_lmo_set_zero.txt",
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


def extract_accuracy_metrics(ld_datatype='int8',
                             ld_mode='Triangular LD',
                             dequantize=False,
                             threads=1,
                             jobs=1,
                             lmo='set_zero'):

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
    lmo = lmo or '*'

    pred_files = [
        f"data/benchmark_results/prediction/fold_*/new_viprs/"
        f"l{ld_datatype}_m{ld_mode}_q{dequantize}_t{threads}_j{jobs}_lmo_{lmo}.csv",
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
                pred['lambda_min'] = fname.split('lmo_')[1].replace('.csv', '')

                pred['Dequantize'] = 'qtrue' in f

            else:
                pred['Model'] = model_versions['old_viprs']
                pred['LD Mode'] = 'Symmetric LD'
                pred['Dequantize'] = False
                pred['LD Data Type'] = 'float64'
                pred['Processes'] = 1
                pred['Threads'] = 1
                pred['Fold'] = int(osp.basename(osp.dirname(f)).replace('fold_', ''))
                pred['lambda_min'] = 'set_zero'

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


def extract_relative_improvement_data(chrom=1):
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
        'Change': 'C++ / CSR LD format',
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

    # Sixth, show improvement with multithreading (8 threads)

    data_improv_df = pd.read_csv(
        f"data/benchmark_results/e_step/new_viprs/chr_{chrom}_timing_results_impcpp_modelALL_lmFalse_dqFalse_prfloat32_threads8.csv"
    )

    data_improv_df = data_improv_df.loc[(data_improv_df['Model'] == 'VIPRS') &
                                        (data_improv_df['axpy_implementation'] == 'Manual')]

    data.append({
        'Change': '+ Threads: 8',
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

    new_viprs_files = glob.glob(f"data/model_fit/benchmark_sumstats/fold_*/new_viprs/"
                                f"l{ld_datatype}_m{ld_mode}_q{dequantize}_t{threads}_j{jobs}_lmo_set_zeroVIPRS*.prof")
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
