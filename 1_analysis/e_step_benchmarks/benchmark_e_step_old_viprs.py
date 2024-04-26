import sys
import os
import os.path as osp

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import math
import timeit
import platform
import argparse

import numpy as np
import pandas as pd
import magenpy as mgp
from magenpy.utils.system_utils import makedir

from viprs.model.VIPRS import VIPRS
from viprs.model.VIPRSMix import VIPRSMix
from viprs.model.VIPRSGrid import VIPRSGrid
from viprs.utils.HyperparameterGrid import HyperparameterGrid
from dask.diagnostics import ResourceProfiler

# ------------------------------------------------------------------------


def measure_e_step_performance(viprs_model, n_experiments=15, n_calls='auto', warm_up=5, initialize_once=False):
    """
    Measure the time it takes to execute the E-Step of a VIPRS model for `n_calls`.
    :param viprs_model: The VIPRS model to benchmark.
    :param n_experiments: The number of experiments to run.
    :param n_calls: The number of times to repeat the E-Step in each experiment.
    :param warm_up: The number of warm-up experiments to run.
    :param initialize_once: If True, initialize the model once and run the E-Step multiple times.
    """

    def exec_func():
        if not initialize_once:
            viprs_model.initialize()
        viprs_model.e_step()

    viprs_model.verbose = False

    if initialize_once:
        viprs_model.initialize()

    if n_calls == 'auto':
        # If auto, then we will determine the number of calls automatically:
        # Here, we roughly repeat until the total time is at least ~1 second:
        # Note that the minimum number of calls is 5. If this takes too long,
        # then set the number of calls manually?
        time_iter = math.ceil(1. / np.mean(timeit.repeat(exec_func, number=5, repeat=5)))
        n_calls = 5 * int(time_iter)

    with ResourceProfiler(dt=0.1) as rprof:
        times = timeit.repeat(exec_func,
                              repeat=n_experiments + warm_up,
                              number=n_calls)

    try:
        peak_mem = np.max([r.mem for r in rprof.results])
        avg_mem = np.mean([r.mem for r in rprof.results])
        peak_cpu = np.max([r.cpu for r in rprof.results])
        avg_cpu = np.mean([r.cpu for r in rprof.results])
        std_cpu = np.std([r.cpu for r in rprof.results])
    except ValueError:
        peak_mem = np.nan
        avg_mem = np.nan
        peak_cpu = np.nan
        avg_cpu = np.nan
        std_cpu = np.nan

    return {
        'Time': times[warm_up:],
        'Repeats': n_calls,
        'Peak_Mem_MB': peak_mem,
        'Avg_Mem_MB': avg_mem,
        'Peak_CPU_Util': peak_cpu,
        'Avg_CPU_Util': avg_cpu,
        'Std_CPU_Util': std_cpu,
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Benchmark the speed of the E-Step in VIPRS (v0.0.4)")

    parser.add_argument('--temp-dir', dest='temp_dir', type=str, required=True,
                        help='Temporary directory to store intermediate files.')
    parser.add_argument('--output-dir', dest='output_dir', type=str, required=True,
                        help='Output directory to store benchmarking results.')
    parser.add_argument('-l', '--ld-panel', dest='ld_dir', type=str,
                        help='The path to the directory where the LD matrices are stored. '
                             'Can be a wildcard of the form ld/chr_*')
    parser.add_argument('-s', '--sumstats', dest='sumstats_path', type=str,
                        default=mgp.ukb_height_fastGWA_path(),
                        help='The summary statistics directory or file. Can be a '
                             'wildcard of the form sumstats/chr_*')
    parser.add_argument('--sumstats-format', dest='sumstats_format',
                        type=str, default='fastGWA')
    parser.add_argument('--n-experiments', dest='n_experiments', type=int, default=15,
                        help='The number of experiments to run.')
    parser.add_argument('--n-experiments-grid', dest='n_experiments_grid', type=int, default=7,
                        help='The number of experiments to run for the grid model.')
    parser.add_argument('--n-calls', dest='n_calls', default='auto',
                        help='The number of times to call the E-Step in each experiment '
                             '(if set to auto, number of calls will be determined automatically).')
    parser.add_argument('--n-calls-grid', dest='n_calls_grid', default='auto',
                        help='The number of times to call the E-Step in each experiment for the grid model '
                             '(if set to auto, number of calls will be determined automatically).')
    parser.add_argument('--warm-up', dest='warm_up', type=int, default=10,
                        help="The number of warm-up experiments to run (this will take care of"
                             "any overhead due to loading required libraries, JIT compilation, CPU warmup time, etc.)")
    parser.add_argument('--initialize-once', dest='initialize_once',
                        default=False, action='store_true',
                        help='If True, initialize the model once and run the E-Step multiple times.')
    parser.add_argument('--grid-size', dest='grid_size', type=int, default=10,
                        help='The number of grid points to use for the VIPRSGrid model.')
    parser.add_argument('--seed', dest='seed', type=int, default=7209,
                        help='The random seed to use for the benchmarking experiments.')
    parser.add_argument('--file-prefix', dest='file_prefix', type=str, default='',
                        help='The prefix to use for the output files.')

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.ld_dir is None:
        gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                                sumstats_files=args.sumstats_path,
                                sumstats_format=args.sumstats_format,
                                temp_dir=args.temp_dir,
                                backend='xarray')
        gdl.compute_ld('shrinkage',
                       genetic_map_ne=11400,
                       genetic_map_sample_size=183,
                       output_dir=args.output_dir)
    else:
        gdl = mgp.GWADataLoader(ld_store_files=args.ld_dir,
                                sumstats_files=args.sumstats_path,
                                sumstats_format=args.sumstats_format,
                                temp_dir=args.temp_dir,
                                backend='xarray')

    if len(gdl.chromosomes) > 1:
        raise Exception("Benchmarking script only works with one chromosome at-a-time!")

    general_info = {
        'System': platform.platform(),
        'n_snps': gdl.n_snps,
        'Chromosome': gdl.chromosomes[0],
        'median_n_neighbors': np.median(np.concatenate([ld.window_size for ld in gdl.ld.values()])),
        'avg_n_neighbors': np.mean(np.concatenate([ld.window_size for ld in gdl.ld.values()])),
        'Low_memory': False,
        'float_precision': 'float64',
        'init_once': args.initialize_once,
        'dequantize_on_the_fly': False,
    }
    n_snps = gdl.n_snps

    dfs = []

    # ------------------------------------------------------------------------

    # Create a grid:
    grid = HyperparameterGrid()
    # Generate a grid for pi using 5 equidistant grid points:
    grid.generate_pi_grid(steps=args.grid_size, n_snps=gdl.n_snps)
    # Generate a grid for sigma epsilon using 5 equidistant grid points:
    grid.generate_sigma_epsilon_grid(steps=args.grid_size)

    # ------------------------------------------------------------------------

    # ======================= Standard VIPRS =======================

    print("Timing: VIPRS | Manual linalg | Cython")

    v = VIPRS(gdl)

    res = measure_e_step_performance(v, 
                                     n_experiments=args.n_experiments,
                                     n_calls=args.n_calls,
                                     warm_up=args.warm_up,
                                     initialize_once=args.initialize_once)
    res.update({
        'Model': 'VIPRS',
        'Threads': 1,
        'E-Step Implementation': 'Cython',
        'axpy_implementation': 'Manual',
    })

    dfs.append(pd.DataFrame(res))
    
    print("Timing: VIPRSMix | Manual linalg | Cython")

    v = VIPRSMix(gdl, K=20)

    res = measure_e_step_performance(v, n_experiments=args.n_experiments,
                                     n_calls=args.n_calls,
                                     warm_up=args.warm_up,
                                     initialize_once=args.initialize_once)
    res.update({
        'Model': 'VIPRSMix(K=20)',
        'Threads': 1,
        'E-Step Implementation': 'Cython',
        'axpy_implementation': 'Manual',
    })

    dfs.append(pd.DataFrame(res))

    # ======================= VIPRS Grid =======================

    print("Timing: VIPRSGrid | Manual linalg | Cython")

    v = VIPRSGrid(gdl, grid)
    res = measure_e_step_performance(v,
                                     n_experiments=args.n_experiments_grid,
                                     n_calls=args.n_calls_grid,
                                     warm_up=args.warm_up,
                                     initialize_once=args.initialize_once,
                                     )
    res.update({
        'Model': 'VIPRSGrid',
        'Threads': 1,
        'E-Step Implementation': 'Cython',
        'axpy_implementation': 'Manual',
    })

    dfs.append(pd.DataFrame(res))

    # ==========================================================

    dfs = pd.concat(dfs)

    for d, di in general_info.items():
        dfs[d] = di

    output_fname = (f"{args.file_prefix}timing_results.csv")

    # Calculate time per-iteration:
    dfs['TimePerIteration'] = dfs['Time'] / dfs['Repeats']
    makedir(args.output_dir)
    dfs.to_csv(osp.join(args.output_dir, output_fname), index=False)

    gdl.cleanup()

    print("--------------------------------")
    print("Benchmark results are stored in:")
    print(args.output_dir)

