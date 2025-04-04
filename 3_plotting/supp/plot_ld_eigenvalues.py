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


def plot_ld_min_eigenvalues(iargs,
                            variant_set='hq_imputed_variants_hm3',
                            missing_strategy='all',
                            ld_estimator='windowed',
                            population='EUR',
                            exclude_lrld=False):
    eigs_tab = []

    if missing_strategy == 'all':
        missing_strategy = 'l*'
    elif missing_strategy == 'no_missing':
        missing_strategy = 'ld'
    else:
        missing_strategy = 'ld_xarray'

    ld_panel = {
        'hq_imputed_variants_hm3': 'HapMap3',
        'hq_imputed_variants_maf001': 'MAF > 0.001 (13m)',
        'hq_imputed_variants': 'MAC > 20 (18m)'
    }

    for ld_path in glob.glob(f"data/{missing_strategy}/{variant_set}/{population}/{ld_estimator}/*/chr_*/"):

        ldm = mgp.LDMatrix.from_directory(ld_path)

        if 'ld_xarray' in ld_path:
            data_type = 'w/ MI'
        else:
            data_type = 'w/o MI'

        estimator = ldm.ld_estimator
        dtype = np.dtype(ldm.stored_dtype).name

        spectral_props = ldm.get_store_attr('Spectral properties')
        eig_val = np.abs(np.minimum(spectral_props['Extremal']['min'], 0.))

        eigs_tab.append(
            {'Chromosome': ldm.chromosome,
             'Estimator': estimator,
             'Dtype': dtype,
             'Data_representation': data_type,
             'Type': f'{dtype} | {data_type}',
             'Lambda_min': eig_val
             }
        )

        if exclude_lrld and ld_estimator == 'windowed':

            if 'Extremal (excluding LRLD)' in spectral_props.keys():
                eig_val = np.abs(np.minimum(spectral_props['Extremal (excluding LRLD)']['min'], 0.))
            else:
                eig_val = np.abs(np.minimum(spectral_props['Extremal']['min'], 0.))

            eigs_tab.append(
                {'Chromosome': ldm.chromosome,
                 'Estimator': estimator,
                 'Dtype': dtype,
                 'Data_representation': data_type,
                 'Type': f'{dtype} | {data_type} | No LRLD',
                 'Lambda_min': eig_val
                 }
            )

    eigs_tab = pd.DataFrame(eigs_tab)

    if len(eigs_tab) < 1:
        raise Exception("No data found for the specified parameters.")

    uniq_types = eigs_tab['Type'].unique()

    if not any(['w/ MI' in t for t in uniq_types]):
        eigs_tab['Type'] = eigs_tab['Type'].str.replace(' | w/ MI', '').str.replace(' | w/o MI', '')

    hue_order = sorted(eigs_tab['Type'].unique())

    plt.figure(figsize=(8, 4.5))
    g = sns.barplot(data=eigs_tab, x='Chromosome', y='Lambda_min', hue='Type',
                    palette='tab20',
                    hue_order=hue_order)
    #g.set_yscale("log")
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Variant set: {ld_panel[variant_set]} | Population: {population} | LD Estimator: {ld_estimator}")
    plt.ylabel(r"Magnitude of Smallest Negative Eigenvalue ($|min(\lambda_{min}, 0)|$)")

    plt.savefig(osp.join(iargs.output_dir,
                         f'lambda_min_{variant_set}_{population}_{ld_estimator}.{iargs.extension}'),
                bbox_inches="tight")

    plt.close()

    return eigs_tab


def compare_min_eigenvalue_int16_float64(iargs):

    import magenpy as mgp

    results = []

    for chrom in range(1, 23):
        ldm_float64 = mgp.LDMatrix.from_path(f"data/ld/eur/converted/ukbb_50k_windowed/float64/chr_{chrom}")
        ldm_int16 = mgp.LDMatrix.from_path(f"data/ld/eur/converted/ukbb_50k_windowed/int16/chr_{chrom}")

        extremal_eigs_fp64 = ldm_float64.estimate_extremal_eigenvalues()
        extremal_eigs_int16 = ldm_int16.estimate_extremal_eigenvalues()

        results.append({
            'Chromosome': chrom,
            'float64': np.abs(np.minimum(extremal_eigs_fp64['min'], 0.)),
            'int16': np.abs(np.minimum(extremal_eigs_int16['min'], 0.))
        })

    results = pd.DataFrame(results)

    sns.scatterplot(data=results, x='float64', y='int16')
    plt.xlabel(r"$|min(\lambda_{min}, 0)|$ (float64)")
    plt.ylabel(r"$|min(\lambda_{min}, 0)|$ (int16)")

    max_min_eig = results[['float64', 'int16']].max().max()

    # Add diagonal line (from 0. to maximum values across both axes):
    plt.plot([0, max_min_eig], [0, max_min_eig], color='grey', linestyle='--')

    plt.savefig(osp.join(iargs.output_dir, f'lambda_min_int16_vs_float64.{iargs.extension}'))
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot the panels of Figure 3.')
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

    sns.set_context("paper")

    """
    plot_ld_min_eigenvalues(args, 'hq_imputed_variants_hm3', ld_estimator='windowed', population='EUR',
                            exclude_lrld=True)

    plot_ld_min_eigenvalues(args, 'hq_imputed_variants_hm3', missing_strategy='no_missing',
                            ld_estimator='windowed', population='AFR',
                            exclude_lrld=True)

    plot_ld_min_eigenvalues(args, 'hq_imputed_variants_hm3', ld_estimator='windowed', population='EAS',
                            exclude_lrld=True)

    plot_ld_min_eigenvalues(args, 'hq_imputed_variants_hm3', ld_estimator='block', population='EUR')
    plot_ld_min_eigenvalues(args, 'hq_imputed_variants_maf001', ld_estimator='block', population='EUR')
    """

    compare_min_eigenvalue_int16_float64(args)
