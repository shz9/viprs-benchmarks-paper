import argparse
import seaborn as sns
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import (
    extract_aggregate_evaluation_metrics,
    extract_aggregate_performance_metrics,
    pivot_evaluation_df,
    get_phenotype_category_palette
)
import matplotlib.pyplot as plt
from magenpy.utils.system_utils import makedir
import pandas as pd
import numpy as np


def plot_training_r2_improvement(iargs, model='VIPRS_EM'):
    # Extract the data:

    sns.set_context("paper", font_scale=1.)

    eval_df = extract_aggregate_evaluation_metrics(test_cohort='ukbb', model=model)
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR') &
                          (eval_df['Test_pop'] == 'EUR')]
    pivoted_df = pivot_evaluation_df(eval_df, metric=iargs.metric)

    x_cols = [
        iargs.metric + '_hq_imputed_variants_hm3',
        iargs.metric + '_hq_imputed_variants_hm3',
        iargs.metric + '_hq_imputed_variants_maf001'
    ]
    y_cols = [
        iargs.metric + '_hq_imputed_variants_maf001',
        iargs.metric + '_hq_imputed_variants',
        iargs.metric + '_hq_imputed_variants'
    ]

    axis_labels = {
        iargs.metric + '_hq_imputed_variants_hm3': 'HapMap3+ Incremental $R^2$',
        iargs.metric + '_hq_imputed_variants_maf001': 'MAF > 0.001 (13m variants) Incremental $R^2$',
        iargs.metric + '_hq_imputed_variants': 'MAC > 20 (18m variants) $R^2$'
    }

    fig, ax = plt.subplots(2, 2, figsize=(7.5, 7.5))
    axes = ax.flatten()

    for i, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
        g = sns.scatterplot(data=pivoted_df,
                            x=x_col,
                            y=y_col,
                            hue='general_category',
                            palette=get_phenotype_category_palette(),
                            s=40,
                            ax=axes[i])

        legend = g.get_legend()

        lims = max(max(axes[i].get_ylim()), max(axes[i].get_xlim()))
        x = np.linspace(0, lims, 1000)
        g.plot(x, x, ls='--', lw=.8, color='grey', label='y = x')
        g.plot(x, 1.5 * x, ls='--', lw=.8, color='#007FFF', label='y = 1.5x')
        g.set_ylim((0., lims))
        g.set_xlim((0., lims))

        handles, labels = axes[i].get_legend_handles_labels()

        axes[i].get_legend().remove()

        g.set_xlabel(axis_labels[x_col])
        g.set_ylabel(axis_labels[y_col])


    legend_ax = fig.add_subplot(2, 2, 4)
    legend_ax.axis('off')  # Turn off axis for this subplot
    legend_ax.set_axis_off()
    axes[-1].axis('off')
    axes[-1].set_axis_off()
    # Add legend:
    legend_1 = legend_ax.legend(handles[:-2],
                                labels[:-2],
                                title='Phenotype category',
                                loc='center left',
                                bbox_to_anchor=(-0.2, 0.5))
    legend_1.get_frame().set_linewidth(0.0)
    legend_ax.add_artist(legend_1)

    legend_2 = legend_ax.legend(handles=handles[-2:],
                                labels=labels[-2:],
                                title='Reference lines',
                                loc='center right',
                                bbox_to_anchor=(1., 0.5))
    legend_2.get_frame().set_linewidth(0.0)
    legend_ax.add_artist(legend_2)

    model_name_map = {
        'VIPRS_EM': 'VIPRS',
        'pathwise_VIPRS_GS': 'VIPRS-GS'
    }

    fig.suptitle(f"Prediction $R^2$ on training set with different variant sets ({model_name_map[model]})",)

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'training_r2_{model}.{iargs.extension}'))
    plt.close()


def plot_prediction_accuracy_hapmap_vs_maf001(iargs):

    # Set the seaborn context:
    sns.set_context("paper", font_scale=1.25)

    # Extract the data:
    eval_df = extract_aggregate_evaluation_metrics()
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR')]
    pivoted_df = pivot_evaluation_df(eval_df, metric=iargs.metric)

    pivoted_df['Test cohort'] = pivoted_df['Test_cohort'].map(
        {'ukbb': 'UKB', 'cartagene': 'CARTaGENE'}) + '-' + pivoted_df['Test_pop']

    test_cohorts = ['CARTaGENE-EUR', 'UKB-AMR', 'UKB-MID', 'UKB-CSA', 'UKB-EAS', 'UKB-AFR']

    if iargs.metric + '_err' in eval_df.columns:
        x_col = iargs.metric + '_hq_imputed_variants_hm3'
        y_col = iargs.metric + '_hq_imputed_variants_maf001'
        xerr = iargs.metric + '_err_hq_imputed_variants_hm3'
        yerr = iargs.metric + '_err_hq_imputed_variants_maf001'
    else:
        x_col = 'hq_imputed_variants_hm3'
        y_col = 'hq_imputed_variants_maf001'
        xerr = yerr = None

    # Create figure with extra space on the right for legend
    fig = plt.figure(figsize=(13.5, 7))  # Increased width to accommodate legend

    # Create GridSpec to manage subplot layout
    gs = fig.add_gridspec(2, 4, width_ratios=(2, 2, 2, 1))  # 2 rows, 4 columns (3 for plots, 1 for legend)

    # Create axes for the plots (2x3)
    axes = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    for i, test_cohort in enumerate(test_cohorts):
        subset = pivoted_df.loc[pivoted_df['Test cohort'] == test_cohort]
        g = sns.scatterplot(data=subset,
                            x=x_col,
                            y=y_col,
                            hue='general_category',
                            palette=get_phenotype_category_palette(),
                            ax=axes[i])

        legend = g.get_legend()

        if iargs.add_scatter_errorbars and xerr is not None:
            hue_categories = [text.get_text() for text in legend.get_texts()]
            colors = [handle.get_color() for handle in legend.legend_handles]

            for category, color in zip(hue_categories, colors):
                axes[i].errorbar(x=x_col,
                                 y=y_col,
                                 xerr=xerr,
                                 yerr=yerr,
                                 data=subset.loc[subset.general_category == category],
                                 linestyle='None', label=None, capsize=2, capthick=0.5,
                                 lw=.5, alpha=.65, color=color)

        lims = max(max(axes[i].get_ylim()), max(axes[i].get_xlim()))
        x = np.linspace(0, lims, 1000)
        l1 = g.plot(x, x, ls='--', lw=.8, color='grey', label='y = x')
        l2 = g.plot(x, 1.5 * x, ls='--', lw=.8, color='#007FFF', label='y = 1.5x')
        axes[i].set_ylim((0., lims))
        axes[i].set_xlim((0., lims))
        axes[i].set_title(f'Test cohort: {test_cohort}')

        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].get_legend().remove()

        g.set_xlabel(None)
        g.set_ylabel(None)

    # Create legend on the right side using the last column of GridSpec
    legend_ax = fig.add_subplot(gs[:, -1])  # Span both rows
    legend_ax.axis('off')

    # Add legends
    legend_1 = legend_ax.legend(handles[:-(2 + int(iargs.add_line))],
                                labels[:-(2 + int(iargs.add_line))],
                                title='Phenotype category',
                                loc='center',
                                bbox_to_anchor=(0.25, 0.75))
    legend_1.get_frame().set_linewidth(0.0)
    legend_ax.add_artist(legend_1)

    legend_2 = legend_ax.legend(handles=handles[-(2 + int(iargs.add_line)):],
                                labels=labels[-(2 + int(iargs.add_line)):],
                                title='Reference lines',
                                loc='center',
                                bbox_to_anchor=(0.25, 0.3))
    legend_2.get_frame().set_linewidth(0.0)
    legend_ax.add_artist(legend_2)

    # Add labels
    fig.supxlabel("HapMap3+ Incremental $R^2$")
    fig.supylabel("MAF > 0.001 (13m variants) Incremental $R^2$")
    fig.suptitle("Out-of-sample prediction accuracy (HapMap3+ vs. 13m variants)",
                 horizontalalignment='center')

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panukb_accuracy_hapmap_vs_maf001.{iargs.extension}'))
    plt.close()


def plot_prediction_accuracy_maf001_vs_mac20(iargs):

    # Set the seaborn context:
    sns.set_context("paper", font_scale=1.25)

    # Extract the data:
    eval_df = extract_aggregate_evaluation_metrics()
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR')]
    pivoted_df = pivot_evaluation_df(eval_df, metric=iargs.metric)

    pivoted_df['Test cohort'] = pivoted_df['Test_cohort'].map(
        {'ukbb': 'UKB', 'cartagene': 'CARTaGENE'}) + '-' + pivoted_df['Test_pop']

    test_cohorts = ['CARTaGENE-EUR', 'UKB-AMR', 'UKB-MID', 'UKB-CSA', 'UKB-EAS', 'UKB-AFR']

    if iargs.metric + '_err' in eval_df.columns:
        x_col = iargs.metric + '_hq_imputed_variants_maf001'
        y_col = iargs.metric + '_hq_imputed_variants'
        xerr = iargs.metric + '_err_hq_imputed_variants_maf001'
        yerr = iargs.metric + '_err_hq_imputed_variants'
    else:
        x_col = 'hq_imputed_variants_maf001'
        y_col = 'hq_imputed_variants'
        xerr = yerr = None

    # Create figure with extra space on the right for legend
    fig = plt.figure(figsize=(12.5, 7))  # Increased width to accommodate legend

    # Create GridSpec to manage subplot layout
    gs = fig.add_gridspec(2, 4, width_ratios=(2, 2, 2, 1))  # 2 rows, 4 columns (3 for plots, 1 for legend)

    # Create axes for the plots (2x3)
    axes = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    for i, test_cohort in enumerate(test_cohorts):
        subset = pivoted_df.loc[pivoted_df['Test cohort'] == test_cohort]
        g = sns.scatterplot(data=subset,
                            x=x_col,
                            y=y_col,
                            hue='general_category',
                            palette=get_phenotype_category_palette(),
                            ax=axes[i])

        legend = g.get_legend()

        if iargs.add_scatter_errorbars and xerr is not None:
            hue_categories = [text.get_text() for text in legend.get_texts()]
            colors = [handle.get_color() for handle in legend.legend_handles]

            for category, color in zip(hue_categories, colors):
                axes[i].errorbar(x=x_col,
                                 y=y_col,
                                 xerr=xerr,
                                 yerr=yerr,
                                 data=subset.loc[subset.general_category == category],
                                 linestyle='None', label=None, capsize=2, capthick=0.5,
                                 lw=.5, alpha=.65, color=color)

        lims = max(max(axes[i].get_ylim()), max(axes[i].get_xlim()))
        x = np.linspace(0, lims, 1000)
        l1 = g.plot(x, x, ls='--', lw=.8, color='grey', label='y = x')
        l2 = g.plot(x, 1.5 * x, ls='--', lw=.8, color='#007FFF', label='y = 1.5x')
        axes[i].set_ylim((0., lims))
        axes[i].set_xlim((0., lims))
        axes[i].set_title(f'Test cohort: {test_cohort}')

        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].get_legend().remove()

        g.set_xlabel(None)
        g.set_ylabel(None)

    # Create legend on the right side using the last column of GridSpec
    legend_ax = fig.add_subplot(gs[:, -1])  # Span both rows
    legend_ax.axis('off')

    # Add legends
    legend_1 = legend_ax.legend(handles[:-(2 + int(iargs.add_line))],
                                labels[:-(2 + int(iargs.add_line))],
                                title='Phenotype category',
                                loc='center',
                                bbox_to_anchor=(0.25, 0.75))
    legend_1.get_frame().set_linewidth(0.0)
    legend_ax.add_artist(legend_1)

    legend_2 = legend_ax.legend(handles=handles[-(2 + int(iargs.add_line)):],
                                labels=labels[-(2 + int(iargs.add_line)):],
                                title='Reference lines',
                                loc='center',
                                bbox_to_anchor=(0.25, 0.3))
    legend_2.get_frame().set_linewidth(0.0)
    legend_ax.add_artist(legend_2)

    # Add labels
    fig.supxlabel("MAF > 0.001 (13m variants) Incremental $R^2$")
    fig.supylabel("MAC > 20 (18m variants) Incremental $R^2$")
    fig.suptitle("Out-of-sample prediction accuracy (13m vs. 18m variants)",
                 horizontalalignment='center')

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panukb_accuracy_maf001_vs_mac20.{iargs.extension}'))
    plt.close()


def plot_accuracy_int8_vs_int16(iargs):

    # Set the seaborn context:
    sns.set_context("paper", font_scale=1.25)

    dfs = []

    variant_set_name = {
        'hq_imputed_variants_hm3': 'HapMap3+',
        'hq_imputed_variants_maf001': 'MAF > 0.001 (13m)',
        'hq_imputed_variants': 'MAC > 20 (18m)'
    }

    for variant_set, vs_name in variant_set_name.items():

        eval_df = extract_aggregate_evaluation_metrics(variant_set=variant_set, ld_estimator='*')
        eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR')]

        # Remove training data:
        eval_df = eval_df.loc[~ ((eval_df['Test_pop'].isin(['EUR', 'all'])) &
                                 (eval_df['Test_cohort'] == 'ukbb'))]

        eval_df = pivot_evaluation_df(eval_df, metric=args.metric, columns='LD_datatype')
        eval_df['Variant set'] = vs_name
        eval_df['Test cohort'] = eval_df['Test_cohort'].map(
            {'ukbb': 'UKB', 'cartagene': 'CARTaGENE'}) + '-' + eval_df['Test_pop']

        dfs.append(eval_df)

    eval_df = pd.concat(dfs)

    if iargs.metric + '_err' in eval_df.columns:
        x_col = iargs.metric + '_int8'
        y_col = iargs.metric + '_int16'
        xerr = iargs.metric + '_err_int8'
        yerr = iargs.metric + '_err_int16'
    else:
        x_col = iargs.metric + '_int8'
        y_col = iargs.metric + '_int16'
        xerr = yerr = None

    plt.figure(figsize=(12, 6))

    g = sns.FacetGrid(eval_df, col="Variant set", sharey=True, sharex=True)
    g.map(sns.scatterplot, x_col, y_col, alpha=.5)
    g.add_legend()

    for ax in g.axes_dict.values():
        ax.axline((0, 0), slope=1., color='grey', ls="--", zorder=0)
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    g.set_xlabels("int8 Incremental $R^2$")
    g.set_ylabels("int16 Incremental $R^2$")
    g.fig.suptitle("Out-of-sample prediction accuracy with different LD data types (int16 vs int8)",
                   horizontalalignment='center')

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panukb_accuracy_int8_vs_int16.{iargs.extension}'))
    plt.close()


def plot_stratified_performance_metrics(iargs, dtype='int16'):

    sns.set_context("paper", font_scale=1.25)

    prof_metrics = extract_aggregate_performance_metrics(ld_estimator=f'block_{dtype}_mi',
                                                         model='VIPRS_EM')

    runtime_data = prof_metrics.drop(columns=
                                     ['LDEstimator', 'phenocode', 'Model',
                                      'Peak_Memory_MB', 'Total_WallClockTime']
                                     ).groupby('Variant_set').mean()
    runtime_data['Total_FitTime'] /= 60.
    runtime_data['Total_LoadTime'] /= 60.
    runtime_data['DataPrep_Time'] /= 60.
    runtime_data.rename(columns={
        'Total_FitTime': 'Inference',
        'Total_LoadTime': 'Loading LD Matrix',
        'DataPrep_Time': 'Data preparation'
    }, inplace=True)

    runtime_data = runtime_data.loc[
        ['hq_imputed_variants_hm3', 'hq_imputed_variants_maf001', 'hq_imputed_variants'],
        ['Data preparation', 'Loading LD Matrix', 'Inference']
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3), sharey=True)

    runtime_data.plot(kind='barh', stacked=True, color=['#9575CD', '#4DB6AC', '#FF7F50'], ax=ax1)
    ax1.set_title("Wallclock Time (minutes)")
    ax1.grid(axis='y', visible=False)

    # Memory:
    mem_df = prof_metrics[['Variant_set', 'Peak_Memory_MB']].copy()
    mem_df['Peak_Memory_GB'] = mem_df['Peak_Memory_MB'] / 1024

    ax = sns.barplot(mem_df, y='Variant_set', x='Peak_Memory_GB',
                     order=['hq_imputed_variants_hm3', 'hq_imputed_variants_maf001', 'hq_imputed_variants'],
                     hue='Variant_set',
                     palette={
                         'hq_imputed_variants_hm3': '#87CEEB',
                         'hq_imputed_variants_maf001': '#20B2AA',
                         'hq_imputed_variants': '#008080'
                     },
                     ax=ax2,
                     width=0.5)
    ax1.set_yticklabels(['HapMap3+', 'MAF > 0.001 (13m)', 'MAC > 20 (18m)'])
    ax1.set_ylabel("Variant Set")
    ax2.set_title("Peak Memory (GB)")
    ax2.set_xlabel('')

    plt.savefig(osp.join(iargs.output_dir, f'{dtype}_performance.{iargs.extension}'), bbox_inches="tight")
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot the panels of Figure 3.')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='figures/supp_figures/',
                        help='The directory where to store the figure panels.')
    parser.add_argument('--extension', dest='extension', type=str,
                        default='eps',
                        help='The extension for the output figure files.')
    parser.add_argument('--add-scatter-errorbars', dest='add_scatter_errorbars', action='store_true',
                        default=False,
                        help='Add errorbars to the scatter plot.')
    parser.add_argument('--add-line', dest='add_line', action='store_true',
                        default=False,
                        help='Add best fit lines to the scatterplot.')
    parser.add_argument('--metric', dest='metric', type=str,
                        default='Incremental_R2',
                        help='The metric to plot.')
    args = parser.parse_args()

    # Create the output directory if it does not exist:
    makedir(args.output_dir)
    makedir(osp.join(args.output_dir, 'figure_data'))

    sns.set_style("whitegrid")

    plot_training_r2_improvement(args)
    plot_training_r2_improvement(args, model='pathwise_VIPRS_GS')
    plot_prediction_accuracy_hapmap_vs_maf001(args)
    plot_prediction_accuracy_maf001_vs_mac20(args)
    plot_accuracy_int8_vs_int16(args)
    plot_stratified_performance_metrics(args, dtype='int8')
    plot_stratified_performance_metrics(args)
