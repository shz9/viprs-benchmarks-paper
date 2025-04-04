import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from magenpy.utils.system_utils import makedir
import os.path as osp
from utils import (
    extract_aggregate_evaluation_metrics,
    extract_external_evaluation_metrics,
    extract_aggregate_performance_metrics,
    extract_aggregate_performance_metrics_external,
    pivot_evaluation_df,
    get_phenotype_category_palette
)


def plot_scatter_per_cohort(dataset,
                            x_col,
                            x_label,
                            y_col,
                            y_label,
                            title=None,
                            x_err=None,
                            y_err=None):

    test_cohorts = ['CARTaGENE-EUR', 'UKB-AMR', 'UKB-MID', 'UKB-CSA', 'UKB-EAS', 'UKB-AFR']

    # Create figure with extra space on the right for legend
    fig = plt.figure(figsize=(12.5, 7))

    # Create GridSpec to manage subplot layout
    gs = fig.add_gridspec(2, 4, width_ratios=(2, 2, 2, 1))  # 2 rows, 4 columns (3 for plots, 1 for legend)

    # Create axes for the plots (2x3)
    axes = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    for i, test_cohort in enumerate(test_cohorts):

        subset = dataset.loc[dataset['Test cohort'] == test_cohort]
        g = sns.scatterplot(data=subset,
                            x=x_col,
                            y=y_col,
                            hue='general_category',
                            palette=get_phenotype_category_palette(),
                            ax=axes[i])

        legend = g.get_legend()

        if x_err is not None and y_err is not None:

            hue_categories = [text.get_text() for text in legend.get_texts()]
            colors = [handle.get_color() for handle in legend.legend_handles]

            for category, color in zip(hue_categories, colors):
                axes[i].errorbar(x=x_col,
                                 y=y_col,
                                 xerr=x_err,
                                 yerr=y_err,
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
    legend_1 = legend_ax.legend(handles[:-2],
                                labels[:-2],
                                title='Phenotype category',
                                loc='center',
                                bbox_to_anchor=(0.25, 0.75))
    legend_1.get_frame().set_linewidth(0.0)
    legend_ax.add_artist(legend_1)

    legend_2 = legend_ax.legend(handles=handles[-2:],
                                labels=labels[-2:],
                                title='Reference lines',
                                loc='center',
                                bbox_to_anchor=(0.25, 0.3))
    legend_2.get_frame().set_linewidth(0.0)
    legend_ax.add_artist(legend_2)

    # Add labels
    fig.supxlabel(x_label)
    fig.supylabel(y_label)

    if title is not None:
        fig.suptitle(title, x=0.01, horizontalalignment='left')


def plot_panel_c(iargs):
    # Extract the data:
    eval_df = extract_aggregate_evaluation_metrics()
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR')]
    pivoted_df = pivot_evaluation_df(eval_df, metric=iargs.metric)

    pivoted_df['Test cohort'] = pivoted_df['Test_cohort'].map(
        {'ukbb': 'UKB', 'cartagene': 'CARTaGENE'}) + '-' + pivoted_df['Test_pop']

    if iargs.metric + '_err' in eval_df.columns:
        x_col = iargs.metric + '_hq_imputed_variants_hm3'
        y_col = iargs.metric + '_hq_imputed_variants'
        xerr = iargs.metric + '_err_hq_imputed_variants_hm3'
        yerr = iargs.metric + '_err_hq_imputed_variants'
    else:
        x_col = 'hq_imputed_variants_hm3'
        y_col = 'hq_imputed_variants'
        xerr = yerr = None

    plot_scatter_per_cohort(pivoted_df,
                            x_col=x_col,
                            x_label="HapMap3+ Incremental $R^2$",
                            y_col=y_col,
                            y_label="MAC > 20 (18m variants) Incremental $R^2$",
                            title=r'$\bf{(c)}$' + " Out-of-sample prediction accuracy (HapMap3+ vs. 18m variants)",
                            x_err=xerr,
                            y_err=yerr)

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panel_c.{iargs.extension}'))
    plt.close()


def plot_panels_a_b(iargs):

    prof_metrics = extract_aggregate_performance_metrics()

    sns.set_context("paper", font_scale=1.25)

    runtime_data = prof_metrics.drop(columns=
                                     ['LDEstimator', 'phenocode',
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 7), sharex=True)

    runtime_data.plot(kind='bar', stacked=True, color=['#9575CD', '#4DB6AC', '#FF7F50'], ax=ax1)
    ax1.set_title(r'$\bf{(a)}$' + " Wallclock Time (minutes)", loc='left')
    ax1.grid(axis='x', visible=False)

    # Memory:
    mem_df = prof_metrics[['Variant_set', 'Peak_Memory_MB']].copy()
    mem_df['Peak_Memory_GB'] = mem_df['Peak_Memory_MB'] / 1024

    ax = sns.barplot(mem_df, y='Peak_Memory_GB', x='Variant_set',
                     order=['hq_imputed_variants_hm3', 'hq_imputed_variants_maf001', 'hq_imputed_variants'],
                     hue='Variant_set',
                     palette={
                         'hq_imputed_variants_hm3': '#87CEEB',
                         'hq_imputed_variants_maf001': '#20B2AA',
                         'hq_imputed_variants': '#008080'
                     },
                     ax=ax2,
                     width=0.5)
    ax2.set_xticklabels(['HapMap3+', 'MAF > 0.001 (13m)', 'MAC > 20 (18m)'], rotation=20)
    ax2.set_xlabel("Variant Set")
    ax2.set_title(r'$\bf{(b)}$' + " Peak Memory (GB)", loc='left')
    ax2.set_ylabel('')

    plt.savefig(osp.join(iargs.output_dir, f'panels_a_b.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_method_comparison(iargs):

    eval_df = extract_aggregate_evaluation_metrics(ld_estimator='block_int8_mi',
                                                   model='*VIPRS_*')
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR')]
    eval_df = eval_df.loc[eval_df['Model'] != 'NOLRLD_VIPRS_EM']

    eval_df['Model'] = eval_df['Model'].map({
        'VIPRS_EM': 'VIPRS v0.1 ',
        'VIPRS_GS': 'VIPRS-GS v0.1 ',
        'pathwise_VIPRS_GS': 'VIPRS-GSp v0.1 ',
    }) + eval_df['Variant_set'].map({
        'hq_imputed_variants_hm3': '(HM3)',
        'hq_imputed_variants_maf001': '(13m)',
        'hq_imputed_variants': '(18m)'
    }) + eval_df['LDEstimator'].apply(lambda x: ['', '-4cM'][x == 'block4cm_int8_mi'])

    eval_df = eval_df.loc[eval_df['Model'].isin([
        'VIPRS v0.1 (HapMap3)', 'VIPRS-GSp v0.1 (HM3)',
        'VIPRS v0.1 (13m)', 'VIPRS-GSp v0.1 (13m)',
        'VIPRS v0.1 (18m)', 'VIPRS-GSp v0.1 (18m)',
    ])]

    eval_df['Model'] = eval_df['Model'].map({
        'VIPRS v0.1 (HapMap3)': 'VIPRS v0.1 (HM3)',
        'VIPRS-GSp v0.1 (HapMap3)': 'VIPRS-GS v0.1 (HM3)',
        'VIPRS v0.1 (13m)': 'VIPRS v0.1 (13m)',
        'VIPRS-GSp v0.1 (13m)': 'VIPRS-GS v0.1 (13m)',
        'VIPRS v0.1 (18m)': 'VIPRS v0.1 (18m)',
        'VIPRS-GSp v0.1 (18m)': 'VIPRS-GS v0.1 (18m)',
    })

    eval_df_extern = extract_external_evaluation_metrics()
    eval_df_extern = eval_df_extern.loc[eval_df_extern['Model'].isin([
        'LDpred2-auto', 'VIPRS_v0.0.4', 'SBayesRC-HapMap3', 'SBayesRC-7m'
        ])]

    eval_df_extern['Model'] = eval_df_extern['Model'].map({
        'LDpred2-auto': 'LDpred2-auto (HM3)',
        'VIPRS_v0.0.4': 'VIPRS v0.0.4 (HM3)',
        'SBayesRC-HapMap3': 'SBayesRC (HM3)',
        'SBayesRC-7m': 'SBayesRC (7m)',
    })

    eval_df = pd.concat([eval_df, eval_df_extern])

    eval_df['Test cohort'] = eval_df['Test_cohort'].map(
        {'ukbb': 'UKB', 'cartagene': 'CARTaGENE'}
    ) + '-' + eval_df['Test_pop']

    model_order = ['LDpred2-auto (HM3)', 'VIPRS v0.0.4 (HM3)',
                   'VIPRS v0.1 (HM3)', 'VIPRS v0.1 (13m)', 'VIPRS v0.1 (18m)',
                   'VIPRS-GS v0.1 (HM3)', 'VIPRS-GS v0.1 (13m)', 'VIPRS-GS v0.1 (18m)',
                   'SBayesRC (HM3)', 'SBayesRC (7m)']

    palette = {
        'LDpred2-auto (HM3)': '#FA8072',  # Salmon
        'VIPRS v0.0.4 (HM3)': '#D8BFD8',  # Thistle
        'VIPRS v0.1 (HM3)': '#87CEEB',  # Light Sky Blue
        'VIPRS v0.1 (13m)': '#20B2AA',  # Light Sea Green
        'VIPRS v0.1 (18m)': '#008080',  # Teal
        'VIPRS-GS v0.1 (HM3)': '#FAFAD2',  # Light Goldenrod Yellow
        'VIPRS-GS v0.1 (13m)': '#FFDAB9',  # Peach Puff
        'VIPRS-GS v0.1 (18m)': '#FFE4E1',  # Misty Rose
        'SBayesRC (HM3)': '#98FB98',  # Pale Green
        'SBayesRC (7m)': '#b4c45a'  # olive green
    }

    # Remove the training set:
    eval_df = eval_df.loc[~eval_df['Test cohort'].isin(['UKB-EUR', 'UKB-all'])]

    # -----------------------------------
    # Plot the accuracy metrics:

    fig = plt.figure(figsize=(9, 6.5))

    ax = sns.boxplot(data=eval_df,
                     x='Test cohort',
                     hue='Model',
                     hue_order=model_order,
                     order=['CARTaGENE-EUR', 'UKB-AMR', 'UKB-MID', 'UKB-CSA', 'UKB-EAS', 'UKB-AFR'],
                     palette=palette,
                     showmeans=True,
                     boxprops=dict(linewidth=.5, edgecolor='black'),
                     meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'red',
                                'markersize': 2.5},
                     y=iargs.metric,
                     showfliers=False)

    # Rotate the x-tick labels by 30 degrees:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_title(r'$\bf{(c)}$' + " Out-of-sample prediction accuracy on Pan-UKB and CARTaGENE cohorts",
                 x=0.01, horizontalalignment='left')

    # Put legend to the right:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Incremental $R^2$')

    plt.savefig(osp.join(iargs.output_dir, f'model_comparison.{iargs.extension}'), bbox_inches="tight")
    plt.close()

    """
    # -----------------------------------
    # Generate the boxplot for the shared phenotypes only:
    import glob
    ukb_pheno = [osp.basename(f).replace('.txt', '') for f in glob.glob('data/phenotypes/ukbb/*.txt')]
    cag_pheno = [osp.basename(f).replace('.txt', '') for f in glob.glob('data/phenotypes/cartagene/*.txt')]

    shared_pheno = list(set(ukb_pheno).intersection(set(cag_pheno)))

    eval_df_shared_pheno = eval_df.loc[eval_df.phenocode.astype(str).isin(
        shared_pheno
    )]

    ax = sns.boxplot(data=eval_df_shared_pheno,
                     x='Test cohort',
                     hue='Model',
                     hue_order=model_order,
                     palette='Set2',
                     showmeans=True,
                     meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'red',
                                'markersize': 3},
                     y=iargs.metric, )

    # Rotate the x-tick labels by 30 degrees:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    # Put legend to the right:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Incremental $R^2$')

    plt.savefig(osp.join(iargs.output_dir, f'model_comparison_shared_pheno.{iargs.extension}'),
                bbox_inches="tight")
    plt.close()

    # -----------------------------------

    pairs_of_methods = [('VIPRS (HapMap3)', 'VIPRS-NOLRLD (HapMap3)'),
                        ('VIPRS (HapMap3)', 'VIPRS (HapMap3-x)'),
                        ('VIPRS (HapMap3)', 'VIPRS_v0.0.4 (HapMap3)'),
                        ('VIPRS (HapMap3)', 'SBayesRC (HapMap3-hq)'),
                        ('VIPRS (HapMap3)', 'LDpred2-auto (HapMap3)'),
                        ('VIPRS (HapMap3)', 'SBayesRC (HapMap3; orig)'),
                        ('SBayesRC (HapMap3-hq)', 'SBayesRC (HapMap3; orig)'),
                        ('SBayesRC (HapMap3-hq-4cM)', 'SBayesRC (HapMap3; orig)'),
                        ('SBayesRC (HapMap3-x-4cM)', 'SBayesRC (HapMap3; orig)'),]
    

    pairs_of_methods = [
        ('VIPRS v0.1 (HapMap3)', 'VIPRS-GS v0.1 (HapMap3)'),
        ('VIPRS-GS v0.1 (HapMap3)', 'VIPRS-GSp v0.1 (HapMap3)'),
        ('VIPRS v0.1 (HapMap3)', 'SBayesRC (HapMap3)'),
        ('VIPRS-GS v0.1 (HapMap3)', 'SBayesRC (HapMap3)'),
        ('VIPRS-GSp v0.1 (HapMap3)', 'VIPRS-GSp v0.1 (13m)'),
        ('VIPRS-GSp v0.1 (HapMap3)', 'VIPRS-GSp v0.1 (18m)'),
        ('VIPRS-GSp v0.1 (13m)', 'SBayesRC (7m)'),
    ]

    # Loop over pairs of methods and generate scatter plots:
    for method1, method2 in pairs_of_methods:

        pivoted_df = pivot_evaluation_df(eval_df.loc[eval_df['Model'].isin([method1, method2])],
                                         metric=iargs.metric, columns='Model')
        plot_scatter_per_cohort(pivoted_df,
                                x_col=iargs.metric + '_' + method1,
                                x_label=method1,
                                y_col=iargs.metric + '_' + method2,
                                y_label=method2,
                                title=f"Comparison of {method1} vs. {method2}")

        plt.tight_layout()
        plt.savefig(osp.join(iargs.output_dir, f'model_comparison_{method1}_{method2}.{iargs.extension}'))
        plt.close()
    """


def plot_method_comparison_computational(iargs):


    prof_metrics = extract_aggregate_performance_metrics(model='*VIPRS_*')
    prof_metrics = prof_metrics.loc[prof_metrics['Model'].isin(
        ['VIPRS_EM', 'pathwise_VIPRS_GS']
    )]

    prof_metrics = prof_metrics.loc[prof_metrics['Variant_set'].isin([
        'hq_imputed_variants_hm3', 'hq_imputed_variants_maf001', 'hq_imputed_variants'
    ])]

    prof_metrics['Model'] = prof_metrics['Model'].map({
        'VIPRS_EM': 'VIPRS v0.1 ',
        'pathwise_VIPRS_GS': 'VIPRS-GS v0.1 ',
    }) + prof_metrics['Variant_set'].map({
        'hq_imputed_variants_hm3': '(HM3)',
        'hq_imputed_variants_maf001': '(13m)',
        'hq_imputed_variants': '(18m)'
    })

    prof_metrics = prof_metrics[['Model', 'Total_WallClockTime', 'Peak_Memory_MB']].reset_index(drop=True)

    prof_metrics_external = extract_aggregate_performance_metrics_external()

    prof_metrics_external = prof_metrics_external[['Model', 'Total_WallClockTime', 'Peak_Memory_MB']]
    prof_metrics_external = prof_metrics_external.loc[prof_metrics_external['Model'].isin([
        'LDpred2-auto', 'VIPRS_v0.0.4', 'SBayesRC-HapMap3', 'SBayesRC-7m'
        ])]

    prof_metrics_external['Model'] = prof_metrics_external['Model'].map({
        'LDpred2-auto': 'LDpred2-auto (HM3)',
        'VIPRS_v0.0.4': 'VIPRS v0.0.4 (HM3)',
        'SBayesRC-HapMap3': 'SBayesRC (HM3)',
        'SBayesRC-7m': 'SBayesRC (7m)'
    })

    prof_metrics = pd.concat([prof_metrics, prof_metrics_external]).reset_index(drop=True)

    prof_metrics['Peak_Memory_GB'] = prof_metrics['Peak_Memory_MB'] / 1024
    prof_metrics['Total_WallClockTime'] /= 60.


    model_order = ['LDpred2-auto (HM3)', 'VIPRS v0.0.4 (HM3)',
                   'VIPRS v0.1 (HM3)', 'VIPRS v0.1 (13m)', 'VIPRS v0.1 (18m)',
                   'VIPRS-GS v0.1 (HM3)', 'VIPRS-GS v0.1 (13m)', 'VIPRS-GS v0.1 (18m)',
                   'SBayesRC (HM3)', 'SBayesRC (7m)']

    palette = {
        'LDpred2-auto (HM3)': '#FA8072',  # Salmon
        'VIPRS v0.0.4 (HM3)': '#D8BFD8',  # Thistle
        'VIPRS v0.1 (HM3)': '#87CEEB',  # Light Sky Blue
        'VIPRS v0.1 (13m)': '#20B2AA',  # Light Sea Green
        'VIPRS v0.1 (18m)': '#008080',  # Teal
        'VIPRS-GS v0.1 (HM3)': '#FAFAD2',  # Light Goldenrod Yellow
        'VIPRS-GS v0.1 (13m)': '#FFDAB9',  # Peach Puff
        'VIPRS-GS v0.1 (18m)': '#FFE4E1',  # Misty Rose
        'SBayesRC (HM3)': '#98FB98',  # Pale Green
        'SBayesRC (7m)': '#b4c45a'  # olive green
    }

    def add_bar_boundary(ax):
        for bar in ax.patches:
            bar.set_edgecolor('black')  # Set the color of the border
            bar.set_linewidth(.5)  # Set the thickness of the border

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 6.5), sharex=True)

    sns.barplot(data=prof_metrics, x='Model', y='Total_WallClockTime',
                hue='Model',
                order=model_order,
                hue_order=model_order,
                palette=palette,
                ax=ax1)
    ax1.set_ylabel('')
    ax1.grid(axis='x', visible=False)
    ax1.set_title(r'$\bf{(a)}$' + " Wallclock Time (minutes)", loc='left')
    ax1.set_yticks(np.arange(0, 280, 30))
    add_bar_boundary(ax1)

    sns.barplot(data=prof_metrics, x='Model', y='Peak_Memory_GB',
                hue='Model',
                order=model_order,
                hue_order=model_order,
                palette=palette,
                ax=ax2)
    ax2.set_ylabel('')
    ax2.set_xlabel("Model")
    ax2.set_title(r'$\bf{(b)}$' + " Peak Memory (GB)", loc='left')
    ax2.set_xlim(-1, 10)
    ax2.set_yticks(np.arange(0, 80, 10))
    add_bar_boundary(ax2)

    # Rotate the x-tick labels by 30 degrees:
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize='small')
    plt.tight_layout()

    plt.savefig(osp.join(iargs.output_dir, f'model_comparison_computational.{iargs.extension}'),
                bbox_inches="tight")
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot the panels of Figure 4.')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='figures/figure_4/',
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
    parser.add_argument('--metric', dest='metric', type=str, default='R2_residualized_target',
                        help='The metric to use for the plots.')
    args = parser.parse_args()

    # Set seaborn context:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.25)

    # Create the output directory if it does not exist:
    makedir(args.output_dir)
    makedir(osp.join(args.output_dir, 'figure_data'))

    #plot_panels_a_b(args)
    #plot_panel_c(args)
    #plot_panel_d(args)
    #plot_panel_e(args)
    plot_method_comparison(args)
    plot_method_comparison_computational(args)
