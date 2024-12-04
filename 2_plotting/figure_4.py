import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from magenpy.utils.system_utils import makedir
import os.path as osp
from utils import (
    extract_aggregate_evaluation_metrics,
    extract_aggregate_performance_metrics,
    pivot_evaluation_df,
    get_phenotype_category_palette
)


def plot_panel_c(iargs):

    # Extract the data:

    eval_df = extract_aggregate_evaluation_metrics(test_cohort='ukbb')
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR')]
    pivoted_df = pivot_evaluation_df(eval_df, metric=iargs.metric)

    test_pops = ['AFR', 'AMR', 'CSA', 'EAS', 'MID']

    if iargs.metric + '_err' in eval_df.columns:
        x_col = iargs.metric + '_hq_imputed_variants_hm3'
        y_col = iargs.metric + '_hq_imputed_variants_maf001'
        xerr = iargs.metric + '_err_hq_imputed_variants_hm3'
        yerr = iargs.metric + '_err_hq_imputed_variants_maf001'
    else:
        x_col = 'hq_imputed_variants_hm3'
        y_col = 'hq_imputed_variants_maf001'
        xerr = yerr = None

    # Generate the figure with its subplots:

    fig, axes_mat = plt.subplots(3, 2, figsize=(7.5, 9))
    axes = axes_mat.flatten()

    for i, test_pop in enumerate(test_pops):

        subset = pivoted_df.loc[pivoted_df.Test_pop == test_pop]
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
        l2 = g.plot(x, 2. * x, ls='--', lw=.8, color='#007FFF', label='y = 2x')
        axes[i].set_ylim((0., lims))
        axes[i].set_xlim((0., lims))
        axes[i].set_title(f'Test population: {test_pop}')

        handles, labels = axes[i].get_legend_handles_labels()

        axes[i].get_legend().remove()

        g.set_xlabel(None)
        g.set_ylabel(None)

    # Create a legend in the empty subplot space
    legend_ax = fig.add_subplot(3, 2, 6)
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
                                bbox_to_anchor=(1.1, 0.5))
    legend_2.get_frame().set_linewidth(0.0)
    legend_ax.add_artist(legend_2)

    fig.supxlabel("HapMap3+ Incremental $R^2$")
    fig.supylabel("MAF > 0.001 (13m variants) Incremental $R^2$")
    fig.suptitle(r'$\bf{(c)}$' + " Cross-population validation (Pan-UKB)", horizontalalignment='left')

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panel_c.{iargs.extension}'))
    plt.close()


def plot_panel_d(iargs):
    # Extract the data:

    sns.set_context("paper", font_scale=1.)

    eval_df = extract_aggregate_evaluation_metrics(test_cohort='cartagene')
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR') &
                          (eval_df['Test_pop'] == 'EUR')]
    pivoted_df = pivot_evaluation_df(eval_df, metric=iargs.metric)

    if iargs.metric + '_err' in eval_df.columns:
        x_col = iargs.metric + '_hq_imputed_variants_hm3'
        y_col = iargs.metric + '_hq_imputed_variants_maf001'
        xerr = iargs.metric + '_err_hq_imputed_variants_hm3'
        yerr = iargs.metric + '_err_hq_imputed_variants_maf001'
    else:
        x_col = 'hq_imputed_variants_hm3'
        y_col = 'hq_imputed_variants_maf001'
        xerr = yerr = None

    fig, ax = plt.subplots(figsize=(4.5, 4))

    g = sns.scatterplot(data=pivoted_df,
                        x=x_col,
                        y=y_col,
                        hue='general_category',
                        palette=get_phenotype_category_palette(),
                        s=40,
                        ax=ax)

    legend = g.get_legend()

    if iargs.add_scatter_errorbars and xerr is not None:
        hue_categories = [text.get_text() for text in legend.get_texts()]
        colors = [handle.get_color() for handle in legend.legend_handles]

        for category, color in zip(hue_categories, colors):
            ax.errorbar(x=x_col,
                        y=y_col,
                        xerr=xerr,
                        yerr=yerr,
                        data=pivoted_df.loc[pivoted_df.general_category == category],
                        linestyle='None', label=None, capsize=2, capthick=0.5,
                        lw=.5, alpha=.65, color=color)

    lims = max(max(ax.get_ylim()), max(ax.get_xlim()))
    x = np.linspace(0, lims, 1000)
    g.plot(x, x, ls='--', lw=.8, color='grey', label='y = x')
    g.plot(x, 2. * x, ls='--', lw=.8, color='#007FFF', label='y = 2x')
    g.set_ylim((0., lims))
    g.set_xlim((0., lims))
    g.set_title('Test Cohort: CARTaGENE-EUR')

    legend.remove()
    #g.legend(title='Phenotype category', loc='center left', bbox_to_anchor=(1, 0.5))

    g.set_xlabel("HapMap3+ Incremental $R^2$")
    g.set_ylabel("MAF > 0.001 (13m variants) Incremental $R^2$")

    fig.suptitle(r'$\bf{(d)}$' + " Cross-biobank validation (UKB → CARTaGENE)", horizontalalignment='left')

    plt.savefig(osp.join(iargs.output_dir, f'panel_d.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_panel_e(iargs):

    def add_error_bars(ax, df, hue_order, metric='Incremental_R2'):

        num_hues = len(hue_order)
        dodge_dists = np.linspace(-0.4, 0.4, 2 * num_hues + 1)[1::2]
        ordered_groups = [l.get_text() for l in ax.get_xticklabels()]
        # Are there better ways to do the same thing?
        for i, hue in enumerate(hue_order):
            dodge_dist = dodge_dists[i]
            df_hue = df.loc[df['Variant Set'] == hue].copy()

            df_hue['ordered_group'] = df_hue["Test cohort"].map(
                dict(zip(ordered_groups, range(len(ordered_groups))))
            )
            df_hue = df_hue.sort_values('ordered_group')
            bars = ax.errorbar(data=df_hue, x='Test cohort', y=metric,
                               yerr=f'{metric}_err', ls='',
                               lw=0.75, color='black')
            xys = bars.lines[0].get_xydata()
            bars.remove()
            ax.errorbar(data=df_hue, x=xys[:, 0] + dodge_dist, y=metric,
                        yerr=f'{metric}_err', ls='',
                        lw=0.75, color='black')

    sns.set_context("paper", font_scale=1.5)
    eval_df = extract_aggregate_evaluation_metrics()
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR')]

    eval_df['Test cohort'] = eval_df['Test_cohort'].map(
        {'ukbb': 'UKB', 'cartagene': 'CARTaGENE'}) + '-' + eval_df['Test_pop']
    eval_df = eval_df.loc[eval_df.description.isin(['Sitting height', 'Standing height']) &
                          eval_df['Test cohort'].isin(['UKB-AFR', 'UKB-AMR', 'UKB-CSA',
                                                       'UKB-EAS', 'UKB-MID', 'CARTaGENE-EUR'])]

    eval_df['Variant Set'] = eval_df['Variant_set'].map({
        'hq_imputed_variants': 'MAC > 20 (18m)',
        'hq_imputed_variants_maf001': 'MAF > 0.001 (13m)',
        'hq_imputed_variants_hm3': 'HapMap3+'
    })

    g = sns.catplot(data=eval_df, kind='bar', x='Test cohort', y=iargs.metric, row='description',
                    hue_order=['HapMap3+', 'MAF > 0.001 (13m)', 'MAC > 20 (18m)'],
                    palette={
                        'HapMap3+': '#87CEEB',
                        'MAF > 0.001 (13m)': '#20B2AA',
                        'MAC > 20 (18m)': '#008080'
                    },
                    order=['CARTaGENE-EUR', 'UKB-AMR', 'UKB-MID', 'UKB-CSA', 'UKB-EAS', 'UKB-AFR'],
                    hue='Variant Set',
                    height=3,
                    aspect=7.5 / 3)

    for ax in g.axes.flat:
        pheno = ax.get_title().split('=')[-1].strip()
        ax.set_title(pheno)
        ax.set_ylabel('Incremental $R^2$')
        data = eval_df[eval_df['description'] == pheno]

        add_error_bars(ax, data, hue_order=['HapMap3+', 'MAF > 0.001 (13m)', 'MAC > 20 (18m)'], metric=iargs.metric)

    g.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.suptitle(r'$\bf{(e)}$' + ' Height prediction accuracy using all well-imputed variants',
                 y=1.05)

    plt.savefig(osp.join(iargs.output_dir, f'panel_e.{iargs.extension}'), bbox_inches="tight")
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3), sharey=True)

    runtime_data.plot(kind='barh', stacked=True, color=['#9575CD', '#4DB6AC', '#FF7F50'], ax=ax1)
    ax1.set_title(r'$\bf{(a)}$' + " Wallclock Time (minutes)", loc='left')
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
    ax2.set_title(r'$\bf{(b)}$' + " Peak Memory (GB)", loc='left')
    ax2.set_xlabel('')

    plt.savefig(osp.join(iargs.output_dir, f'panels_a_b.{iargs.extension}'), bbox_inches="tight")
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
    parser.add_argument('--metric', dest='metric', type=str, default='R2_residualized_target',
                        help='The metric to use for the plots.')
    args = parser.parse_args()

    # Set seaborn context:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.25)

    # Create the output directory if it does not exist:
    makedir(args.output_dir)
    makedir(osp.join(args.output_dir, 'figure_data'))

    plot_panels_a_b(args)
    plot_panel_c(args)
    plot_panel_d(args)
    plot_panel_e(args)
