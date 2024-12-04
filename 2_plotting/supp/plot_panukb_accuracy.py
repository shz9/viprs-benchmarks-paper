import argparse
import seaborn as sns
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import extract_aggregate_evaluation_metrics, pivot_evaluation_df, get_phenotype_category_palette
import matplotlib.pyplot as plt
import numpy as np


def plot_training_r2_improvement(iargs):
    # Extract the data:

    sns.set_context("paper", font_scale=1.)

    eval_df = extract_aggregate_evaluation_metrics(test_cohort='ukbb')
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
        g.plot(x, 2. * x, ls='--', lw=.8, color='#007FFF', label='y = 2x')
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

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'training_r2.{iargs.extension}'))
    plt.close()


def plot_cartagene_improvement(iargs):
    # Extract the data:

    sns.set_context("paper", font_scale=1.)

    eval_df = extract_aggregate_evaluation_metrics(test_cohort='cartagene')
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
        g.plot(x, 2. * x, ls='--', lw=.8, color='#007FFF', label='y = 2x')
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

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'cartagene_improvement.{iargs.extension}'))
    plt.close()


def plot_panukb_cross_pop_hapmap_vs_mac20(iargs):

    # Extract the data:

    eval_df = extract_aggregate_evaluation_metrics(test_cohort='ukbb')
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR')]
    pivoted_df = pivot_evaluation_df(eval_df, metric=iargs.metric)

    test_pops = ['AFR', 'AMR', 'CSA', 'EAS', 'MID']

    if iargs.metric + '_err' in eval_df.columns:
        x_col = iargs.metric + '_hq_imputed_variants_hm3'
        y_col = iargs.metric + '_hq_imputed_variants'
        xerr = iargs.metric + '_err_hq_imputed_variants_hm3'
        yerr = iargs.metric + '_err_hq_imputed_variants'
    else:
        x_col = 'hq_imputed_variants_hm3'
        y_col = 'hq_imputed_variants'
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
    fig.supylabel("MAC > 20 (18m variants) Incremental $R^2$")

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panukb_crosspop_hapmap_vs_mac20.{iargs.extension}'))
    plt.close()


def plot_panukb_cross_pop_maf001_vs_mac20(iargs):

    # Extract the data:

    eval_df = extract_aggregate_evaluation_metrics(test_cohort='ukbb')
    eval_df = eval_df.loc[(eval_df['LD_w_MI'] == True) & (eval_df['Training_pop'] == 'EUR')]
    pivoted_df = pivot_evaluation_df(eval_df, metric=iargs.metric)

    test_pops = ['AFR', 'AMR', 'CSA', 'EAS', 'MID']

    if iargs.metric + '_err' in eval_df.columns:
        x_col = iargs.metric + '_hq_imputed_variants_maf001'
        y_col = iargs.metric + '_hq_imputed_variants'
        xerr = iargs.metric + '_err_hq_imputed_variants_maf001'
        yerr = iargs.metric + '_err_hq_imputed_variants'
    else:
        x_col = 'hq_imputed_variants_maf001'
        y_col = 'hq_imputed_variants'
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

    fig.supxlabel("MAF > 0.001 (13m variants) Incremental $R^2$")
    fig.supylabel("MAC > 20 (18m variants) Incremental $R^2$")

    plt.tight_layout()
    plt.savefig(osp.join(iargs.output_dir, f'panukb_crosspop_maf001_vs_mac20.{iargs.extension}'))
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
    parser.add_argument('--metric', dest='metric', type=str,
                        default='R2_residualized_target',
                        help='The metric to plot.')
    args = parser.parse_args()

    sns.set_style("whitegrid")

    plot_training_r2_improvement(args)
    plot_cartagene_improvement(args)
    plot_panukb_cross_pop_hapmap_vs_mac20(args)
    plot_panukb_cross_pop_maf001_vs_mac20(args)
