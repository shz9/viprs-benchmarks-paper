import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from magenpy.utils.system_utils import makedir
from utils import *


def plot_relative_improvement(iargs):
    """
    Plot panel C of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_relative_improvement_data()

    # Save the figure data:
    df.to_csv(osp.join(iargs.output_dir, 'figure_data', 'e_step_improvements.csv'), index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Change', y='Improvement', data=df, color='salmon')

    # Change the angle for the x-labels to 45 degrees:
    plt.xticks(rotation=45, ha='right')

    plt.title("Improvement in Runtime-per-Iteration over v0.0.4 (CHR 1)")
    plt.ylabel("Median Fold Improvement")
    plt.xlabel("Incremental changes (left -> right)")
    # plt.title("Fold runtime improvements in Coordinate Ascent Step")

    plt.savefig(osp.join(iargs.output_dir, f'relative_improvement_e_step.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_accuracy_by_ld_mode(iargs):
    """
    Plot panel D of Figure 2.
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_accuracy_metrics(ld_datatype=None, ld_mode=None, threads=None)

    # Save the figure data:
    df.to_csv(osp.join(iargs.output_dir, 'figure_data', 'accuracy_by_ld_mode.csv'), index=False)

    # Create two sub-panels (2 columns, 1 row) with the left sub-panel showing
    # symmetric LD mode and the right sub-panel showing triangular LD mode:

    fig, axs = plt.subplots(ncols=2, figsize=(15, 6), sharey=True,
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    # First show the accuracy of v0.0.4 as its own bar:
    sns.barplot(x='Model', y='R-Squared', data=df.loc[df.Model == 'v0.0.4'], color='skyblue', ax=axs[0])

    # Then show the accuracy of v0.1 as a function of threads:
    sns.barplot(x='LD Data Type', y='R-Squared',
                data=df.loc[(df.Model == 'v0.1') & (df['LD Mode'] == 'Symmetric LD')],
                order=['float64', 'float32', 'int16', 'int8'],
                hue='Threads', ax=axs[0])

    # Then show the data for triangular LD:
    sns.barplot(x='LD Data Type', y='R-Squared',
                data=df.loc[(df.Model == 'v0.1') & (df['LD Mode'] == 'Triangular LD')],
                order=['float64', 'float32', 'int16', 'int8'],
                hue='Threads', ax=axs[1])

    # Change the angle for the x-labels to 45 degrees:
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45)

    axs[0].set_title("Symmetric LD")
    axs[1].set_title("Triangular LD")

    axs[0].set_xlabel("Model / LD Data Type")
    axs[1].set_xlabel("LD Data Type")

    axs[0].set_ylabel("Prediction R-Squared")

    plt.savefig(osp.join(iargs.output_dir, f'accuracy_by_ld_mode.{iargs.extension}'), bbox_inches="tight")
    plt.close()


def plot_e_step_runtime_all_models(iargs):
    """
    :param iargs: The commandline arguments captured by the argparse parser.
    """

    df = extract_e_step_stats(model=None, threads=None, aggregate=False)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 6), sharey=True,
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    for i, model in enumerate(df.Model.unique()):

        df_model = df.loc[df.Model == model]

        df_model_old = df_model.loc[df_model['ModelVersion'] == 'v0.0.4']
        df_model_old = df_model_old.groupby(['Model', 'ModelVersion', 'Chromosome', 'n_snps', 'Threads']).agg(
            {'TimePerIteration': 'mean'}
        ).reset_index()
        df_model_old['Threads'] = 'v0.0.4'

        sns.lineplot(data=df_model_old,
                     x='n_snps', y='TimePerIteration',
                     hue='Threads',
                     linewidth=3,
                     marker='o',
                     palette={'v0.0.4': 'skyblue'},
                     markersize=7,
                     ax=axs[i])

        df_model_new = df_model.loc[(df_model['ModelVersion'] == 'v0.1')]
        df_model_new = df_model_new.groupby(['Model', 'ModelVersion', 'Chromosome', 'n_snps', 'Threads']).agg(
            {'TimePerIteration': 'mean'}
        ).reset_index()

        sns.lineplot(data=df_model_new,
                     x='n_snps', y='TimePerIteration', hue='Threads',
                     linewidth=3,
                     marker='o',
                     markersize=7,
                     ax=axs[i])
        if i == 0:
            axs[i].set_ylabel("Time per Iteration (s)")
        else:
            axs[i].set_ylabel(None)

        if i == 1:
            axs[i].set_xlabel("Variants per Chromosome")
        else:
            axs[i].set_xlabel(None)

        if model == 'VIPRSGrid':
            model = 'VIPRSGrid(J=100)'
        axs[i].set_title(model, pad=10)
        axs[i].set_yscale('log')

    # Create a shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Version / Threads', loc='center right', bbox_to_anchor=(1.08, 0.5))

    # Remove individual legends from each subplot
    for ax in axs:
        ax.legend().set_visible(False)

    plt.savefig(osp.join(iargs.output_dir, f'runtime_per_iteration_all_models.{iargs.extension}'),
                bbox_inches="tight")
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

    # Set seaborn context:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=2)

    # Create the output directory if it does not exist:
    makedir(args.output_dir)
    makedir(osp.join(args.output_dir, 'figure_data'))

    plot_relative_improvement(args)
    plot_accuracy_by_ld_mode(args)
    plot_e_step_runtime_all_models(args)
