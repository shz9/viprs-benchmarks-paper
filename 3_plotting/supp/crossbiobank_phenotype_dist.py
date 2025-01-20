import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_phenotype_metadata


def get_shared_files(dir1, dir2):
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    return list(files1.intersection(files2))


def plot_distributions(dir1, dir2, shared_files):

    phenos = load_phenotype_metadata()
    pheno_map = dict(zip(phenos.phenocode.astype(str), phenos.description))

    # Calculate the grid size
    n_files = len(shared_files)
    n_cols = 6
    import math
    n_rows = math.ceil(n_files / n_cols)

    # Create a large figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 3*n_rows))

    # Flatten the axes array for easier indexing
    axes = axes.flatten() if n_files > 1 else [axes]

    for i, file in enumerate(sorted(shared_files)):
        df1 = pd.read_csv(osp.join(dir1, file), header=None, sep='\t')
        df2 = pd.read_csv(osp.join(dir2, file), header=None, sep='\t')

        ax = axes[i]
        ax.hist(df1[2], bins=30, alpha=0.5, density=True, label='UKB')
        ax.hist(df2[2], bins=30, alpha=0.5, density=True, label='CARTaGENE')

        pheno_code = file.replace('.txt', '')
        pheno_desc = pheno_map[pheno_code]

        ax.set_title(f'{pheno_desc}\n({pheno_code})', fontsize=10)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('figures/supp_figures/crossbiobank_phenotype_distribution.pdf',
                bbox_inches='tight')
    plt.close()


def main():
    dir_1 = 'data/phenotypes/ukbb/'
    dir_2 = 'data/phenotypes/cartagene/'

    shared_files = get_shared_files(dir_1, dir_2)

    if not shared_files:
        print("No shared files found between the directories.")
        return

    print("Number of shared files:", len(shared_files))

    sns.set_context("paper", font_scale=1)

    plot_distributions(dir_1, dir_2, shared_files)
    print(f"Created {len(shared_files)} distribution comparison plots.")


if __name__ == "__main__":
    main()
