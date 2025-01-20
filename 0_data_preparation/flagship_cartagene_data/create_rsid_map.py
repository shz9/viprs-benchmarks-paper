import os.path as osp
import pandas as pd
import argparse

"""
NOTE: CARTaGENE genotype data is called based on hg38 coordinates, while the UKB data used in the remaining analyses
used hg19. The purpose of this script is to translate the variant IDs from hg38 to hg19 so that we can use the
UKB data in the CARTaGENE analysis.
"""


def main(args):

    map_dfs = []

    for chrom in range(1, 23):

        print("Processing Chromosome:", chrom)

        # Read the map table + the genotype table for the chromosome
        olink_map_df = pd.read_csv(osp.join(args.olink_map_dir,
            f"olink_rsid_map_mac5_info03_b0_7_chr{chrom}_patched_v2.tsv.gz"), sep="\t")
        pgen_var_df = pd.read_csv(osp.join(args.pgen_dir, f"chr{chrom}.imputed_metaminimac_w_CaG_TOPMed_r3.pvar"),
                                  sep=r"\s+", comment='#',
                                  names=['CHR', 'POS', 'ID', 'REF', 'ALT', 'FILTER', 'INFO'])

        # Add the variant IDs to the map table:
        olink_map_df['ID_hg38'] = f'chr{chrom}:' + olink_map_df['POS38'].astype(str) + ':' + olink_map_df['REF'] + ':' + olink_map_df['ALT']
        olink_map_df['ID_hg38_rev'] = f'chr{chrom}:' + olink_map_df['POS38'].astype(str) + ':' + olink_map_df['ALT'] + ':' + olink_map_df['REF']

        # Merge the two tables and add the result to map_dfs:
        merge_1 = olink_map_df.merge(pgen_var_df, left_on='ID_hg38', right_on='ID')[['CHR', 'rsid', 'REF_x', 'ALT_x', 'ID_hg38']]
        merge_1.columns = ['CHR', 'SNP', 'A2', 'A1', 'ID']
        map_dfs.append(merge_1)

        merge_2 = olink_map_df.merge(pgen_var_df, left_on='ID_hg38_rev', right_on='ID')[['CHR', 'rsid', 'REF_x', 'ALT_x', 'ID_hg38_rev']]
        merge_2.columns = ['CHR', 'SNP', 'A2', 'A1', 'ID']
        map_dfs.append(merge_2)

    map_dfs = pd.concat(map_dfs)
    map_dfs.to_csv("metadata/rsid_map_cartagene.csv.gz", index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create a map of variant IDs to rsIDs')
    parser.add_argument('--olink-map-dir', type=str, dest='olink_map_dir',
                        help='Path to the Olink rsid map directory',
                        default='~/projects/ctb-sgravel/data/rsid_map_ukb/')
    parser.add_argument('--genotype-pgen-dir', type=str, dest='pgen_dir',
                        help='Path to the genotype pgen directory',
                        default='~/projects/ctb-sgravel/cartagene/research/flagship_project/processed_data/flagship_array_imputed_w_cag_and_topmed_r3/PGEN/')

    args = parser.parse_args()

    main(args)
