"""
Author: Shadi Zabad
Date: April 2024
"""

import os.path as osp
import glob
from magenpy.utils.system_utils import makedir
import pandas as pd
import functools
print = functools.partial(print, flush=True)


# ----------- Options -----------

# *** Variant options ***:

min_info_score = 0.3

sample_keep_file = "data/keep_files/ukbb_qc_individuals.keep"
variant_keep_file = "data/keep_files/ukbb_qc_variants.keep"
variant_hm3_keep_file = "data/keep_files/ukbb_qc_variants_hm3.keep"


# -------- Sample quality control --------
# Read the sample QC file from the UKBB archive
print("> Extracting individuals with white British ancestry...")

ind_list = pd.read_csv("/lustre03/project/6004777/projects/uk_biobank/lists/ukb_sqc_v2_fullID_head.txt",
                       sep=r"\s+")

# Apply the standard filters:

ind_list = ind_list.loc[
    (ind_list['IID'] > 0) &  # Remove redacted samples
    (ind_list['excess.relatives'] == 0) &  # Remove samples with excess relatives
    (ind_list['putative.sex.chromosome.aneuploidy'] == 0)  # Remove samples with sex chr aneuploidy
]


# Write the list of remaining individuals to file:
makedir(osp.dirname(sample_keep_file))
ind_list[['FID', 'IID']].to_csv(sample_keep_file, sep="\t", header=False, index=False)

# --------------------------------------------
# ---------------- Variant QC ----------------
# --------------------------------------------

print("> Performing variant filtering and selection...")

info_files = "/lustre03/project/6004777/projects/uk_biobank/imputed_data/full_UKBB/v3_snp_stats/ukb_mfi_chr*_v3.txt"

# Read the long-range LD regions dataframe:
lr_ld_df = pd.read_csv("metadata/long_range_ld.txt", sep=r"\s+")

# Read the SNP information:
v_dfs = []
for f in glob.glob(info_files):

    try:
        vdf = pd.read_csv(f, sep="\t",
                          names=['ID', 'SNP', 'POS', 'A1', 'A2', 'MAF', 'MinorAllele', 'INFO'])

        # Extract the chromosome information from the file name:
        chrom = int(f.split('_')[-2].replace('chr', ''))
        vdf['CHR'] = chrom
        v_dfs.append(vdf)
    except Exception as e:
        print(e)
        continue

variant_df = pd.concat(v_dfs)

# Exclude all SNPs with duplicate IDs:
# IMPORTANT: This must be done before any filtering!
variant_df = variant_df.drop_duplicates(subset='SNP', keep=False)

# Exclude variants with imputation score less than `min_info_score`
variant_df = variant_df.loc[variant_df['INFO'] >= min_info_score]

# Exclude all SNPs with ambiguous strand (A/T or G/C):
variant_df = variant_df.loc[
    ~(
        ((variant_df['A1'] == 'A') & (variant_df['A2'] == 'T')) |
        ((variant_df['A1'] == 'T') & (variant_df['A2'] == 'A')) |
        ((variant_df['A1'] == 'G') & (variant_df['A2'] == 'C')) |
        ((variant_df['A1'] == 'C') & (variant_df['A2'] == 'G'))
    )
]

# Exclude SNPs in long-range LD regions:
snp_lr_ld = variant_df.merge(lr_ld_df, on='CHR')
snp_lr_ld = snp_lr_ld.loc[(snp_lr_ld['POS'] >= snp_lr_ld['StartPosition']) &
                          (snp_lr_ld['POS'] <= snp_lr_ld['EndPosition']), ['SNP']]
snp_lr_ld = snp_lr_ld.drop_duplicates(subset='SNP', keep=False)

variant_df = variant_df.loc[~variant_df['SNP'].isin(snp_lr_ld['SNP'])]

# Write to file:
makedir(osp.dirname(variant_keep_file))
variant_df['SNP'].to_csv(variant_keep_file, header=False, index=False)
