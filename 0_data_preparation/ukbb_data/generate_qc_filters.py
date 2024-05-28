"""
Author: Shadi Zabad
Date: April 2024
"""

from magenpy.utils.system_utils import makedir
import pandas as pd
import functools
print = functools.partial(print, flush=True)


# ----------- Options -----------

# *** Variant options ***:

min_info_score = 0.8
makedir("data/keep_files/")

# -------- Sample quality control --------
# Read the sample QC file from the UKBB archive
print("> Extracting sample metadata...")

ind_list = pd.read_csv("/lustre03/project/6008063/neurohub/UKB/Returns/2442/"
                       "all_pops_non_eur_pruned_within_pop_pc_covs.tsv",
                       sep="\t")

bridge_df = pd.read_csv("/lustre03/project/6008063/neurohub/UKB/Returns/2442/ukb45551bridge31063.txt",
                        sep=r"\s+", names=['IID', 's'])

# Merge the bridge file with the individual list:
ind_list = ind_list.merge(bridge_df).drop(columns=['s'])
# Keep only unrelateds:
ind_list = ind_list.loc[~ind_list.related]
ind_list[['IID', 'IID']].to_csv("data/keep_files/ukbb_qc_individuals_all.keep", header=False, index=False)

for pop in ind_list['pop'].unique():

    pop_df = ind_list.loc[ind_list['pop'] == pop, ['IID', 'IID']]
    print("Population:", pop, "| Number of individuals:", len(pop_df))

    pop_df.to_csv(
        f"data/keep_files/ukbb_qc_individuals_{pop}.keep", header=False, index=False
    )

del ind_list

# --------------------------------------------
# ---------------- Variant QC ----------------
# --------------------------------------------

print("> Performing variant filtering and selection...")

info_files = ("/lustre03/project/6008063/neurohub/UKB/Bulk/Imputation/"
              "UKB_imputation_from_genotype/ukb22828_c{}_b0_v3.mfi.txt")

# Read the long-range LD regions dataframe:
# lr_ld_df = pd.read_csv("metadata/long_range_ld.txt", sep=r"\s+")

# Read the SNP information:
v_dfs = []
for chrom in range(1, 23):
    print("Chromosome:", chrom)

    try:
        vdf = pd.read_csv(info_files.format(chrom), sep="\t",
                          names=['ID', 'SNP', 'POS', 'A1', 'A2', 'MAF', 'MinorAllele', 'INFO'])
    except Exception as e:
        print(e)
        continue

    print("Length before filtering:", len(vdf))

    # Exclude all SNPs with duplicate IDs:
    # IMPORTANT: This must be done before any filtering!
    vdf = vdf.drop_duplicates(subset='SNP', keep=False)

    # Exclude variants with imputation score less than `min_info_score`
    # And keep variants with equivalent MAC of 20 in 500k:
    vdf = vdf.loc[(vdf['INFO'] >= min_info_score) &
                  (vdf['MAF'] >= 0.00004), ]

    # Exclude all SNPs with ambiguous strand (A/T or G/C):
    vdf = vdf.loc[
        ~(
            ((vdf['A1'] == 'A') & (vdf['A2'] == 'T')) |
            ((vdf['A1'] == 'T') & (vdf['A2'] == 'A')) |
            ((vdf['A1'] == 'G') & (vdf['A2'] == 'C')) |
            ((vdf['A1'] == 'C') & (vdf['A2'] == 'G'))
        ),
    ]

    print("Length after filtering:", len(vdf))
    print('-----')

    vdf['CHR'] = chrom
    v_dfs.append(vdf)


variant_df = pd.concat(v_dfs)
print("Extracted a total of", len(variant_df), "variants.")

del vdf, v_dfs

# Output the entire set of variants:
variant_df[['SNP']].to_csv("data/keep_files/hq_imputed_variants.txt", index=False, header=False)

# Restrict to variants that have global MAF >= 0.001:
cond = variant_df.MAF >= 0.001
print("Number of variants with MAF >= 0.001:", cond.sum())

variant_df.loc[cond][['SNP']].to_csv(
    "data/keep_files/hq_imputed_variants_maf001.csv", index=False, header=False
)

# Restrict to variants in "expanded HapMap3" set from Prive et al.:
hm3_df = pd.read_csv("metadata/hm3_plus_prive.csv.gz")
merged_df = variant_df.merge(hm3_df, left_on='SNP', right_on='rsid')
del variant_df, hm3_df
print("Number of variants in HM3 set:", len(merged_df))
merged_df[['SNP']].to_csv(
    "data/keep_files/hq_imputed_variants_hm3.csv", index=False, header=False
)
