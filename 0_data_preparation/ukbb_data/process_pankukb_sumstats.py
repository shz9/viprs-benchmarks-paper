import pandas as pd
import glob
import os

merged_df = pd.read_csv("data/sumstats/panukb_sumstats/subset_pheno_manifest.csv")

# Process the sumstats:
# Here, we read the sumstats files, apply some filters,
# split them per population, and prepare them for PRS inference with VIPRS:

merged_df['phenocode'] = merged_df['phenocode'].astype(str)

for f in glob.glob("data/sumstats/panukb_sumstats/*.tsv.bgz"):

    print("> Extracting and processing:", f)

    phenocode = os.path.basename(f).split('-')[1]
    sub_pheno_df = merged_df.loc[merged_df['phenocode'] == phenocode]

    pops = sub_pheno_df['pop'].unique()

    # Extract the columns to keep:
    cols = ['chr', 'pos', 'ref', 'alt']
    for pop in pops:
        cols += [f'af_{pop}', f'beta_{pop}', f'se_{pop}', f'low_confidence_{pop}']

    # Extract sample sizes:
    sample_sizes = {pop: sub_pheno_df[f'n_cases_{pop}'].values[0] for pop in pops}

    # Read the sumstats:
    ss_df = pd.read_csv(f, sep="\t", compression='gzip', usecols=cols)

    # Filter to only autosomes:
    ss_df = ss_df.loc[ss_df['chr'].astype(str).str.isdigit()]

    # Loop over the populations and write a separate, filtered sumstats file for each:
    for pop in pops:

        print(">>> Processing population:", pop)

        out_df = ss_df[['chr', 'pos', 'ref', 'alt',
                        f'af_{pop}', f'beta_{pop}', f'se_{pop}', f'low_confidence_{pop}']].copy()
        out_df.columns = ['CHR', 'POS', 'A2', 'A1', 'MAF', 'BETA', 'SE', 'LOW_CONF']

        # Filter out variants with low confidence:
        out_df = out_df.loc[out_df['LOW_CONF'] == False]

        # Filter out variants with missing values:
        out_df = out_df.dropna(subset=['BETA', 'SE'])

        # Set the sample size:
        out_df['N'] = sample_sizes[pop]

        out_df.drop('LOW_CONF', axis=1, inplace=True)

        print("Remaining variants:", len(out_df))

        # Write the sumstats:
        os.system(f"mkdir -p data/sumstats/panukb_sumstats/{pop}/")
        out_df.to_csv(f"data/sumstats/panukb_sumstats/{pop}/{phenocode}.sumstats.gz",
                      sep="\t", index=False)
