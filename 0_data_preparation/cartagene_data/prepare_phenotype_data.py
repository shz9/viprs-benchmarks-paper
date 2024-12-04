import os.path as osp
import pandas as pd
import numpy as np
from magenpy.stats.transforms.phenotype import detect_outliers
from magenpy.utils.system_utils import makedir


measures_cols_dict = {
    'file111': 'IID',
    'RES_MEASURED_FEV1': '3063',
    'RES_MEASURED_FVC': '3062',
    'RES_MEASURED_PEF': '3064',
    'RES_BODY_MASS_INDEX': '23104',
    'CALC_AVG_HEIGHT_CM': '50',
    'CALC_DIFF_SITTING_HEIGHT': '20015',
    'CALC_AVG_WAIST_CM': '48',
    'CALC_AVG_HIPS_CM': '49',
    'WEIGHT_KG': '21002',
    'RES_FAT_FREE_MASS': '23101',
    'RES_TOTAL_BODY_WATER_MASS': '23102',
    'RES_FAT_MASS': '23100',
    'RES_BODY_IMPEDANCE': '23106',
    'RES_FAT_PERCENT': '23099',
    'RES_BASAL_METABOLIC_RATE': '23105',
    'CALC_AVG_PULSE_RATE': '4194',
    'CALC_AVG_SYSTOLIC_BP': '4080',
    'CALC_RH_BEST_MUSCLE_STRENGTH': '47',
    'CALC_LH_BEST_MUSCLE_STRENGTH': '46',
}

labdata_cols_dict = {
    'FILE_111': 'IID',
    'ALBUMIN': '30600',
    'ALT': '30620',
    'CREATININE': '30700',
    'GGT': '30730',
    'HBA1C': '30750',
    'TRIG': '30870',
    'RBC': '30010',
    'PLATELET': '30080',
    'NEUTRO_N': '30140',
    'MPV': '30100',
    'LYMPHO_P': '30180',
    'MONO_P': '30190',
    'NEUTRO_P': '30200'
}

cartagene_homedir = "$HOME/projects/ctb-sgravel/cartagene/research/quebec_structure_936028/data/"
cartagene_homedir = osp.expandvars(cartagene_homedir)

makedir("data/phenotypes/cartagene/")
# --------------------------------------------------------------------------------

pheno_df = pd.read_csv(osp.join(cartagene_homedir, "old_metadata/data_Gravel936028_2.zip"),
                       usecols=list(measures_cols_dict.keys()))

pheno_df.columns = [measures_cols_dict[c] for c in pheno_df.columns]
pheno_df['FID'] = 0

# Output the data for each phenotype separately:
for pheno in pheno_df.columns:
    if pheno not in ('FID', 'IID'):
        print(f"Processing phenotype: {pheno}")
        sub_df = pheno_df[['FID', 'IID', pheno]].copy()

        # Remove missing values:
        sub_df[pheno] = np.where((sub_df[pheno] == -9) | (sub_df[pheno] == 999), np.nan, sub_df[pheno])
        if pheno == '3064':
            # Convert to L/min:
            sub_df[pheno] *= 60.

        # Remove outliers:
        sub_df[pheno] = np.where(detect_outliers(sub_df[pheno], sigma_threshold=3), np.nan, sub_df[pheno])
        sub_df.to_csv(f"data/phenotypes/cartagene/{pheno}.txt", sep="\t", header=False, index=False)

# --------------------------------------------------------------------------------
# Blood biochemistry phenotypes

ind_df = pd.read_csv(osp.join(cartagene_homedir,
                              "phenotypes/Gravel936028_5/Gravel_936028_5.genetic_codes.csv.gz"))
blood_pheno_df = pd.read_csv(osp.join(cartagene_homedir,
                             "phenotypes/Gravel936028_5/Gravel_936028_5.mesures_biochimiques.csv.gz"))


blood_pheno_df = ind_df.merge(blood_pheno_df, on='PROJECT_CODE')
blood_pheno_df.columns = [c.upper() for c in blood_pheno_df.columns]
blood_pheno_df = blood_pheno_df[list(labdata_cols_dict.keys())]
blood_pheno_df.columns = [labdata_cols_dict[c.upper()] for c in blood_pheno_df.columns]
blood_pheno_df['FID'] = 0

# Output the data for each phenotype separately:
for pheno in blood_pheno_df.columns:
    if pheno not in ('FID', 'IID'):
        print(f"Processing phenotype: {pheno}")
        sub_df = blood_pheno_df[['FID', 'IID', pheno]].copy()
        # Remove missing values:
        sub_df[pheno] = np.where((sub_df[pheno] == -9) | (sub_df[pheno] == 999), np.nan, sub_df[pheno])
        if pheno in ('30180', '30190', '30200'):
            # Convert proportions to percentages:
            sub_df[pheno] *= 100.

        if pheno == '30750':
            # Convert HBA1c from % to mmol/mol:
            sub_df[pheno] = sub_df[pheno]*100. * 10.93 - 23.5

        # Remove outliers:
        sub_df[pheno] = np.where(detect_outliers(sub_df[pheno], sigma_threshold=3), np.nan, sub_df[pheno])

        sub_df.to_csv(f"data/phenotypes/cartagene/{pheno}.txt", sep="\t", header=False, index=False)

