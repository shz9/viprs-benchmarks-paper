import numpy as np
import pandas as pd
from magenpy.utils.system_utils import makedir


fit_df = pd.read_csv("https://portals.broadinstitute.org/collaboration/giant/images/f/fa/GIANT_HEIGHT_YENGO_2022_PGS_WEIGHTS_EUR.gz",
                     sep="\t")
fit_df.rename(columns={
    'RSID': 'SNP',
    'PGS_EFFECT_ALLELE': 'A1',
    'PGS_OTHER_ALLELE': 'A2',
    'PGS_WEIGHT': 'BETA',
}, inplace=True)

fit_df['PIP'] = np.nan
fit_df['VAR_BETA'] = np.nan

makedir("data/model_fit/external/Yengo/EUR/50/")

fit_df[['CHR', 'SNP', 'POS', 'A1', 'A2', 'BETA', 'PIP', 'VAR_BETA']].to_csv(
    "data/model_fit/external/Yengo/EUR/50/Yengo_height.fit.gz", sep="\t", index=False
)
