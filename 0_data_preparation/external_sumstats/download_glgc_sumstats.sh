#!/bin/bash

mkdir -p data/sumstats/glgc_sumstats/EUR/
mkdir -p data/sumstats/glgc_sumstats/AFR/
mkdir -p data/sumstats/glgc_sumstats/CSA/
mkdir -p data/sumstats/glgc_sumstats/EAS/
mkdir -p data/sumstats/glgc_sumstats/AMR/

source env/viprs/bin/activate

# Download sumstats for HDL Cholesterol from the Global Lipids Genetics Consortium:

# European sumstats:
wget -O data/sumstats/glgc_sumstats/EUR/30760.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_HDL_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/EUR/30760.sumstats.gz

# African sumstats:
wget -O data/sumstats/glgc_sumstats/AFR/30760.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_HDL_INV_AFR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/AFR/30760.sumstats.gz

# South Asian sumstats:

wget -O data/sumstats/glgc_sumstats/CSA/30760.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_HDL_INV_SAS_1KGP3_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/CSA/30760.sumstats.gz

# Download sumstats for LDL Cholesterol from the Global Lipids Genetics Consortium:

# European sumstats:
wget -O data/sumstats/glgc_sumstats/EUR/30780.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_LDL_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/EUR/30780.sumstats.gz

# African sumstats:
wget -O data/sumstats/glgc_sumstats/AFR/30780.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_LDL_INV_AFR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/AFR/30780.sumstats.gz

# South Asian sumstats:
wget -O data/sumstats/glgc_sumstats/CSA/30780.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_LDL_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/CSA/30780.sumstats.gz

# Download sumstats for Total Cholesterol from the Global Lipids Genetics Consortium:

# European sumstats:
wget -O data/sumstats/glgc_sumstats/EUR/30690.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_TC_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/EUR/30690.sumstats.gz

# African sumstats:
wget -O data/sumstats/glgc_sumstats/AFR/30690.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_TC_INV_AFR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/AFR/30690.sumstats.gz

# South Asian sumstats:
wget -O data/sumstats/glgc_sumstats/CSA/30690.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_TC_INV_SAS_1KGP3_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/CSA/30690.sumstats.gz

# Download sumstats for log(TG) from the Global Lipids Genetics Consortium:

# European sumstats:
wget -O data/sumstats/glgc_sumstats/EUR/30870.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_logTG_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/EUR/30870.sumstats.gz

# African sumstats:
wget -O data/sumstats/glgc_sumstats/AFR/30870.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_logTG_INV_AFR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/AFR/30870.sumstats.gz

# South Asian sumstats:
wget -O data/sumstats/glgc_sumstats/CSA/30870.sumstats.gz \
  https://csg.sph.umich.edu/willer/public/glgc-lipids2021/results/ancestry_specific/without_UKB_logTG_INV_SAS_1KGP3_ALL.meta.singlevar.results.gz

python 0_data_preparation/external_sumstats/postprocess_glgc_sumstats.py data/sumstats/glgc_sumstats/CSA/30870.sumstats.gz
