#!/bin/bash

# Download height sumstats (with UK Biobank removed) from the GIANT consortium:

mkdir -p data/sumstats/giant_sumstats_noukb/EUR/
mkdir -p data/sumstats/giant_sumstats_noukb/AFR/
mkdir -p data/sumstats/giant_sumstats_noukb/CSA/
mkdir -p data/sumstats/giant_sumstats_noukb/EAS/
mkdir -p data/sumstats/giant_sumstats_noukb/AMR/

source env/viprs/bin/activate

# Download sumstats for Europeans:

wget -O data/sumstats/giant_sumstats_noukb/EUR/50.sumstats.gz \
  https://portals.broadinstitute.org/collaboration/giant/images/8/8e/GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_EUR_excluding_UKB.gz

python 0_data_preparation/external_sumstats/postprocess_giant_sumstats.py data/sumstats/giant_sumstats_noukb/EUR/50.sumstats.gz

# Download sumstats for Africans:
wget -O data/sumstats/giant_sumstats_noukb/AFR/50.sumstats.gz \
  https://portals.broadinstitute.org/collaboration/giant/images/c/c5/GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_AFR_excluding_UKB.gz

python 0_data_preparation/external_sumstats/postprocess_giant_sumstats.py data/sumstats/giant_sumstats_noukb/AFR/50.sumstats.gz

# Download sumstats for South Asians:
wget -O data/sumstats/giant_sumstats_noukb/CSA/50.sumstats.gz \
  https://portals.broadinstitute.org/collaboration/giant/images/1/1d/GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_SAS_excluding_UKB.gz

python 0_data_preparation/external_sumstats/postprocess_giant_sumstats.py data/sumstats/giant_sumstats_noukb/CSA/50.sumstats.gz

# Download sumstats for East Asians:
wget -O data/sumstats/giant_sumstats_noukb/EAS/50.sumstats.gz \
  https://portals.broadinstitute.org/collaboration/giant/images/a/ad/GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_EAS_excluding_UKB.gz

python 0_data_preparation/external_sumstats/postprocess_giant_sumstats.py data/sumstats/giant_sumstats_noukb/EAS/50.sumstats.gz

# Download sumstats for Americans:
wget -O data/sumstats/giant_sumstats_noukb/AMR/50.sumstats.gz \
  https://portals.broadinstitute.org/collaboration/giant/images/b/b3/GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_HIS_excluding_UKB.gz

python 0_data_preparation/external_sumstats/postprocess_giant_sumstats.py data/sumstats/giant_sumstats_noukb/AMR/50.sumstats.gz
