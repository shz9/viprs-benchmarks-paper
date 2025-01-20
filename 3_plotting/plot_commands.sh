#!/bin/bash

mkdir -p figures
mkdir -p figures/supp_figures/

source env/viprs/bin/activate

python 3_plotting/figure_2.py
python 3_plotting/figure_3.py
python 3_plotting/figure_4.py

python 3_plotting/supp/plot_benchmark_supp.py
python 3_plotting/supp/plot_ld_eigenvalues.py
python 3_plotting/supp/plot_panukb_analysis_supp.py
python 3_plotting/supp/crossbiobank_phenotype_dist.py
