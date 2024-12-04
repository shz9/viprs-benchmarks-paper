#!/bin/bash

source env/viprs/bin/activate

python 2_plotting/figure_2.py
python 2_plotting/figure_3.py
python 2_plotting/figure_4.py

python 2_plotting/supp/crossbiobank_phenotype_dist.py
python 2_plotting/supp/plot_benchmark_supp.py
python 2_plotting/supp/plot_ld_eigenvalues.py
python 2_plotting/supp/plot_panukb_accuracy.py
