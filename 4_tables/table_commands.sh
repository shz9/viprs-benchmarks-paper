#!/bin/bash

mkdir -p tables/supp/

source env/viprs/bin/activate
python 4_tables/generate_phenotype_table.py

