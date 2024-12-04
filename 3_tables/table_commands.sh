#!/bin/bash

mkdir -p tables/supp/

source env/viprs/bin/activate
python 3_tables/generate_phenotype_table.py

