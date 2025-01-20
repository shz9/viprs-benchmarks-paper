import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)), '44_plotting'))
from utils import load_phenotype_metadata


pheno_metadata = load_phenotype_metadata()
pheno_metadata.columns = ['Phenocode', 'Description', 'Category', 'LDSC $h_2$ (EUR)']
pheno_metadata = pheno_metadata.sort_values(['Category', 'Phenocode'])
pheno_metadata.to_excel("tables/supp/phenotype_metadata.csv", index=False)
pheno_metadata.to_latex("tables/supp/phenotype_metadata.tex", index=False, float_format="{:.3f}".format)
