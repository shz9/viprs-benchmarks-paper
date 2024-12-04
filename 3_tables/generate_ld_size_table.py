import glob
import os.path as osp
import magenpy as mgp
import pandas as pd
import subprocess


def get_dir_size_bytes(path):
    result = subprocess.check_output(['du', '-sb', path])
    return int(result.decode().split()[0])


def estimate_ld_matrix_size(path, units='GB'):

    n_snps = 0

    for ld_path in glob.glob(osp.join(path, "chr_*")):

        mat = mgp.LDMatrix.from_path(ld_path)
        n_snps += mat.n_snps

    m_size_bytes = get_dir_size_bytes(path)

    if units == 'GB':
        norm = 1e9
    elif units == 'MB':
        norm = 1e6
    else:
        norm = 1.
        units = 'Bytes'

    return {
        'Number of variants': n_snps,
        'LD Matrix Storage Size': f"{m_size_bytes / norm:.2f} {units}"
    }


def construct_ld_size_table(prefix='data/ld_xarray', storage_dtype='int16', estimator='block'):

    rows = []

    variant_set_map = {
        'hq_imputed_variants': 'MAC > 20',
        'hq_imputed_variants_hm3': 'HapMap3+',
        'hq_imputed_variants_maf001': 'MAF > 0.1%'
    }

    for path in glob.glob(osp.join(prefix, f"*/*/{estimator}/{storage_dtype}/")):

        variant_set, pop = path.split("/")[2:4]
        rows.append({
            'Ancestry group': pop,
            'Variant set': variant_set_map[variant_set]
        })

        rows[-1].update(estimate_ld_matrix_size(path))

    return pd.DataFrame(rows).sort_values(['Variant set', 'Ancestry group'])


if __name__ == '__main__':

    main_ld_mat = construct_ld_size_table()
    main_ld_mat.to_csv("tables/ld_matrix_size_int16.csv", index=False)
    main_ld_mat.to_latex("tables/ld_matrix_size_int16.tex",
                         index=False, float_format="{:.3f}".format)

    ld_mat_int8 = construct_ld_size_table(storage_dtype='int8')
    ld_mat_int8.to_csv("tables/ld_matrix_size_int8.csv", index=False)
    ld_mat_int8.to_latex("tables/ld_matrix_size_int8.tex",
                         index=False, float_format="{:.3f}".format)

    ld_mat_windowed_int8 = construct_ld_size_table(prefix='data/ld', storage_dtype='int8', estimator='windowed')
    ld_mat_windowed_int8.to_csv("tables/supp/ld_matrix_size_windowed_int8.csv", index=False)
    ld_mat_windowed_int8.to_latex("tables/supp/ld_matrix_size_windowed_int8.tex",
                                  index=False, float_format="{:.3f}".format)
