#!/usr/bin/env python3

"""
Author: Shadi Zabad

This is a commandline script that enables users to perform
posterior inference for polygenic risk score models using
variational inference techniques.

NOTE: Updated to add profiler metrics (memory usage and runtime)
"""


def get_memory_usage():
    """
    Get the memory usage of the current process in Mega Bytes (MB)
    """
    import os
    import psutil

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)


def main():

    import argparse
    import viprs as vp

    print(f"""
        **********************************************
                    _____
            ___   _____(_)________ ________________
            __ | / /__  / ___  __ \__  ___/__  ___/
            __ |/ / _  /  __  /_/ /_  /    _(__  )
            _____/  /_/   _  .___/ /_/     /____/
                          /_/
        Variational Inference of Polygenic Risk Scores
        Version: {vp.__version__} | Release date: June 2022
        Author: Shadi Zabad, McGill University
        **********************************************
        < Fit VIPRS model to GWAS summary statistics >
    """)

    parser = argparse.ArgumentParser(description="""
    Commandline arguments for fitting the VIPRS models
    """)

    # Required input/output data:
    parser.add_argument('-l', '--ld-panel', dest='ld_dir', type=str, required=True,
                        help='The path to the directory where the LD matrices are stored. '
                             'Can be a wildcard of the form ld/chr_*')
    parser.add_argument('-s', '--sumstats', dest='sumstats_path', type=str, required=True,
                        help='The summary statistics directory or file. Can be a '
                             'wildcard of the form sumstats/chr_*')
    parser.add_argument('--output-file', dest='output_file', type=str, required=True,
                        help='The output file where to store the inference results. Only include the prefix, '
                             'the extensions will be added automatically.')

    # Optional input data:
    parser.add_argument('--sumstats-format', dest='sumstats_format', type=str, default='plink',
                        choices={'plink', 'COJO', 'magenpy', 'fastGWA', 'custom'},
                        help='The format for the summary statistics file(s).')

    parser.add_argument('--snp', dest='snp', type=str, default='SNP',
                        help='The column name for the SNP rsID in the summary statistics file (custom formats).')
    parser.add_argument('--a1', dest='a1', type=str, default='A1',
                        help='The column name for the effect allele in the summary statistics file (custom formats).')
    parser.add_argument('--n-per-snp', dest='n_per_snp', type=str, default='N',
                        help='The column name for the sample size per SNP in '
                             'the summary statistics file (custom formats).')
    parser.add_argument('--z-score', dest='z_score', type=str, default='Z',
                        help='The column name for the z-score in the summary statistics file (custom formats).')
    parser.add_argument('--beta', dest='beta', type=str, default='BETA',
                        help='The column name for the beta (effect size estimate) in the '
                             'summary statistics file (custom formats).')
    parser.add_argument('--se', dest='se', type=str, default='SE',
                        help='The column name for the standard error in the summary statistics file (custom formats).')

    parser.add_argument('--temp-dir', dest='temp_dir', type=str, default='temp',
                        help='The temporary directory where to store intermediate files.')

    parser.add_argument('--validation-bed', dest='validation_bed', type=str,
                        help='The BED files containing the genotype data for the validation set. '
                             'You may use a wildcard here (e.g. "data/chr_*.bed")')
    parser.add_argument('--validation-pheno', dest='validation_pheno', type=str,
                        help='A tab-separated file containing the phenotype for the validation set. '
                             'The expected format is: FID IID phenotype (no header)')
    parser.add_argument('--validation-keep', dest='validation_keep', type=str,
                        help='A plink-style keep file to select a subset of individuals for the validation set.')

    # Model:
    parser.add_argument('-m', '--model', dest='model', type=str, default='VIPRS',
                        help='The PRS model to fit',
                        choices={'VIPRS', 'VIPRSMix', 'VIPRSAlpha'})
    parser.add_argument('--annealing-schedule', dest='annealing_schedule', type=str, default='linear',
                        help='The type of schedule for updating the temperature parameter in deterministic annealing.',
                        choices={'linear', 'harmonic', 'geometric'})
    parser.add_argument('--annealing-steps', dest='annealing_steps', type=int, default=0,
                        help="The number of deterministic annealing steps to perform.")
    parser.add_argument('--initial-temperature', dest='initial_temperature', type=float, default=5.,
                        help="The initial temperature for the deterministic annealing procedure.")
    parser.add_argument('--n-components', dest='n_components', type=int, default=3,
                        help='The number of non-null Gaussian mixture components to use with the VIPRSMix model '
                             '(i.e. excluding the spike component).')
    parser.add_argument('--prior-mult', dest='prior_mult', type=str, default='0.01,0.1,1.',
                        help='Prior multipliers on the variance of the non-null Gaussian mixture component.')

    # Hyperparameter tuning
    parser.add_argument('--hyp-search', dest='hyp_search', type=str, default='EM',
                        choices={'EM', 'GS', 'BO', 'BMA'},
                        help='The strategy for tuning the hyperparameters of the model. '
                             'Options are EM (Expectation-Maximization), GS (Grid search), '
                             'BO (Bayesian Optimization), and BMA (Bayesian Model Averaging).')
    parser.add_argument('--grid-metric', dest='grid_metric', type=str, default='validation',
                        help='The metric for selecting best performing model in grid search.',
                        choices={'ELBO', 'validation', 'pseudo_validation'})
    parser.add_argument('--opt-params', dest='opt_params', type=str, default='pi',
                        help='The hyperparameters to tune using GridSearch/BMA/Bayesian optimization (comma-separated).'
                             'Possible values are pi, sigma_beta, and sigma_epsilon. Or a combination of them.')

    # Grid-related parameters:
    parser.add_argument('--pi-grid', dest='pi_grid', type=str,
                        help='A comma-separated grid values for the hyperparameter pi (see also --pi-steps).')
    parser.add_argument('--pi-steps', dest='pi_steps', type=int, default=10,
                        help='The number of steps for the (default) pi grid. This will create an equidistant '
                             'grid between 1/M and (M-1)/M on a log10 scale, where M is the number of SNPs.')

    parser.add_argument('--sigma-epsilon-grid', dest='sigma_epsilon_grid', type=str,
                        help='A comma-separated grid values for the hyperparameter sigma_epsilon '
                             '(see also --sigma-epsilon-steps).')
    parser.add_argument('--sigma-epsilon-steps', dest='sigma_epsilon_steps', type=int, default=10,
                        help='The number of steps for the (default) sigma_epsilon grid.')

    parser.add_argument('--sigma-beta-grid', dest='sigma_beta_grid', type=str,
                        help='A comma-separated grid values for the hyperparameter sigma_beta '
                             '(see also --sigma-beta-steps).')
    parser.add_argument('--sigma-beta-steps', dest='sigma_beta_steps', type=int, default=10,
                        help='The number of steps for the (default) sigma_beta grid.')

    parser.add_argument('--h2-informed-grid', dest='h2_informed', action='store_true', default=False,
                        help='Construct a grid for sigma_epsilon/sigma_beta based on informed '
                             'estimates of the trait heritability.')

    # Generic:

    parser.add_argument('--compress', dest='compress', action='store_true', default=False,
                        help='Compress the output files')
    parser.add_argument('--genomewide', dest='genomewide', action='store_true', default=False,
                        help='Fit all chromosomes jointly')
    parser.add_argument('--backend', dest='backend', type=str, default='xarray',
                        choices={'xarray', 'plink'},
                        help='The backend software used for computations on the genotype matrix.')
    parser.add_argument('--max-attempts', dest='max_attempts', type=int, default=3,
                        help='The maximum number of model restarts (in case of optimization divergence issues).')
    parser.add_argument('--use-multiprocessing', dest='use_multiprocessing', action='store_true', default=False,
                        help='Use multiprocessing where applicable. For now, this mainly affects the '
                             'GridSearch/Bayesian Model Averaging implementations.')
    parser.add_argument('--n-jobs', dest='n_jobs', type=int, default=1,
                        help='The number of processes/threads to launch for the hyperparameter search (default is '
                             '1, but we recommend increasing this depending on system capacity).')

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Import required modules:

    import pandas as pd
    import os.path as osp
    from magenpy.stats.h2.ldsc import simple_ldsc
    from magenpy.utils.system_utils import get_filenames, makedir
    from magenpy.utils.model_utils import identify_mismatched_snps
    from magenpy.GWADataLoader import GWADataLoader

    from viprs.utils.HyperparameterGrid import HyperparameterGrid

    from viprs.model.VIPRS import VIPRS
    from viprs.model.VIPRSMix import VIPRSMix
    from viprs.model.VIPRSAlpha import VIPRSAlpha
    from viprs.model.VIPRSGridSearch import VIPRSGridSearch
    from viprs.model.VIPRSBMA import VIPRSBMA
    from viprs.model.HyperparameterSearch import BayesOpt, GridSearch, BMA

    # ----------------------------------------------------------
    print('{:-^62}\n'.format('  Parsed arguments  '))

    for key, val in vars(args).items():
        if val is not None and val != parser.get_default(key):
            print("--", key, ":", val)

    # ----------------------------------------------------------

    total_start_time = time.time()

    print('\n{:-^62}\n'.format('  Sanity checking and data preparation  '))
    # Sanity checking and data preparation:

    # Check the validation dataset:
    if args.hyp_search in ('BO', 'GS') and args.grid_metric in ('validation', 'pseudo_validation'):

        if args.validation_bed is None or args.validation_pheno is None:
            raise ValueError("To perform cross-validation, you need to provide BED files and a phenotype file "
                             "for the validation set (use --validation-bed and --validation-pheno).")
        else:
            valid_bed_files = get_filenames(args.validation_bed, extension='.bed')

            if len(valid_bed_files) < 1:
                raise FileNotFoundError(f"No BED files were identified at the "
                                        f"specified location: {args.validation_bed}")

            if not osp.isfile(args.validation_pheno):
                raise FileNotFoundError(f"No phenotype file found at {args.validation_pheno}")

        print("> Reading the validation dataset...")
        validation_gdl = GWADataLoader(bed_files=valid_bed_files,
                                       keep_file=args.validation_keep,
                                       phenotype_file=args.validation_pheno,
                                       backend=args.backend,
                                       temp_dir=args.temp_dir,
                                       n_threads=args.n_jobs)

        if args.grid_metric == 'pseudo_validation':
            print("> Performing GWAS to obtain validation summary statistics...")
            validation_gdl.perform_gwas()

    else:
        validation_gdl = None

    # Check the hyperparameters for the VIPRSMix model:
    if args.model == 'VIPRSMix':

        prior_mult = list(map(float, args.prior_mult.split(",")))
        if args.n_components != len(prior_mult):
            raise ValueError("The number of prior multipliers should match the "
                             "number of components for the Mixture prior.")

    # Find the set of hyperparameters to tune via search strategies:
    if args.hyp_search in ('BMA', 'GS', 'BO'):

        opt_params = args.opt_params.split(',')

    # Generate the hyperparameter grid:
    if args.hyp_search in ('BMA', 'GS'):

        if args.pi_grid is not None:
            pi_grid = list(map(float, args.pi_grid.split(",")))
        else:
            pi_grid = None

        if args.sigma_epsilon_grid is not None:
            sigma_epsilon_grid = list(map(float, args.sigma_epsilon_grid.split(",")))
        else:
            sigma_epsilon_grid = None

        if args.sigma_beta_grid is not None:
            sigma_beta_grid = list(map(float, args.sigma_beta_grid.split(",")))
        else:
            sigma_beta_grid = None

        print("> Constructing the hyperparameter grid...")
        grid = HyperparameterGrid(pi=pi_grid,
                                  sigma_epsilon=sigma_epsilon_grid,
                                  sigma_beta=sigma_beta_grid,
                                  search_params=args.opt_params.split(','))

    # Prepare the summary statistics parsers:
    if args.sumstats_format == 'custom':
        from magenpy.parsers.sumstats_parsers import SumstatsParser

        ss_parser = SumstatsParser(col_name_converter={
            args.snp: 'SNP',
            args.a1: 'A1',
            args.n_per_snp: 'N',
            args.z_score: 'Z',
            args.beta: 'BETA',
            args.se: 'SE'
        })
        ss_format = None
    else:
        ss_format = args.sumstats_format
        ss_parser = None

    # ----------------------------------------------------------

    print('\n{:-^62}\n'.format('  Reading input data  '))

    # Construct a GWADataLoader object using LD + summary statistics:
    gdl = GWADataLoader(ld_store_files=args.ld_dir,
                        temp_dir=args.temp_dir)

    gdl.read_summary_statistics(args.sumstats_path, sumstats_format=ss_format, parser=ss_parser)
    gdl.harmonize_data()

    if args.genomewide:
        data_loaders = [gdl]
    else:
        # If we are not performing inference genome-wide,
        # then split the GWADataLoader object into multiple loaders,
        # one per chromosome.
        print("> Splitting the data by chromosome...")
        data_loaders = gdl.split_by_chromosome().values()

    # ----------------------------------------------------------

    print('\n{:-^62}\n'.format('  Model details  '))

    print("- Model:", args.model)
    hyp_map = {'GS': 'Grid search', 'BO': 'Bayesian optimization',
               'BMA': 'Bayesian model averaging', 'EM': 'Expectation maximization'}
    print("- Hyperparameter tuning strategy:", hyp_map[args.hyp_search])
    if args.hyp_search in ('BO', 'GS'):
        print("- Model selection criterion:", args.grid_metric)

    # ----------------------------------------------------------
    print('\n{:-^62}\n'.format('  Model fitting  '))
    # List of effect size estimates
    eff_tables = []
    # List of hyperparameter estimates:
    hyp_tables = []
    # List of validation tables:
    valid_tables = []
    # Profiler metrics:
    prof_metrics = []

    # Lists to keep track of heritability and proportion of causal variants
    # per chromosome:
    h2g = []
    prop_causal = []

    for dl in data_loaders:

        converged = False
        n_attempts = 0

        if args.hyp_search in ('BMA', 'GS'):

            if args.h2_informed and ('sigma_epsilon' in opt_params or 'sigma_beta' in opt_params):
                h2 = simple_ldsc(dl)
            else:
                h2 = None

            for p in opt_params:
                if p == 'pi' and args.pi_grid is None:
                    grid.generate_pi_grid(steps=args.pi_steps, n_snps=dl.n_snps)
                if p == 'sigma_epsilon' and args.sigma_epsilon_grid is None:
                    grid.generate_sigma_epsilon_grid(steps=args.sigma_epsilon_steps, h2=h2)
                if p == 'sigma_beta' and args.sigma_beta_grid is None:
                    grid.generate_sigma_beta_grid(steps=args.sigma_beta_steps, h2=h2, n_snps=dl.n_snps)

        # Record the amount of memory used before fitting the model:
        mem_before = get_memory_usage()
        # Record time taken to load LD data:
        load_t0 = time.time()
        dl.load_ld()
        load_t1 = time.time()

        while n_attempts < args.max_attempts and not converged:

            # Fit the model to the data:
            if args.genomewide:
                print("> Performing model fit on all chromosomes jointly...")
            else:
                print("> Performing model fit on chromosomes:", dl.chromosomes)

            try:

                if args.model == 'VIPRS':
                    prs_m = VIPRS(dl)
                elif args.model == 'VIPRSMix':
                    prs_m = VIPRSMix(dl, K=args.n_components, prior_multipliers=prior_mult)
                elif args.model == 'VIPRSAlpha':
                    prs_m = VIPRSAlpha(dl)

                if args.hyp_search == 'EM':

                    final_m = prs_m.fit(annealing_schedule=args.annealing_schedule,
                                        annealing_steps=args.annealing_steps,
                                        init_temperature=args.initial_temperature)

                elif args.hyp_search == 'BO':

                    prs_m = BayesOpt(dl,
                                     opt_params,
                                     model=prs_m,
                                     validation_gdl=validation_gdl,
                                     criterion=args.grid_metric)
                    final_m = prs_m.fit()

                elif args.hyp_search == 'GS':

                    if args.use_multiprocessing:
                        prs_m = GridSearch(dl,
                                           grid,
                                           model=prs_m,
                                           criterion=args.grid_metric,
                                           validation_gdl=validation_gdl,
                                           n_jobs=args.n_jobs)

                        final_m = prs_m.fit()

                    else:

                        prs_m = VIPRSGridSearch(dl, grid)
                        prs_m.fit()
                        final_m = prs_m.select_best_model(validation_gdl=validation_gdl, criterion=args.grid_metric)

                elif args.hyp_search == 'BMA':

                    if args.use_multiprocessing:
                        prs_m = BMA(dl,
                                    grid,
                                    model=prs_m,
                                    n_jobs=args.n_jobs)

                        final_m = prs_m.fit()
                    else:
                        prs_m = VIPRSBMA(dl, grid)
                        prs_m.fit()
                        final_m = prs_m.average_models()

                converged = True
            except Exception as e:
                print(e)
                if e.__class__.__name__ == 'OptimizationDivergence' and n_attempts + 1 < args.max_attempts:

                    current_p_val_cutoff = 5e-8
                    filtered_snps = 0

                    while filtered_snps < 1 and current_p_val_cutoff <= .05:
                        # -----------------------------------------------------------
                        # Identify mismatched SNPs and remove them from analysis:
                        mismatched_snps = identify_mismatched_snps(dl, p_dentist_threshold=current_p_val_cutoff)
                        for c, mis_mask in mismatched_snps.items():
                            n_filt_snps = mis_mask.sum()
                            if n_filt_snps > 0:
                                filtered_snps += n_filt_snps
                                dl.filter_snps(dl.snps[c][~mis_mask], chrom=c)

                        if filtered_snps < 1:
                            current_p_val_cutoff *= 10.

                    if filtered_snps > 0:
                        print(f"> Filtered {filtered_snps} SNPs due to mismatch between "
                              f"summary statistics and LD reference panel.")
                        dl.harmonize_data()
                    else:
                        raise Exception("> Re-attempting model fit without filtering any new variants. Exiting...")
                    # -----------------------------------------------------------

                    n_attempts += 1
                elif n_attempts + 1 == args.max_attempts:
                    raise Exception("Error: Reached the maximum number of attempts "
                                    "for fitting the model without convergence!")
                else:
                    raise e

        fit_t1 = time.time()
        mem_after = get_memory_usage()

        # Record performance statistics:
        prof_metrics.append({
            'Chromosome': dl.chromosomes[0],
            'Memory_usage_MB': mem_after - mem_before,
            'Load_time': round(load_t1 - load_t0, 2),
            'Fit_time': round(fit_t1 - load_t1, 2)
        })

        # Extract the inferred model parameters:
        eff_tables.append(final_m.to_table())

        if args.hyp_search != 'BMA':
            # Extract inferred hyperparameters:
            m_h2g = final_m.get_heritability()
            m_p = final_m.get_proportion_causal()

            theta_table = final_m.to_theta_table()
            if not args.genomewide:
                theta_table['CHR'] = dl.chromosomes[0]

            hyp_tables.append(theta_table)
            h2g.append(m_h2g)
            prop_causal.append(m_p)

        # Extract validation tables:
        if args.hyp_search in ('GS', 'BO'):
            valid_table = prs_m.to_validation_table()
            if not args.genomewide:
                valid_table['CHR'] = dl.chromosomes[0]
            valid_tables.append(valid_table)

        # Cleanup:
        dl.cleanup()
        if validation_gdl is not None:
            validation_gdl.cleanup()

        del dl

        print("* * * * * * * * * *")

    print("\n>>> Writing the inference results to:\n", osp.dirname(args.output_file))

    # Record end time:
    total_end_time = time.time()

    makedir(osp.dirname(args.output_file))

    # If the user wants the files to be compressed, append `.gz` to the name:
    c_ext = ['', '.gz'][args.compress]

    if len(eff_tables) > 0:
        pd.concat(eff_tables).to_csv(args.output_file + '.fit' + c_ext, sep="\t", index=False)

    if len(hyp_tables) > 0:
        pd.concat(hyp_tables).to_csv(args.output_file + '.hyp' + c_ext, sep="\t", index=False)

    if len(valid_tables) > 0:
        pd.concat(valid_tables).to_csv(args.output_file + '.validation' + c_ext, sep="\t", index=False)

    if len(prof_metrics) > 0:
        prof_df = pd.DataFrame(prof_metrics)
        prof_df['Total_WallClockTime'] = round(total_end_time - total_start_time, 2)
        prof_df.to_csv(args.output_file + '.time', sep="\t", index=False)


if __name__ == '__main__':

    import time
    from datetime import timedelta

    # Record start time:
    start_time = time.time()

    # Run VIPRS:
    main()

    print("Done!")
    # Record the end time:
    end_time = time.time()
    print('Total runtime:', timedelta(seconds=end_time - start_time))