# Towards whole-genome inference of polygenic scores with fast and memory-efficient algorithms (2025)

### **Authors:** Shadi Zabad[1], Chirayu Anant Haryan[2], Simon Gravel[1], Sanchit Misra[2], and Yue Li[1]
**[1]** McGill University and **[2]** Intel Labs

## Introduction

This repository contains scripts and guidelines to reproduce the analyses in the manuscript 
"Towards whole-genome inference of polygenic scores with fast and memory-efficient algorithms" (2025). 
The manuscript describes new efficient algorithms and data structures for Polygenic Risk Score (PRS) inference at 
whole-genome scale. These algorithms are implemented in two open source python packages that are available on 
the Python Package Index (PyPI). The homepages for these packages are:

1. `viprs`: https://github.com/shz9/viprs
2. `magenpy`: https://github.com/shz9/magenpy

The `viprs` package provides a high-level interface for PRS inference, while the `magenpy` package provides 
data structures and routines for computing LD matrices, interacting with genotype data, and harmonizing various 
data sources used in statistical genetics applications.

## Setup / Data

In order to reproduce the analyses in the manuscript, you will need to clone this repository and 
set up the compute environment with all the necessary dependencies. In order to facilitate this, 
we provide a script for `SLURM`-like clusters that creates `python` virtual environments where the 
necessary packages are installed. You can use set up the environment by cloning the repo and then 
running the `setup_environment.sh` script as follows:


```bash
git clone https://github.com/shz9/viprs-benchmarks-paper.git
cd viprs-benchmarks-paper
bash setup_environment.sh
```

If you don't have access to a `SLURM`-like cluster, you may need to minimally modify the `setup_environment.sh`
script to point to the correct `python` and `pip` executables on your system.

To access the data used in or produced as part of this work, please consult the pages for the 
corresponding Zenodo records:

- Five-fold benchmarking summary statistics for Standing Height: https://doi.org/10.5281/zenodo.14270953
- The Pan-UKB phenotype manifest with heritability estimates, QC flags, and hyperlinks to download GWAS summary statistics:
https://docs.google.com/spreadsheets/d/1AeeADtT0U1AukliiNyiVzVRdLYPkTbruQSk38DeutU8
- Mapping files for rsIDs across genome builds `hg19` and `hg38` from the UKB-PPP: 
https://www.synapse.org/Synapse:syn51364943/wiki/622119
- LD matrices for the six continental ancestry groups are available for download via 
GitHub (https://github.com/shz9/viprs/releases/tag/v0.1.2) and Zenodo (https://zenodo.org/records/14614207)
- LD blocks defined by `LDetect`: https://bitbucket.org/nygcresearch/ldetect-data/src/master/

For other data sources not listed here, consult the `Data Availability` section of the manuscript.

## Steps to reproduce the analyses

The analyses in the manuscript are structure according to the following steps:

### **(1) Data preparation** (`0_data_prepataion/`): Download and preprocess the data used in the manuscript. 

This step includes pre-processing the genotype data, downloading summary statistics, and computing 
the LD matrices used in the benchmarking experiments and PRS inference. Some of the data being processed 
here is private (e.g. UK Biobank data), and you may need to request access to these data sources in order 
to reproduce the analyses. If you do have access to these data sources, then you should be able to reproduce 
all the steps by modifying the `global_config.sh` script to point towards the correct path on your 
system. The subtasks here are:

1. **Downloading and post-processing GWAS summary statistics** (`0_data_preparation/external_sumstats/`):
The main scripts to execute here are:
   - `download_benchmarking_sumstats.sh`: This will download the 5-fold cross-validation summary statistics 
   for standing height that were used in the benchmarking experiments.
   - `download_panukb_data.py`: This will download the GWAS summary statistics for the Pan-UKB resource. After 
   downloading the requisite data, also call `process_panukb_sumstats.py` to transform the sumstats files 
   for the purposes of PRS inference.
2. **Pre-process the UK Biobank data** (`0_data_prepataion/ukbb_data/`): This step requires access to the UK 
Biobano genotype and phenotype data. The main steps here are:
    - Generate the QC filters for the UK Biobank data by running the python script `generate_qc_filters.py`.
    - Extract the genotype data for all UK Biobank samples and transform to `BED` file format. This can be done with 
   the pair of scripts `batch_qc.sh` and `ukbb_qc_job.sh`. The former script will submit `SLURM` jobs to do 
   the processing per chromosome separately.
    - Extract phenotype data by running the script `prepare_phenotype_data.py`.
3. **Compute the LD matrices** (`0_data_prepataion/ld/`): This step requires access to the UK Biobank genotype 
data from the previous step. The main steps here are:
    - Download pre-computed LD matrices and transform them to the new format using the scripts `download_precomputed_ld.sh`
    and `convert_precomputed_ld.sh`.
    - Download `LDetect` blocks by executing the script `download_ldetect_blocks.sh`.
    - Compute new LD matrices by executing the script `batch_ld.sh` which will submit `SLURM` jobs to compute 
    LD matrices for all populations and chromosomes. If you'd like to compute individual matrices, check 
   the script `compute_ld.sh` and see the required arguments. For our main experiments, we called the `batch_ld.sh` 
   script with the following arguments:

```bash
source 0_data_prepataion/ld/batch_ld.sh hq_imputed_variants_hm3 block int8 xarray
source 0_data_prepataion/ld/batch_ld.sh hq_imputed_variants_maf001 block int8 xarray
source 0_data_prepataion/ld/batch_ld.sh hq_imputed_variants block int8 xarray
```

4. **Pre-process data for CARTaGENE Biobank** (`0_data_prepataion/flagship_cartagene_data/`): This step requires 
access to individual-level data from the CARTaGENE biobank. The main steps here are:
   - The processed genotype data is already provided to us via the Flagship project.
   - Extracting the phenotype data is done via the script `prepare_phenotype_data.py`.
   - Extracting the covariates (used in evaluation) is done via the script `prepare_covariates.py`.
   - Due to the fact that the CARTaGENE data is called based on `hg38`, we also need to create mapping files for 
    `hg19` and `hg38` using the script `create_rsid_map.py`.

After completing the data preparation steps, you should have all the necessary data to proceed with the
benchmarking and Pan-UKB analyses.

### **(2) Benchmarking experiments** (`1_benchmarks/`)

The analyses in this step are aimed at benchmarking the new `viprs` software (`v0.1`) and compare it to the older 
implementation (`v0.0.4`). The benchmarks are designed to assess 4 aspects of the computational performance of the 
`VIPRS` model:

1. **Wall-clock time**: This is the time taken to run the PRS inference algorithm genome-wide. This metric includes
the time to read the input data, harmonize GWAS and LD metadata, and perform inference.
2. **Peak memory**: This is the maximum amount of memory used by the PRS inference algorithm during execution.
3. **Prediction accuracy**: This is the correlation between the predicted PRS and the true phenotype.
4. **Runtime-per-iteration**: This is the time it takes to perform a single coordinate ascent step in the 
Coordinate Ascent Variational Inference (CAVI) algorithm (E-Step in our algorithm).

Metrics 1-3 are computed by running the scripts in `1_benchmarks/total_runtime_benchmarks/`. If you have 
access to a `SLURM` system, you can run all of these experiments by simply executing the script: 

```bash
source 1_benchmarks/total_runtime_benchmarks/batch_benchmark.sh
```

Metric 4 is computed by running the scripts in `1_benchmarks/e_step_benchmarks/`. If you have access to a
`SLURM` system, you can run all of these experiments by simply executing the script:

```bash
source 1_benchmarks/e_step_benchmarks/batch_benchmark_e_step.sh
```

### **(3) Pan-UK Biobank experiments** (`2_panukb_analysis/`)

The analyses in this step are aimed at evaluating the performance of the various PRS models (including `VIPRS`) 
on large-scale GWAS summary statistics from the Pan-UK Biobank data resource. 
This step is broken down into 3 sub-tasks:

1. **PRS inference** (`2_panukb_analysis/model_fit/`): This task involves performing PRS 
inference on the GWAS summary statistics for the 75 phenotypes, potentially using different LD reference panels.
The methods represented are `viprs` (new version), `old_viprs` (older version), `SBayesRC`, `LDpred2`, 
and `PRScs`. Each of these methods has its own subdirectory with appropriate setup scripts that download the software,
install the dependencies, and run the inference. If you have access to a `SLURM` system, you can run
the inference for each method by invoking the `batch_fit.sh` script in the corresponding subdirectory.

For example, the following commands will run the standard version of `VIPRS` on all 75 phenotypes and 3 variant sets:


```bash
source 2_panukb_analysis/model_fit/viprs/batch_fit.sh panukb_sumstats hq_imputed_variants_hm3 block int8
source 2_panukb_analysis/model_fit/viprs/batch_fit.sh panukb_sumstats hq_imputed_variants_maf001 block int8
source 2_panukb_analysis/model_fit/viprs/batch_fit.sh panukb_sumstats hq_imputed_variants block int8
```

To run `SBayesRC`, first you need to call `source 2_panukb_analysis/model_fit/SBayesRC/setup_sbayesrc_env.sh` to set up the environment. 
Then you can run the inference by executing the script `batch_fit.sh` in the `SBayesRC` subdirectory.

```bash
source 2_panukb_analysis/model_fit/SBayesRC/batch_fit.sh
source 2_panukb_analysis/model_fit/SBayesRC/batch_fit.sh 7m
```


2. **Computing Polygenic Scores (linear scoring)** (`2_panukb_analysis/score/`): This task involves computing polygenic
scores for all participants in the UK Biobank and CARTaGENE Biobank. Assume you have access to the individual-level
genotype data for these participants, you can run the following scripts to compute the PRS:

```bash
source 2_panukb_analysis/score/batch_score.sh  # For the UKB
source 2_panukb_analysis/score/cartagene_score/batch_score.sh  # For the CARTaGENE Biobank
```

This will generate polygenic scores for all the 75 phenotypes and across the 3 variant sets.

3. **Model evaluation** (`2_panukb_analysis/evaluation/`): This task involves evaluating the performance of the 
polygenic scores computed in the previous step. This step assumes that you have access to the 
individual-level phenotype data already. The main script to run here is:

```bash
source 2_panukb_analysis/evaluation/batch_evaluation.sh
```

### **(4) Plotting and visualization** (`3_plotting/`)

This step involves generating all the plots and visualizations used in the manuscript (including Supplementary Material). 
Assuming you have access to the data from the previous steps, this can be simply done by running the master script:

```bash
source 3_plotting/plot_commands.sh
```

### **(5) Tables** (`4_tables/`)

This step involves generating (some) of the tables used in the manuscript (including Supplementary Material).
Assuming you have access to the data from the previous steps, this can be simply done by running the master script:

```bash
source 4_tables/table_commands.sh
```

### Other questions or concerns

For more information about the analyses and scripts found within this repository, please contact the 
corresponding authors:

- Yue Li (yueli@cs.mcgill.ca)
- Simon Gravel (simon.gravel@mcgill.ca)

Or open an [issue](https://github.com/shz9/viprs-benchmarks-paper/issues) on the `GitHub` repository 
for this project.