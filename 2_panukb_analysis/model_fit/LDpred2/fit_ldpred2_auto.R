# NOTE: This code is modified from:
# https://github.com/privefl/paper-ldpred2/blob/master/code/run-ldpred2-gwide.R#L34-L118

library(bigsnpr)
library(dplyr)

args <- commandArgs(trailingOnly=TRUE)
ss_path <- args[1]
ld_panel_path <- "2_panukb_analysis/model_fit/LDpred2/data/ld/"

# Set the number of cores:
NCORES <- 8
print(paste("Using up to", NCORES, "threads."))

# Extract information about the trait and configuration:
# For the trait, replace ".sumstats.gz" with "" and take the basename:
trait <- gsub(".sumstats.gz", "", basename(ss_path))
train_pop <- "EUR"  # For now fixing to Europeans

data_prep_start <- Sys.time()

# ----------------------------------------------------------
# Step 1: Prepare the GWAS summary statistics:

print("> Reading sumstats file...")

sumstats <- read.table(ss_path, header=TRUE)
names(sumstats) <- c("chr", "pos", "rsid", "a0", "a1", "maf", "beta", "beta_se", "n_eff")

# ----------------------------------------------------------
# Step 2: Match the GWAS summary statistics and the LD reference panel

print("> Matching sumstats with LD reference panel...")

map_ldref <- readRDS(file.path(ld_panel_path, "map_hm3_plus.rds"))

sumstats <- snp_match(sumstats, map_ldref, join_by_pos=F)
(sumstats <- tidyr::drop_na(tibble::as_tibble(sumstats)))

tmp <- tempfile(tmpdir = Sys.getenv("SLURM_TMPDIR", unset="temp"))
on.exit(file.remove(paste0(tmp, ".sbk")), add = TRUE)

for (chr in 1:22) {

  print(paste("> Matching chromosome:", chr))

  ## indices in 'sumstats'
  ind.chr <- which(sumstats$chr == chr)
  ## indices in 'corr'
  ind.chr2 <- sumstats$`_NUM_ID_`[ind.chr]
  ## indices in 'corr'
  ind.chr3 <- match(ind.chr2, which(map_ldref$chr == chr))

  corr0 <- readRDS(file.path(ld_panel_path, sprintf("LD_with_blocks_chr%d.rds", chr)))[ind.chr3, ind.chr3]

  if (chr == 1) {
    df_beta <- sumstats[ind.chr, c("beta", "beta_se", "n_eff", "_NUM_ID_")]
    ld <- Matrix::colSums(corr0^2)
    corr <- as_SFBM(corr0, tmp)
  } else {
    df_beta <- rbind(df_beta, sumstats[ind.chr, c("beta", "beta_se", "n_eff", "_NUM_ID_")])
    ld <- c(ld, Matrix::colSums(corr0^2))
    corr$add_columns(corr0, nrow(corr))
  }
}

subset_map_ldref <- map_ldref[df_beta$`_NUM_ID_`,]

data_prep_end <- Sys.time()

data_prep_time <- as.numeric(data_prep_end - data_prep_start, units = "secs")

# ----------------------------------------------------------
# Step 3: Perform model fitting

inference_start_time <- Sys.time()

print("> Performing model fit...")

# Estimate heritability using LD score regression:
print("> Running LDSC...")
(ldsc <- with(df_beta, snp_ldsc(ld, length(ld), chi2 = (beta / beta_se)^2,
                                sample_size = n_eff, blocks = NULL,
                                ncores = NCORES)))
h2_est <- ldsc[["h2"]]

# If the heritability estimate is negative, set it to a reasonable small value (e.g. 0.01)
if (h2_est < 0.){
  h2_est <- 0.01
}

print("Running LDpred2-auto...")
multi_auto <- snp_ldpred2_auto(corr, df_beta, h2_init = h2_est,
                               vec_p_init = seq_log(1e-4, 0.2, 30),
                               ncores = NCORES)

# Get the betas from snp_ldpred2_auto
# Copied from: https://github.com/comorment/containers/blob/main/scripts/pgs/LDpred2/fun.R
#' @param fitAuto The return value form snp_ldpred2_auto
#' @param quantile Range of estimates to keep
getBetasAuto <- function (fitAuto, quantile=0.95, verbose=T) {
  corrRange <- sapply(fitAuto, function (auto) diff(range(auto$corr_est)))
  # Keep chains that pass the filtering below
  keep <- (corrRange > (0.95 * quantile(corrRange, 0.95, na.rm=T)))
  nas <- sum(is.na(keep))
  if (nas > 0 && verbose) cat('Omitting', nas, 'chains out of', length(keep), ' due to missing values in correlation range\n')
  keep[is.na(keep)] <- F
  beta <- rowMeans(sapply(fitAuto[keep], function (auto) auto$beta_est))
  beta
}

print("> Extracting betas...")
final_beta_auto <- getBetasAuto(multi_auto)

inference_end_time <- Sys.time()
inference_time <- as.numeric(inference_end_time - inference_start_time, units = "secs")

# ----------------------------------------------------------
# Step 4: Write the posterior mean effect sizes

# Create the output directory:

# Write out the effect sizes:
print("> Writing the results to file...")

dir.create(sprintf("data/model_fit/panukb_sumstats/external/LDpred2-auto/%s/%s/", train_pop, trait),
           showWarnings = FALSE,
           recursive = TRUE)

output_df <- subset_map_ldref[, c("chr", "pos", "rsid", "a1", "a0")]
output_df$BETA <- final_beta_auto
names(output_df) <- c("CHR", "POS", "SNP", "A1", "A2", "BETA")

output_f <- sprintf("data/model_fit/panukb_sumstats/external/LDpred2-auto/%s/%s/LDpred2-auto.fit", train_pop, trait)

write.table(output_df,
            output_f,
            row.names = F,
            sep = "\t")
system(sprintf("gzip -f %s", output_f))

# Write the detailed metrics:

prof_df <- data.frame(DataPrep_Time = data_prep_time,
                      Fit_time = inference_time,
                      Total_WallClockTime = data_prep_time + inference_time)

# Write the data frame to a .tsv file
write.table(prof_df,
            sprintf("data/model_fit/panukb_sumstats/external/LDpred2-auto/%s/%s/LDpred2-auto_detailed.prof", train_pop, trait),
            sep = "\t", row.names = FALSE, quote = FALSE)
