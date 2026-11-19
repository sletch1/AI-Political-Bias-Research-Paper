# Independent re-implementation of the nested variance-components model with R/lme4, plus boundary-robust intervals.
#   y ~ 1 + (1|model) + (1|model:instrument) + (1|model:trait)       (REML)
# Compares REML variance shares with the Python/statsmodels fit in results/nested_design/, and reports
# (i) likelihood-profile intervals for each standard deviation and (ii) parametric-bootstrap percentile intervals for each share.
# usage: Rscript crosscheck_lme4.R [nsim]
suppressMessages(library(lme4))
args <- commandArgs(TRUE); nsim <- if (length(args)) as.integer(args[1]) else 500
dir <- file.path("..", "results", "lme4_check")
shares <- function(fit) { v <- as.data.frame(VarCorr(fit))$vcov; names(v) <- as.data.frame(VarCorr(fit))$grp; v <- v/sum(v); v[c("model","model:instrument","model:trait","Residual")] }
res <- list()
for (p in c("two", "four")) {
  d <- read.csv(file.path(dir, sprintf("panel_%s.csv", p)))
  d$model <- factor(d$model); d$instrument <- factor(d$instrument); d$trait <- factor(d$trait)
  d$mi <- interaction(d$model, d$instrument, drop = TRUE); d$mt <- interaction(d$model, d$trait, drop = TRUE)
  fit <- lmer(y ~ 1 + (1 | model) + (1 | mi) + (1 | mt), data = d, REML = TRUE,
              control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
  vc <- as.data.frame(VarCorr(fit)); v <- setNames(vc$vcov, vc$grp); v <- v[c("model","mi","mt","Residual")]; sh <- v / sum(v)
  cat(sprintf("\n== %s-instrument panel (n=%d, %d models) ==\nREML shares (lme4): model %.1f%%  model x instrument %.1f%%  model x axis %.1f%%  noise %.1f%%   singular=%s\n",
              p, nrow(d), nlevels(d$model), 100*sh[1], 100*sh[2], 100*sh[3], 100*sh[4], isSingular(fit)))
  pr <- tryCatch(confint(fit, method = "profile", oldNames = FALSE, quiet = TRUE), error = function(e) NULL)
  if (!is.null(pr)) { cat("Likelihood-profile 95% intervals for the standard deviations (z-scale):\n"); print(round(pr[grep("^sd_|sigma", rownames(pr)), ], 3)) }
  fun <- function(f) { vc <- as.data.frame(VarCorr(f)); v <- setNames(vc$vcov, vc$grp)[c("model","mi","mt","Residual")]; v/sum(v) }
  set.seed(20261119)
  bs <- bootMer(fit, fun, nsim = nsim, type = "parametric", use.u = FALSE, parallel = "multicore", ncpus = max(1, parallel::detectCores() - 2))
  ci <- apply(bs$t, 2, quantile, c(.025, .975), na.rm = TRUE)
  cat(sprintf("Parametric-bootstrap (%d sims) 95%% percentile intervals for shares, %%:\n", nsim))
  nm <- c("model","model x instrument","model x axis","noise"); for (i in 1:4) cat(sprintf("  %-18s %5.1f  [%5.1f, %5.1f]\n", nm[i], 100*sh[i], 100*ci[1,i], 100*ci[2,i]))
  cat(sprintf("  share of bootstrap fits with model share < 0.5%%: %.1f%%\n", 100*mean(bs$t[,1] < .005, na.rm = TRUE)))
  res[[p]] <- list(shares = sh, ci = ci, profile = pr, singular = isSingular(fit), p_boundary = mean(bs$t[,1] < .005, na.rm = TRUE))
}
saveRDS(res, file.path(dir, "lme4_check.rds"))
