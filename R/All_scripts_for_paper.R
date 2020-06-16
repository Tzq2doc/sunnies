source("datasets/simulated_datasets.R")
source("shapley_helpers.R")
source("utility_functions.R")
source("comparison_helpers.R")
source("shapley_helpers.R")
source("Applications_helpers.R")
library(xgboost)
library(dplyr)

### ERDA Example 1 -----------------------------------------------------------
# The repetitions producing violin plots is in python, but here is similar:
result1 <- run_evaluations(dat_unif_squared, utility = DC, n = 1e3, plots = T)

### ERDA Examples 2 and 3  ----------------------------------------------------
# (which should be the same example)
# This example should be done exactly by hand, but here is a simulation.
# When it is calculated exactly, the shapley values will be equal to each other:
result2 <- run_evaluations(dat_catcat_XOR, utility = DC, n = 1e3, plots = T)

### DARRP EXAMPLE 1 -----------------------------------------------------------
# One feature becomes more important, one less important.
dat_t <- function(n, d, t, max_t, extra_features = T, de = 46) {
  X <- matrix(rnorm(n*d,0,2), nrow = n, ncol = d)
  extras <- if (extra_features) {
    Xe <- matrix(rnorm(n*de,0,sqrt(0.05)), nrow = n, ncol = de)
    rowSums(Xe)
  } else {0}
  y <- X %*% c(rep(1,d-2), 1 + t/max_t, 1 - t/max_t) + extras
  
  if (extra_features) {return(cbind(y,X,Xe))}
  return(cbind(y,X))
}
n <- 1e4; m <- 10; d <- 4; N <- 100; s <- 1000
shaps_lab <- matrix(0, nrow = m+1, ncol = d) 
shaps_res <- matrix(0, nrow = m+1, ncol = d)
mse <- vector(mode = "numeric", length = m+1)
datt <- dat_t(n, d, t = 0, max_t = m)
sdatt <- split_dat(datt)
xgb <- basic_xgb_fit(sdatt) # Model is only fit once
cat("Number of features: ", xgb$nfeatures)
cdN_all <- array(NA, dim = c(6,d,N,m))
for (t in 0:m) {
  print(paste0("round ",t))
  xgbtt <- basic_xgb_test(xgb, sdatt) # Model is tested each time
  mse[t+1] <- xgbtt$test_mse
  cdNt <- compare_DARRP_N(sdatt, xgbtt, features = 1:4, 
                         feature_names = paste0("x",1:4),
                         sample_size = s, N = N,
                         valid = F, all_labs = F)
  cdN_all[,,,t] <- cdNt
  datt <- dat_t(n, d, t = t, max_t = m)
  sdatt <- split_dat(datt)
}
#saveRDS(cdN_all, "run1_cdN_drift.Rds")
#saveRDS(mse, "run1_cdN_drift_mse.Rds")
plot_compare_DARRP_N_drift <- function(cdN_all, p = c(0.025,0.975), 
                                       feature_names = paste0("x", 1:4),
                                       main = "test run") {
  m <- dim(cdN_all)[4]
  for (t in 1:m) {
    cdt <- apply(cdN_all[,,,t], FUN = mean, MARGIN = c(1,2))
    cdt_L <- apply(cdN_all[,,,t], FUN = quantile, MARGIN = c(1,2), probs = p[1])
    cdt_U <- apply(cdN_all[,,,t], FUN = quantile, MARGIN = c(1,2), probs = p[2])
  }
  
}
plot_compare_DARRP_N_drift(cdN_all[,,,3], main = "test run", all_labs = F)
#pdf("diagnostics_drift.pdf", width = 6, height = 3)
#par(mfrow = c(1,3))
matplot(x = 0:m, y = shaps_res, type = 'b', 
        xlab = "time", ylab = "residual importance")
matplot(x = 0:m, y = shaps_lab, type = 'b', 
        xlab = "time", ylab = "response importance")
plot(0:m, mse, type = 'b', xlab = "time", ylab = "MSE")
#dev.off()

### DARRP EXAMPLE 2 -----------------------------------------------------------
n <- 1e4; d <- 5
X <- matrix(rnorm(n*(d-2),0,1), nrow = n, ncol = (d-2))
X4 <- sample(0:1, replace = T, n)
X5 <- sample(0:1, replace = T, n)
y <- rowSums(X[,-(d-2)]) + 5*(X4 & X5)*X[,(d-2)] + rnorm(n,0,0.1)
dat <- cbind(y,X,X4,X5)
sdat <- split_dat(dat, df = T)
lmodel <- lm(y ~ x1 + x2 + x3 + x4 + x5, dat = sdat$df_yx_train)
lmodelt <- basic_lmodel_test(lmodel, sdat)
cdN <- compare_DARRP_N(sdat, lmodelt, features = 1:5, 
                       feature_names = paste0("x",1:5),
                       sample_size = 1000, N = 100,
                       valid = F, all_labs = F)
#saveRDS(cdN, "run1_cdN_linear1.Rds")
plot_compare_DARRP_N(cdN, main = "test run", all_labs = F)
lmodel2 <- lm(y ~ x1 + x2 + x3 + x4 + x5 + x3:x4:x5, dat = sdat$df_yx_train)
lmodel2t <- basic_lmodel_test(lmodel2, sdat)
cdN2 <- compare_DARRP_N(sdat, lmodel2t, features = 1:5, 
                       feature_names = paste0("x",1:5),
                       sample_size = 1000, N = 100,
                       valid = F, all_labs = F)
#saveRDS(cdN2, "run1_cdN_linear2.Rds")
plot_compare_DARRP_N(cdN2, main = "test run", all_labs = F)


### DARRP EXAMPLE 3 -----------------------------------------------------------
# This will be simpson's paradox (if we include it).


### Application 1 -----------------------------------------------------------
# The NHANES I dataset from [14]
Xh <- read.csv("../RL_data/X_data_with_header.csv")
y <- read.csv("../RL_data/y_data.csv", header = F)
names(y) <- "logRR"
Xh <- apply(Xh, FUN = function(x){x[is.nan(x)] <- NA; x}, MARGIN = 2)
Xh <- as_tibble(Xh)
X_dr <- remove_all_missing(Xh2, ncols = 3)
y_dr <- y[attr(X_dr, "keep"),]
dat <- cbind(y_dr, X_dr)
interesting <- c("age", "physical_activity", "systolic_blood_pressure")
fts <- which(colnames(dat) %in% interesting) - 1
fnams <- c("age", "PA", "SBP")
cdN3way <- compare_DARRRP_N_gender_3way(
  dat, sample_size = 1000, N = 100,
  features = fts, feature_names = fnams)
#save(cdN3way, file = "run1_cdN3way.dat")
plot_compare_DARRP_N(cdN1000[[1]], main = "cdN3way")

