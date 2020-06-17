source("datasets/simulated_datasets.R")
source("shapley_helpers.R")
source("utility_functions.R")
source("comparison_helpers.R")
source("shapley_helpers.R")
source("Applications_helpers.R")
library(xgboost)
library(dplyr)
library(tidyr)
library(ggplot2)
library(latex2exp)
library(reticulate)
library(reshape2)

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
cdN_all <- readRDS("run1_cdN_drift.Rds")
mse <- readRDS("run1_cdN_drift_mse.Rds")
plot_compare_DARRP_N_drift <- function(
  cdN_all, p = c(0.025,0.975),
  feature_names = paste0("X",1:d), 
  shap_index = c(1,5), #only two allowed
  shap_type = c("ADL", "ADR"), 
  y_name = "Shapley value",
  leg_labs = unname(c(TeX("$X_1"),TeX("$X_2"),TeX("$X_3"),TeX("$X_4")))) {
  
  m <- dim(cdN_all)[4]
  d <- dim(cdN_all)[2]
  cd2 <- matrix(ncol = d, nrow = 0)
  for (t in 1:m) {
    cdt <- cdN_all[shap_index,,,t] %>% 
      apply(MARGIN = c(1,2), FUN = function(x){ 
        c(mean(x), quantile(x, probs = p)) 
      })
    for (i in 1:length(shap_index)) {cd2 <- rbind(cd2, cdt[,i,])}
  }
  colnames(cd2) <- feature_names
  cd2 <- cd2 %>% 
    data.frame() %>% 
    cbind(S = rep(shap_type,each=3), 
          CI = 1:3, time = rep(0:(m-1), each = 6)) %>% 
    pivot_longer(all_of(feature_names), names_to = "feature") %>% 
    pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
  
  plt <- ggplot(data=cd2, aes(x=time, y=CI1, colour=feature, fill=feature,
                             shape=feature, linetype=feature)) + 
    geom_point() + geom_line() +
    geom_ribbon(aes(ymin=CI2, ymax=CI3), alpha=0.1) +
    scale_x_continuous(breaks = 0:m) +
    scale_y_continuous(name = y_name) +
    scale_colour_discrete(labels=leg_labs) +
    scale_shape_discrete(labels=leg_labs) +
    scale_fill_discrete(labels=leg_labs) +
    scale_linetype_discrete(labels=leg_labs) + 
    theme_set(theme_minimal())  +
    facet_grid(S ~ .) +
    theme(strip.text.y.right = element_text(angle = 0))
  return(plt)
}
pdf(file="Drift_ADL_ADR.pdf",width=5,height=4)
plot_compare_DARRP_N_drift(cdN_all, shap_index = c(1,5))
dev.off()


plot_compare_DARRP_N_drift2 <- function(
  cdN_all, p = c(0.025,0.975), d = dim(cdN_all)[2],
  feature_names = paste0("X",1:d), shap_index = c(1,5), #only two allowed
  shap_type = c("ADL", "ADR"), y_name = "Shapley value",
  leg_labs = unname(c(TeX("$X_1"),TeX("$X_2"),TeX("$X_3"),TeX("$X_4")))) {
  
  m <- dim(cdN_all)[4]
  N <- dim(cdN_all)[3]
  d <- dim(cdN_all)[2]
  s <- dim(cdN_all)[1]
  dimnames(cdN_all) <- list(
    S = 1:s, feature = paste0("X",1:d), i = 1:N, t = 0:(m-1))
  cdN_lines <- as_tibble(reshape2::melt(cdN_all)) %>% 
    dplyr::filter(S %in% shap_index)
  cdN_lines$S <- rep(shap_type, nrow(cdN_lines)/2)
  cd2 <- matrix(ncol = d, nrow = 0)
  for (t in 1:m) {
    cdt <- cdN_all[shap_index,,,t] %>% 
      apply(MARGIN = c(1,2), FUN = function(x){ 
        c(mean(x), quantile(x, probs = p)) 
      })
    for (i in 1:length(shap_index)) {cd2 <- rbind(cd2, cdt[,i,])}
  }
  colnames(cd2) <- feature_names
  cd2 <- cd2 %>% 
    data.frame() %>% 
    cbind(S = rep(shap_type,each=3), 
          CI = 1:3, time = rep(0:(m-1), each = 6)) %>% 
    pivot_longer(all_of(feature_names), names_to = "feature") %>% 
    pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
  
  plt <- ggplot(data=cd2, aes(x=time, y=CI1, fill=feature,
                              shape=feature)) + 
    geom_point() + geom_line() +
    geom_ribbon(aes(ymin=CI2, ymax=CI3), alpha=0.1) +
    scale_x_continuous(breaks = 0:m) +
    scale_y_continuous(name = y_name) +
    scale_shape_discrete(labels=leg_labs) +
    scale_fill_discrete(labels=leg_labs) +
    theme_set(theme_minimal())  +
    facet_grid(S ~ .) +
    theme(strip.text.y.right = element_text(angle = 0)) +
    geom_line(data = cdN_lines,
              mapping = aes(y=value, x=t, group=interaction(i,feature),
                            colour=interaction(i,feature)),
              alpha=0.02, colour="grey20"); plt
  return(plt)
}
pdf(file="Drift_ADL_ADR_lines.pdf",width=6,height=5)
plot_compare_DARRP_N_drift2(cdN_all, shap_index = c(1,5))
dev.off()

ggplot() +
geom_line(data = dplyr::filter(cdN_lines, S == "ADR",
                               feature == 1), 
          mapping = aes(y=value, x=t, group=i),
          alpha=1, colour="grey20")



X11(); plt






df <- dcast(melt(cdN_all), Var1+Var2~Var3) 
test <- apply(cdN_all, 1L, c)
expand.grid(dimnames(cdN_all)[1:2])

#m <- dim(cdN_all)[4]
#cd <- array(dim = c(0,d))
#for (t in 1:m) {
#  cd <- cdN_all[shap_index,,,t] %>% 
#    apply(MARGIN = 1, FUN = function(x){ 
#      c(mean(x), quantile(x, probs = p)) 
#    }) %>% rbind(cd, .)
#}
#colnames(cd) <- feature_names
#cd <- cd %>% 
#  cbind(CI = 1:3, time = rep(0:(m-1), each = 3)) %>% 
#  data.frame() %>% 
#  pivot_longer(all_of(feature_names), names_to = "feature") %>% 
#  pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
#
#plt <- ggplot(data=cd, aes(x=time, y=CI1, colour=feature, fill=feature,
#                           shape=feature, linetype=feature)) + 
#  geom_point() + geom_line() +
#  geom_ribbon(aes(ymin=CI2, ymax=CI3), linetype=2, alpha=0.1) +
#  scale_x_continuous(breaks = 0:m) +
#  scale_y_continuous(name = y_name) +
#  scale_colour_discrete(labels=leg_labs) +
#  scale_shape_discrete(labels=leg_labs) +
#  scale_fill_discrete(labels=leg_labs) +
#  scale_linetype_discrete(labels=leg_labs) + 
#  theme_set(theme_minimal())
#cd <- rep(list(array(NA, dim = c(m,d))),3)
#names(cd) <- c("cd","cd_L","cd_U")
#cd$cd[t,] <- apply(vals, FUN = mean, MARGIN = 1)
#cd$cd_L[t,] <- apply(vals, FUN = quantile, MARGIN = 1, probs = p[1])
#cd$cd_U[t,] <- apply(vals, FUN = quantile, MARGIN = 1, probs = p[2])
#cd <- lapply(cd, function(x) {
#  colnames(x) <- feature_names
#  cbind(time = 1:m, x)
#  pivot_longer(data.frame(x), all_of(feature_names))
#})
#matrix(cd)
#data.frame(test, rownames(test))

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

