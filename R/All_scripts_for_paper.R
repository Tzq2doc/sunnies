source("datasets/simulated_datasets.R")
source("shapley_helpers.R")
source("utility_functions.R")
source("comparison_helpers.R")
source("shapley_helpers.R")
source("Applications_helpers.R")
library(xgboost)
library(dplyr)

### ERDA Example 1
# The repetitions producing violin plots is in python, but here is similar:
result1 <- run_evaluations(dat_unif_squared, utility = DC, n = 1e3, plots = T)

### ERDA Examples 2 and 3 (which should be the same example)
# This example should be done exactly by hand, but here is a simulation.
# When it is calculated exactly, the shapley values will be equal to each other:
result2 <- run_evaluations(dat_catcat_XOR, utility = DC, n = 1e3, plots = T)

### DARRP EXAMPLE 1
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
m <- 10; d <- 4
shaps_lab <- matrix(0, nrow = m+1, ncol = d) 
shaps_res <- matrix(0, nrow = m+1, ncol = d)
mse <- vector(mode = "numeric", length = m+1)
datt <- dat_t(n, d, t = 0, max_t = m)
sdatt <- split_dat(datt)
xgb <- basic_xgb_fit(sdatt) # Model is only fit once
cat("Number of features: ", xgb$nfeatures)
for (t in 0:m) {
  xgbtt <- basic_xgb_test(xgb, sdatt) # Model is tested each time
  mse[t+1] <- xgbtt$test_mse
  diagn <- diagnostics(sdatt, xgbtt, plot = "rvf",
                       features = 1:4)
  shaps_lab[t+1,] <- diagn$shap_lab
  shaps_res[t+1,] <- diagn$shap_res
  datt <- dat_t(n, d, t = t, max_t = m)
  sdatt <- split_dat(datt)
}
#pdf("diagnostics_drift.pdf", width = 6, height = 3)
#par(mfrow = c(1,3))
matplot(x = 0:m, y = shaps_res, type = 'b', 
        xlab = "time", ylab = "residual importance")
matplot(x = 0:m, y = shaps_lab, type = 'b', 
        xlab = "time", ylab = "response importance")
plot(0:m, mse, type = 'b', xlab = "time", ylab = "MSE")
#dev.off()

### DARRP EXAMPLE 2
n <- 1e4; d <- 5
X <- matrix(rnorm(n*(d-2),0,1), nrow = n, ncol = (d-2))
X4 <- sample(0:1, replace = T, n)
X5 <- sample(0:1, replace = T, n)
y <- rowSums(X[,-(d-2)]) + 5*(X4 & X5)*X[,(d-2)] + rnorm(n,0,0.1)
dat <- cbind(y,X,X4,X5)
sdat <- split_dat(dat, df = T)
lmodel <- lm(y ~ x1 + x2 + x3 + x4 + x5, dat = sdat$df_yx_train)
check_model <- function(sdat, lmodel) {
  lm_pred_test <- predict(lmodel, data.frame(sdat$x_test))
  plot(sdat$y_test, lm_pred_test, xlab = "labels", ylab = "predictions")
  slm <- summary(lmodel); slm
  slabs <- shapley(sdat$y_test, sdat$x_test, utility = DC)
  spred <- shapley(lm_pred_test, sdat$x_test, utility = DC)
  sresd <- shapley(round(lm_pred_test - sdat$y_test, digits = 10), 
                   sdat$x_test, utility = DC)
  barplot(rbind(slabs, spred, sresd),
          xlab = "Feature",
          ylab = "Attribution",
          col = c("black","gray","red"),
          beside = T)
  legend(x = "top", legend = c("labels","predictions","residuals"), 
         col = c("black","gray","red"), pch = c(15,15,15))
}
check_model(sdat,lmodel)
lmodel_perfect <- lm(y ~ x1 + x2 + x3:x4:x5, dat = sdat$df_yx_train)
check_model(sdat,lmodel_perfect)

### DARRP EXAMPLE 3
# This will be simpson's paradox (if we include it).


