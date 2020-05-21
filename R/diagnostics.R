source("datasets/simulated_datasets.R")
source("shapley_helpers.R")
source("utility_functions.R")
source("comparison_helpers.R")
source("shapley_helpers.R")
library(xgboost)
library(dplyr)

########################################################
######### RESIDUAL DEPENDENCE ATTRIBUTION
### Feature drift detection: when DGP changes over time.

### Goal here is to find out which feature is contributing most to 
# dependence in the res vs fits plot.

# At time = 0, Y is just a sum of X
n <- 1e3; d <- 4
X <- matrix(rnorm(n*d), nrow = n, ncol = d)
y <- rowSums(X)
dat <- cbind(y,X)

sdat <- split_dat(dat)
xgb <- basic_xgb_fit(sdat)
xgbt <- basic_xgb_test(bst, sdat)
diagnostics(sdat, xgbt)

# Then, at time = 1, the last feature loses its influence
X <- matrix(rnorm(n*d), nrow = n, ncol = d)
y <- rowSums(X[,-d])
dat2 <- cbind(y,X)
sdat2 <- split_dat(dat2)
xgbt2 <- basic_xgb_test(bst, sdat2)
diagnostics(sdat2, xgbt2)

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
pdf("diagnostics_drift.pdf", width = 6, height = 3)
par(mfrow = c(1,3))
  matplot(x = 0:m, y = shaps_res, type = 'b', 
          xlab = "time", ylab = "residual importance")
  matplot(x = 0:m, y = shaps_lab, type = 'b', 
          xlab = "time", ylab = "response importance")
  plot(0:m, mse, type = 'b', xlab = "time", ylab = "MSE")
dev.off()
summary(lm(t ~ ., data = data.frame(t = 0:m, shaps_res)))




## RESIDUAL DEPENDENCE ATTRIBUTION
################ Example 2 (unused)
###### In this example, the diagnostic plot may lead us to suspect
# certain features, but the situation usually improves when we remove
# features by considering the dependence plot instead.
# As a bonus, we see that xgboost didn't really perform feature
# selection for us, since removing the feature was an improvement.

dat <- dat_unif_squared(n = 1e3, d = 5, add_noise = T)

sdat <- split_dat(dat)
xgb <- basic_xgb(sdat)

# Based on the labels shap plot, we decide to remove feature 1.
diagnostics(sdat, xgb)
sdat2 <- split_dat(subset(dat,select = -x1))
xgb2 <- basic_xgb(sdat2)
xgb$test_mse - xgb2$test_mse # positive => improvement

# Based on the labels shap plot, we now also decide to remove feature 2 
diagnostics(sdat2, xgb2)
sdat3 <- split_dat(subset(dat, select = -c(x1,x2)))
xgb3 <- basic_xgb(sdat3)
xgb3$test_mse - xgb2$test_mse # almost 0 => neutral, less features (improved)

# The situation has improved in the diagnostic plot 
# (overall dependence is low), so no need to remove anything else.
diagnostics(sdat3, xgb3)
plot(sdat3$x_test[,1], xgb$pred_test)


##########################################################################
############# Failed example 2
# RESIDUALS VS ACTUAL FITS attempt ---------------------------------------
## This attempt involved introducing a mean drop method.

dat <- dat_unif_squared(n = 1e3, d = 5, add_noise = T)

sdat <- split_dat(dat)
xgb <- basic_xgb(sdat)

shaps <- shapley(sdat$y_test, sdat$x_test, 
                 utility = DC_rf, 
                 model = xgb$bst, drop_method = "mean")
barplot(shaps, main = "res vs fits shap")

shaps2 <- shapley(xgb$residuals_test, sdat$x_test, utility = DC)
barplot(shaps2, main = "res vs X shap")

shaps3 <- shapley(sdat$y_test, sdat$x_test, utility = DC)
barplot(shaps3, main = "labels shap")

normalise <- function(x) {(x-mean(x))/sd(x)}

### Here I was comparing shapley value of predicted labels
# to Shapley value of actual labels

sp <- shapley(cbind(preds_test, x_test), utility = DC)
sl <- shapley(cbind(y_test, x_test), utility = DC)
barplot(sp)
barplot(sl)

imp <- xgb.importance(model = model)
barplot(imp$Gain, names.arg = imp$Feature)

