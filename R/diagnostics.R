source("datasets/simulated_datasets.R")
source("shapley_helpers.R")
source("utility_functions.R")
source("comparison_helpers.R")
source("shapley_helpers.R")
library(xgboost)
library(dplyr)

### Goal here is to find out which feature is contributing most to 
# dependence in the res vs fits plot.

#dat <- dat_unif_independent(n = 1e4)
#dat <- dat_unif_squared_corr(n = 1e3)



###############################################################
## RESIDUALS VS X
################ Example 1
###### In this example, the diagnostic plot may lead us to suspect
# certain features, but the situation usually improves when we remove
# features by considering the dependence plot instead.
# As a bonus, we see that xgboost didn't really perform feature
# selection for us, since removing the feature was an improvement.

dat <- dat_unif_squared(n = 1e3, d = 5, add_noise = T)

sdat <- split_dat(dat)
xgb <- basic_xgb(sdat)

# Based on the labels shap plot, we decide to remove feature 1.
diagnostic_plots(sdat, xgb)
sdat2 <- split_dat(subset(dat,select = -x1))
xgb2 <- basic_xgb(sdat2)
xgb$test_mse - xgb2$test_mse # positive => improvement

# Based on the labels shap plot, we now also decide to remove feature 2 
diagnostic_plots(sdat2, xgb2)
sdat3 <- split_dat(subset(dat, select = -c(x1,x2)))
xgb3 <- basic_xgb(sdat3)
xgb3$test_mse - xgb2$test_mse # almost 0 => neutral, less features (improved)

# The situation has improved in the diagnostic plot 
# (overall dependence is low), so no need to remove anything else.
diagnostic_plots(sdat3, xgb3)
plot(sdat3$x_test[,1], xgb$pred_test)


##########################################################################
############# Failed example 2
# RESIDUALS VS FITS attempt -----------------------------------------------

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


### Here I was trying to compare shapley value of predicted labels
# to Shapley value of actual labels

sp <- shapley(cbind(preds_test, x_test), utility = DC)
sl <- shapley(cbind(y_test, x_test), utility = DC)
barplot(sp)
barplot(sl)

imp <- xgb.importance(model = model)
barplot(imp$Gain, names.arg = imp$Feature)

