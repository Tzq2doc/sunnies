source("datasets/simulated_datasets.R")
source("shapley_helpers.R")
source("utility_functions.R")
source("comparison_helpers.R")
source("shapley_helpers.R")
library(xgboost)
library(dplyr)

########################################################
######### PREDICTION DEPENDENCE ATTRIBUTION EXAMPLE 3
# Simpson's paradox thing: between and within group effects

# The chance of being 1 increases with group number
n <- 3e3
n_groups <- 10 # when I change it to 100 p0 screws up - need fix
p0 <- 0.1 # Probability of being in group 0 (and then it grows linearly by group)
p0 <- p0*n_groups/2
d <- 4 # The number of low impact variables
X <- matrix(rnorm(n*d,0,1), nrow = n, ncol = d)
p_group <- seq(from = p0, to = 1-p0, length.out = n_groups)
p_group <- p_group/sum(p_group)
group <- sample(0:(n_groups-1), n, replace = T, prob = p_group)
p_bin <- group/(n_groups-1) # probability that bin = 1 in each group
binary <- rbinom(n, 1, p_bin)
# If there are more 1s in a group, then the response tends to be higher.
# Within a group, then the response tends to be lower for 1s.
beta_0 <- 0# intercept
beta_1 <- -0.5 # The within-effect of binary: beta_1*(binij - p_groupj)
beta_2 <- 0.72 # The between-effect of binary: beta_2*p_groupj
beta_3 <- 0.0001 # the impact of low-impact variables

y <- beta_0 + beta_1*(binary - p_bin) + beta_2*p_bin + rnorm(n,0,0.1)
  #beta_3*rowSums(X) + rnorm(n,0,0.1)

## Plot the effect of group k
k <- 8
index_k <- which(group == k)
yk <- y[index_k]
bink <- binary[index_k]
index_0 <- which(bink == 0)
plot(bink, yk, main = paste0("Group ", k), xlab = "binary", ylab = "y")
lines(x = c(0,1), 
      y = c(mean(yk[index_0]), mean(yk[-index_0])),
      col = "red", lwd = 2)

# Plot the effect of binary
plot(binary,y)
lines(x = c(0,1), 
      y = c(mean(y[which(binary == 0)]), mean(y[which(binary == 1)])),
      col = "red", lwd = 2)

dat <- cbind(y, binary, group)#, X)
sdat <- split_dat(dat)
xgb <- basic_xgb_fit(sdat)
xgbt <- basic_xgb_test(xgb, sdat)

# The model is doing great overall
plot(xgbt$pred_test, sdat$y_test)

diagn <- diagnostics(sdat, xgbt, plot = "all",
                      features = 1:2)
diagn

# But lets look at predictions on just one group
k <- c(1:2)
index_k <- which(group %in% k)
x_group_k <- dat[index_k,-1]
y_group_k <- dat[index_k,1]
colnames(x_group_k) <- paste0("x",1:ncol(x_group_k))
sdatk <- split_dat(dat[index_k,])
xgbtk <- basic_xgb_test(xgb, sdatk)
diagnk <- diagnostics(sdatk, xgbtk, plot = "all",
                      features = 1:2)

diagnk
#pred_group_k <- predict(xgb$bst, x_group_k)
plot(xgbtk$pred_test, sdatk$y_test)



########################################################
######### PREDICTION DEPENDENCE ATTRIBUTION EXAMPLE 1
##### This one is pretty standard and it works 
n <- 1e3; d <- 4
X <- matrix(rnorm(n*d,0,1), nrow = n, ncol = d)
y <- rowSums(X[,-d]) + X[,d]^2 + rnorm(n,0,0.5)
dat <- cbind(y,X)

# ### PREDICTION DEPENDENCE ATTRIBUTION EXAMPLE 2
# ## I want to use this when we have shapley interaction values
# n <- 1e3; d <- 4
# X <- matrix(rnorm(n*d,0,1), nrow = n, ncol = d)
# y <- rowSums(X[,-d]) + X[,d]*X[,d-1] +  rnorm(n,0,0.5)
# dat <- cbind(y,X)

# ### PREDICTION DEPENDENCE ATTRIBUTION EXAMPLE 3
# ## Linear model with XOR categorical effect
n <- 1e3; d <- 5
X <- matrix(rnorm(n*(d-2),0,1), nrow = n, ncol = (d-2))
X1 <- sample(0:1, replace = T, n)
X2 <- sample(0:1, replace = T, n)
y <- rowSums(X[,-(d-2)]) + xor(X1,X2) + xor(X1,X2)*X[,(d-2)]
y <- rowSums(X[,-(d-2)]) + 5*(X1 & X2)*X[,(d-2)] 
dat <- cbind(y,X,X1,X2) #,X1,X2


sdat <- split_dat(dat, df = T)

#lmodel <- lm(y ~ . + I(sign(x2)*x3*x4), dat = sdat$df_yx_train)
lmodel <- lm(y ~ x1 + x2 + x3 + x4 + x5, dat = sdat$df_yx_train)
lmodel_perfect <- lm(y ~ x1 + x2 + x3:x4:x5, dat = sdat$df_yx_train)
lm_pred_test <- predict(lmodel, data.frame(sdat$x_test))
plot(sdat$y_test, lm_pred_test, xlab = "labels", ylab = "predictions")
slm <- summary(lmodel); slm
coefs <- slm$coefficients[,"Estimate"]
#plot(lmodel)

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

for (i in 1:d) {
  plot(sdat$y_test, sdat$x_test[,i], main = paste0("Feature x",i))
  abline(a = coefs[1], b = coefs[i+1], col = "red")
}

lmodel2 <- lm(y ~ . + I(x4^2), dat = sdat$df_yx_train)

lm_pred_test2 <- predict(lmodel2, data.frame(sdat$x_test))
plot(sdat$y_test, lm_pred_test2, xlab = "labels", ylab = "predictions")
slm <- summary(lmodel2); slm
coefs <- slm$coefficients[,"Estimate"]
#plot(lmodel)

slabs <- shapley(sdat$y_test, sdat$x_test, utility = DC)
spred <- shapley(lm_pred_test2, sdat$x_test, utility = DC)
sresd <- shapley(lm_pred_test2 - sdat$y_test, sdat$x_test, utility = DC)
barplot(rbind(slabs, spred, sresd),
        xlab = "Feature",
        ylab = "Attribution",
        col = c("black","gray","red"),
        beside = T)
legend(x = "top", legend = c("labels","predictions","residuals"), 
       col = c("black","gray","red"), pch = c(15,15,15))

# There always seems to be more correlation with the predictions,
# which makes a lot of sense if the model is biased. 

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

