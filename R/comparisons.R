source("comparison_helpers.R")

U <- DC
n <- 1e3
xgb01 <- run_evaluations(dat_concat_XOR         , U, n)
xgb09 <- run_evaluations(dat_catcat_XOR         , U, n) 
xgb08 <- run_evaluations(dat_concon_XOR         , U, n)
xgb02 <- run_evaluations(dat_unif_squared       , U, n)
xgb03 <- run_evaluations(dat_nonrandom_squared  , U, n)
xgb04 <- run_evaluations(dat_unif_squared_corr  , U, n)
xgb05 <- run_evaluations(dat_unif_independent   , U, n)
xgb06 <- run_evaluations(dat_unif_cos           , U, n) 
xgb07 <- run_evaluations(dat_unif_step          , U, n)


library(shapr)

##### NOISY DATA EXAMPLE
n <- 1e3
x1 <- rnorm(n, 0, 1)
x2 <- rnorm(n, 0, 4)
x3 <- rnorm(n, 0, 6)
y <- x1 + rnorm(n, 0, 4)
dat <- cbind(y,x1,x2,x3)

barplot(shapley(y, dat[,-1], DC))
#barplot(shapley(y, dat[,-1], HSIC))


#n <- 1e3
#x1 <- rnorm(n, 0, 1)
#x2 <- 2*x1 + rnorm(n, 0.001)
#x3 <- 3*x1  + rnorm(n, 0.001)
#y <- x1 + x2 + x3
#dat <- cbind(y,x1,x2,x3)

# Produce the simulated data
#dat <- dat_catcat_XOR(n = 1e3)
n <- nrow(dat)
train <- sample(1:n, floor(n/2))
x_train <- dat[train,-1]
colnames(x_train) <- paste0("x",1:ncol(x_train))
x_test <- dat[-train,-1]
colnames(x_test) <- paste0("x",1:ncol(x_train))
y_train <- dat[train,1, drop = F]
colnames(y_train) <- "y"
binary <- T
if (length(unique(y_train)) > 2) {binary <- F}
obj <- if (binary) {"binary:logistic"} else {"reg:squarederror"}

## Fit XGBoost model
model <- xgboost(
  data = x_train,
  label = y_train,
  nround = 20,
  verbose = FALSE
)


sum((predict(model, x_test) - dat[-train,1])^2)/nrow(x_test)

#
## Prepare the data for explanation
#explainer <- shapr(x_train, model)
#
## Specifying the phi_0, i.e. the expected prediction without any features
#p <- mean(y_train)
#
## Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
## the empirical (conditional) distribution approach with bandwidth parameter sigma = 0.1 (default)
#N <- 50
#explanation <- explain(
#  x_test[sample(1:nrow(x_test),N),],
#  approach = "empirical",
#  explainer = explainer,
#  prediction_zero = p
#)
#print(explanation$dt)
#expl <- explanation$dt
#expl

#install.packages("SHAPforxgboost")
library(SHAPforxgboost)
## https://github.com/liuyanguu/SHAPforxgboost

mod <- model
dataX <- x_train
# To return the SHAP values and ranked features by mean|SHAP|
shap_values <- shap.values(xgb_model = mod, X_train = dataX)

# The ranked features by mean |SHAP|
shap_values$mean_shap_score
barplot(shap_values$mean_shap_score)

# To prepare the long-format data:
shap_long <- shap.prep(xgb_model = mod, X_train = dataX)
# # is the same as: using given shap_contrib
# shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = dataX)
# # (Notice that there will be a data.table warning from `melt.data.table` due to `dayint` coerced from integer to double)

# **SHAP summary plot**
shap.plot.summary(shap_long)

# sometimes for a preview, you want to plot less data to make it faster using `dilute`
shap.plot.summary(shap_long, x_bound  = 1.2, dilute = 10)

# # Alternatives options to make the same plot:
# # option 1: start with the xgboost model
# shap.plot.summary.wrap1(mod, X = as.matrix(dataX))
# 
# # option 2: supply a self-made SHAP values dataset (e.g. sometimes as output from cross-validation)
# shap.plot.summary.wrap2(shap_values$shap_score, as.matrix(dataX))

## **SHAP dependence plot**
# if without y, will just plot SHAP values of x vs. x
shap.plot.dependence(data_long = shap_long, x = "x2")


# optional to color the plot by assigning `color_feature` (Fig.A)
shap.plot.dependence(data_long = shap_long, x= "x2",
                     color_feature = "x2")

# optional to put a different SHAP values on the y axis to view some interaction (Fig.B)      
shap.plot.dependence(data_long = shap_long, x= "x2",
                     y = "x3", color_feature = "x3")


# prepare the data using: 
# (this step is slow since it calculates all the combinations of features. This example spends 10s.)
shap_int <- predict(mod, as.matrix(dataX), predinteraction = TRUE)

# **SHAP interaction effect plot **
shap.plot.dependence(data_long = shap_long,
                     data_int = shap_int,
                     x = "x2",
                     y = "x3", 
                     color_feature = "x3")

# choose to show top 4 features by setting `top_n = 4`, set 6 clustering groups.  
plot_data <- shap.prep.stack.data(shap_contrib = shap_values$shap_score, top_n = 4, n_groups = 6)

# choose to zoom in at location 500, set y-axis limit using `y_parent_limit`  
# it is also possible to set y-axis limit for zoom-in part alone using `y_zoomin_limit`  
shap.plot.force_plot(plot_data, zoom_in_location = 500, y_parent_limit = c(-1,1))

# plot by each cluster
shap.plot.force_plot_bygroup(plot_data)





# Optionally set colours using RColorBrewer
library(RColorBrewer)
display.brewer.all()
cols = brewer.pal(11, "Spectral")
pal = colorRampPalette(cols)
expl$order = findInterval(expl$none, sort(expl$none))

plot(expl$x4, rep(4,nrow(expl)), 
     col=pal(nrow(expl))[expl$order], ylim = c(0,5))
points(expl$x3, rep(3,nrow(expl)),
       col=pal(nrow(expl))[expl$order])
points(expl$x2, rep(2,nrow(expl)),
       col=pal(nrow(expl))[expl$order])
points(expl$x1, rep(1,nrow(expl)),
       col=pal(nrow(expl))[expl$order])