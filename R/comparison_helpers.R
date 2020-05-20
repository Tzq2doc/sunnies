source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

#devtools::install_github("NorskRegnesentral/shapr")
library(shapr)
library(xgboost)
library(SHAPforxgboost)

split_dat <- function(dat) {
  n <- nrow(dat)
  train <- sample(1:n, floor(n/2))
  x_train <- dat[train,-1]
  colnames(x_train) <- paste0("x",1:ncol(x_train))
  x_test <- dat[-train,-1]
  colnames(x_test) <- paste0("x",1:ncol(x_train))
  y_train <- dat[train,1, drop = F]
  colnames(y_train) <- "y"
  y_test <- dat[-train, 1, drop = F]
  colnames(y_train) <- "y"
  return(list(dat = dat, 
              x_train = x_train,
              y_train = y_train,
              x_test = x_test,
              y_test = y_test))
}

diagnostic_plots <- function(sdat,xgb) {
  plot(xgb$pred_test, xgb$residuals_test,
       ylab = "Residuals", xlab = "Fitted values",
       main = "res vs fits")
  abline(h = 0, col = "red")
  barplot(shapley(xgb$residuals_test, sdat$x_test, utility = DC),
          main = "residuals shap (test set)")
  barplot(shapley(sdat$y_train, sdat$x_train, utility = DC),
          main = "labels shap (training set)")
}

# Default xgb with 10 rounds and 50/50 test split, returns model, preds and accuracy 
basic_xgb <- function(dat, plots = F) {
  binary <- T
  if (length(unique(dat$y_train)) > 2) {binary <- F}
  obj <- if (binary) {"binary:logistic"} else {"reg:squarederror"}
  bst <- xgboost(
    data = dat$x_train,
    label = dat$y_train,
    nround = 20,
    verbose = FALSE,
    objective = obj
  )
  pred_test <- predict(bst, dat$x_test)
  pred_train <- predict(bst, dat$x_train)
  residuals_test <- dat$y_test - pred_test
  residuals_train <- dat$y_train - pred_train
  if (plots) {plot(pred_test, dat$y_test)}
  if (binary) {
    pred_test <- as.numeric(pred_test > 0.5)
    acc <- sum(pred_test == dat$y_test)/length(dat$y_test)
    mse <- "not applicable (binary response)"
  } else {
    mse <- mean((pred_test - dat$y_test)^2) 
    acc <- "not applicable (continuous response)"
  }
  test_mse <- mse
  test_acc <- acc
  return(list(bst = bst, 
              pred_test = pred_test,
              test_mse = test_mse,
              test_acc = test_acc,
              pred_train = pred_train,
              residuals_test = residuals_test,
              residuals_train = residuals_train))
}

## Utility of each feature alone, then utility of all features together
examine_utility <- function(dat, utility) {
  for (i in 2:ncol(dat)) { 
    cat(paste0("C({",i-1,"}): ", utility(dat[,1,drop=F], dat[,i,drop=F])),"\n") 
  }
  cat(paste0("C([d]): ", utility(dat[,1,drop=F], dat[,-1,drop=F])),"\n") 
}

# Use SHAPforxgboost library to get SHAP values
examine_SHAP <- function(bst, x_train, plots = F) {
  shap_values <- shap.values(xgb_model = bst, X_train = x_train)
  shap_long <- shap.prep(xgb_model = bst, X_train = x_train)
  return(list(shapm = shap_values$mean_shap_score, 
              shapp = shap_long))
}

# Run and print all the evaluations, returning the xgb model
## Parameters:
# data_gen: data generating function
# utility: utility function
# n: sample size
# ...: other agurments to data_gen
run_evaluations <- function(data_gen, utility, n = 1e3,  plots = F, ...) {
  dgp_name <- toupper(as.character(substitute(data_gen)))
  cat(paste0("\n-----",dgp_name,"-----\n"))
  dat <- data_gen(n = n, ...)
  dat <- split_dat(dat)
  xgb <- basic_xgb(dat, plots = plots)
  cat("---\n")
  imp <- xgb.importance(model = xgb$bst)
  print(imp)
  cat("---\n")
  cat(paste0("xgb acc: ", xgb$test_acc,"\n"))
  cat(paste0("xgb mse: ", xgb$test_mse,"\n"))
  cat("---\n")
  examine_utility(dat$dat, utility)
  cat("---\n")
  shaps_train <- shapley(cbind(dat$y_train, dat$x_train), utility = utility)
  shaps_test <- shapley(cbind(dat$y_test, dat$x_test), utility = utility)
  shaps_preds_train <- shapley(
    cbind(xgb$pred_train, dat$x_train), utility = utility)
  shaps_preds_test <- shapley(
    cbind(xgb$pred_test, dat$x_test), utility = utility)
  shaps_diff_train <- shaps_preds_train - shaps_train
  shaps_diff_test <- shaps_preds_test - shaps_test
  shaps_res_train <- shapley(xgb$residuals_train, dat$x_train, utility = DC)
  shaps_res_test <- shapley(xgb$residuals_test, dat$x_test, utility = DC)
  cat("Sunnies test: ", shaps_test, "\n")
  cat("Sunnies train: ", shaps_train, "\n")
  cat("Sunnies preds train: ", shaps_preds_train, "\n")
  cat("Sunnies preds test: ", shaps_preds_test, "\n")
  cat("Sunnies diffs (pred-lab) train: ", shaps_diff_test, "\n")
  cat("Sunnies diffs (pred-lab) test: ", shaps_diff_test, "\n")
  cat("---\n")
  SHAP_train <- examine_SHAP(xgb$bst, dat$x_train, plots = plots)
  SHAP_test <- examine_SHAP(xgb$bst, dat$x_test, plots = plots)
  if (plots) {
    barplot(imp$Gain, main = "xgb.importance", names.arg = imp$Feature)
    barplot(SHAP_train$shapm, main = "SHAP train")
    barplot(SHAP_test$shapm, main = "SHAP test")
    U <- as.character(substitute(utility))
    plot(xgb$pred_test, xgb$residuals_test,
         ylab = "Residuals", xlab = "Fitted values",
         main = "res vs fits")
    abline(h = 0, col = "red")
    barplot(shaps_test, main = paste0("sunnies test "))
    barplot(shaps_train, main = paste0("sunnies train "))
    barplot(shaps_preds_test, main = paste0("sunnies preds test "))
    barplot(shaps_preds_train, main = paste0("sunnies preds train "))
    barplot(shaps_diff_test, main = paste0("s(pred)-s(lab) test "))
    barplot(shaps_diff_train, main = paste0("s(pred)-s(lab) train "))
    barplot(shaps_res_test, main = "s(lab - pred) test ")
    barplot(shaps_res_train, main = "s(lab - pred) train ")
  }
  cat("SHAP train: ", SHAP_train$shapm, "\n")
  cat("SHAP test: ", SHAP_test$shapm, "\n")
  #if (plots) {shap.plot.summary(SHAP$shapp)}
  cat("---\n")
  cat("\n")
  return(list(xgb = xgb, 
              shapp_train = SHAP_train$shapp,
              shapp_test = SHAP_test$shapp,
              dat = dat))
}
