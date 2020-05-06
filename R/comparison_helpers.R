source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

#devtools::install_github("NorskRegnesentral/shapr")
library(shapr)
library(xgboost)
library(SHAPforxgboost)








# Default xgb with 10 rounds and 50/50 test split, returns model, preds and accuracy 
basic_xgb <- function(dat, plots = F) {
  n <- nrow(dat)
  x_train <- dat[train,-1]
  colnames(x_train) <- paste0("x",1:ncol(x_train))
  x_test <- dat[-train,-1]
  colnames(x_test) <- paste0("x",1:ncol(x_train))
  y_train <- dat[train,1, drop = F]
  colnames(y_train) <- "y"
  y_test <- dat[-train, 1, drop = F]
  colnames(y_train) <- "y"
  binary <- T
  if (length(unique(y_train)) > 2) {binary <- F}
  obj <- if (binary) {"binary:logistic"} else {"reg:squarederror"}
  bst <- xgboost(
    data = x_train,
    label = y_train,
    nround = 20,
    verbose = FALSE,
    objective = obj
  )
  pred <- predict(bst, x_test)
  if (plots) {plot(pred, y_test)}
  if (binary) {
    pred <- as.numeric(pred > 0.5)
    acc <- sum(pred == y_test)/length(y_test)
    mse <- "not applicable (binary response)"
  } else {
    mse <- mean((pred - y_test)^2) 
    acc <- "not applicable (continuous response)"
  }
  attr(pred, "mse") <- mse
  attr(pred, "acc") <- acc
  return(list(bst = bst, pred = pred, x_train = x_train))
}

## Utility of each feature alone, then utility of all features together
examine_utility <- function(dat, utility) {
  for (i in 2:ncol(dat)) { 
    cat(paste0("C({",i-1,"}): ", utility(dat[,1], dat[,i])),"\n") 
  }
  cat(paste0("C([d]): ", utility(dat[,1], dat[,-1])),"\n") 
}

# Use SHAPforxgboost library to get SHAP values
examine_SHAP <- function(bst, x_train, plots = F) {
  shap_values <- shap.values(xgb_model = bst, X_train = x_train)
  shap_long <- shap.prep(xgb_model = bst, X_train = x_train)
  if (plots) {
    barplot(shap_values$mean_shap_score, main = "SHAP")
  }
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
  xgb <- basic_xgb(dat, plots = plots)
  cat("---\n")
  print(xgb.importance(model = xgb$bst))
  cat("---\n")
  cat(paste0("xgb acc: ", attributes(xgb$pred)$acc),"\n")
  cat(paste0("xgb mse: ", attributes(xgb$pred)$mse),"\n")
  cat("---\n")
  examine_utility(dat, utility)
  cat("---\n")
  shaps <- shapley(dat, utility = utility)
  if (plots) {
    U <- as.character(substitute(utility))
    barplot(shaps, main = paste0("sunnies ", U))
  }
  cat("Shapley: ", shaps,"\n")
  cat("---\n")
  SHAP <- examine_SHAP(xgb$bst, xgb$x_train, plots = plots)
  cat("SHAP: ", SHAP$shapm, "\n")
  #if (plots) {shap.plot.summary(SHAP$shapp)}
  cat("---\n")
  cat("\n")
  return(list(xgb,shapp = SHAP$shapp))
}
