source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")
library(xgboost)

# Default xgb with 10 rounds and 50/50 test split, returns model, preds and accuracy 
basic_xgb <- function(dat) {
  n <- nrow(dat)
  train <- sample(1:n, floor(n/2))
  y_train <- dat[train,1]
  binary <- T
  if (length(unique(y_train)) > 2) {binary <- F}
  obj <- if (binary) {"binary:logistic"} else {"reg:squarederror"}
  bst <- xgboost(data = dat[train,-1], label = y_train, 
                 nrounds = 10, objective = obj)
  pred <- predict(bst, dat[-train,-1])
  if (binary) {
    pred <- as.numeric(pred > 0.5)
    acc <- sum(pred == dat[-train,1])/length(dat[-train,1])
    mse <- "not applicable (binary response)"
  } else {
    mse <- mean((pred - dat[-train,1])^2) 
    acc <- "not applicable (continuous response)"
  }
  attr(pred, "mse") <- mse
  attr(pred, "acc") <- acc
  return(list(bst = bst, pred = pred))
}

## Utility of each feature alone, then utility of all features together
examine_utility <- function(dat, utility) {
  for (i in 2:ncol(dat)) { 
    cat(paste0("C({",i-1,"}): ", utility(dat[,1], dat[,i])),"\n") 
  }
  cat(paste0("C([d]): ", utility(dat[,1], dat[,-1])),"\n") 
}

# Run and print all the evaluations, returning the xgb model
## Parameters:
# data_gen: data generating function
# utility: utility function
# n: sample size
# ...: other agurments to data_gen
run_evaluations <- function(data_gen, utility, n = 1e3, ...) {
  dgp_name <- toupper(as.character(substitute(data_gen)))
  cat(paste0("\n-----",dgp_name,"-----\n"))
  dat <- data_gen(n = n, ...)
  xgb <- basic_xgb(dat)
  cat("---\n")
  print(xgb.importance(model = xgb$bst))
  cat("---\n")
  cat(paste0("xgb acc: ", attributes(xgb$pred)$acc),"\n")
  cat(paste0("xgb mse: ", attributes(xgb$pred)$mse),"\n")
  cat("---\n")
  examine_utility(dat, DC)
  cat("---\n")
  shapl <- paste0(shapley(dat, utility = DC), collapse = " ")
  cat(paste0("Shapley: ", shapl),"\n")
  cat("---\n")
  cat("\n")
  return(xgb)
}