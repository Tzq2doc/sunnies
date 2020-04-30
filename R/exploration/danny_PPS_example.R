source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")
source("simulations.R")

library(reticulate)
#use_python("C:/Users/Doony/Anaconda3/envs/r-tensorflow/python.exe")
use_python("C:/Users/Doony/.julia/conda/3/python.exe")
#use_condaenv("r-tensorflow")
library(keras)
library(xgboost)


library(tree)
library(Rfast)
# Simulate data

### The simulated example from the medium article
dat_medium_unif_square <- function(n = 1e3) {
  x <- runif(n, -2, 2)
  eps <- runif(n, -0.5, 0.5)
  data.frame(y = x^2 + eps, x = x)
}

### A linear relationship with a twist
dat_hidden_variable <- function(n = 1e3) {
  x1 <- runif(n, 0, 1)
  x2 <- runif(n, 0, 1)
  data.frame(y = x1 + 0.7*(x2 > 0.5), x = x1)
}

# An XOR-like thing with continuity
dat_unif_XORlike <- function(n = 1e3) {
  x1 <- runif(n,-1,1)
  x2 <- runif(n,-1,1)
  y <- x1*(x1 > 0 & x2 < 0) + x2*(x1 < 0 & x2 > 0) +
    x1*(x1 < 0 & x2 < 0) - x2*(x1 > 0 & x2 > 0)
  return(data.frame(y = y, x = x1))
}

#### Choose your dataset here:
dat <- dat_unif_XORlike() # dat_hidden_variable() # dat_medium_unif_square()
plot(dat$x, dat$y)

# Split train and test sets
train_ind <- sample(1:n, floor(n/2))
train <- dat[train_ind,]
valid <- dat[-train_ind,]

# Train y ~ x
t_yx <- tree::tree(y ~ x, data = train)
summary(t_yx); plot(t_yx); text(t_yx, pretty = 0)

# Train x ~ y
t_xy <- tree::tree(x ~ y, data = train)
summary(t_xy); plot(t_xy); text(t_xy, pretty = 0)

# Calculate MAE and PPS
baseline <- function(y) {sum(abs(median(y) - y))/length(y)}
MAE <- function(yhat, y) {sum(abs(yhat - y))/length(y)}
PPS <- function(MAE, baseline) {1 - MAE/baseline}
predict_PPS <- function(t, valid) {
  base <- baseline(valid$y)
  yhat <- predict(t, valid)
  return(max(0, PPS(MAE(yhat, valid$y), base)))
}

# Validate y ~ x
predict_PPS(t_yx, valid)

# Validate x ~ y
predict_PPS(t_xy, valid)

cov(dat$y, dat$x)
Rfast::dcor(cbind(dat$y), cbind(dat$x))$dcor
Rfast::dcor(cbind(dat$x), cbind(dat$y))$dcor
