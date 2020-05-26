source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")



library(tree)
library(Rfast)
#### Simulate data

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
  data.frame(y = x1 + 0.7*(x2 > 0.5), x = x1, hidden = x2)
}

# An XOR-like thing with continuity
dat_unif_XORlike <- function(n = 1e3) {
  x1 <- runif(n,-1,1)
  x2 <- runif(n,-1,1)
  y <- x1*(x1 > 0 & x2 < 0) + x2*(x1 < 0 & x2 > 0) +
    x1*(x1 < 0 & x2 < 0) - x2*(x1 > 0 & x2 > 0)
  return(data.frame(y = y, x = x1, hidden = x2))
}

#### Choose your dataset here:
n <- 1e3
dat <- dat_unif_XORlike(n = n) # dat_hidden_variable() # dat_medium_unif_square()
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

# Train y ~ x + hidden
t_full <- tree::tree(y ~ x + hidden, data = train)
summary(t_full); plot(t_full); text(t_full, pretty = 0)

# Train y ~ hidden
t_hid <- tree::tree(y ~ hidden, data = train)
summary(t_full); plot(t_full); text(t_full, pretty = 0)

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

# Validate y ~ hidden
predict_PPS(t_hid, valid)

# Validate x ~ y
predict_PPS(t_xy, valid)

# Validate y ~ x + hidden
predict_PPS(t_full, valid)

cov(dat$y, dat$x)
Rfast::dcor(cbind(dat$y), cbind(dat$x))$dcor
Rfast::dcor(cbind(dat$x), cbind(dat$y))$dcor


### My comments from Slack conversation
# 
# In this example you have y as a one-to-one "XOR-like" 
# function of (x, z), but if we look at y as a function 
# of x alone, then the relationship appears one-to-many 
# (as pictured). The PPS scores of x -> y, and z -> y are 
# both 0, so that an exploratory analysis would possibly 
# lead us to ignore the relationship (x,z) -> y.
# The distance correlation, on the other hand, is 
# around 0.5 for both y ~ x and y ~ z. So the relationship 
# would be detected.
# If both x and z are used to predict y together, 
# then a very accurate decision tree is produced. 
# But PPS does not consider all possible combinations 
# of the explanatory variables, so (x,z) -> y is ignored. 
# There do exist methods that do look at all combinations. 
# For example, see the SHAP package https://github.com/slundberg/shap 
# for shapley values.

# The XOR function, for binary categorical 
# variables, is a great example where you can 
# have three random variables z,x,y with Z = XOR(x,y) 
# that are "causally" (or at least "functionally") dependent 
# but are pairwise statistically independent (though still 
# mutually statistically dependent). 
# See https://stats.stackexchange.com/questions/286741/is-there-an-example-of-two-causally-dependent-events-being-logically-probabilis

