source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")
source("simulations.R")

# Shapley is a decomposition, not representing pairwise correlations ------
dat <- dat_unif_squared()
y <- dat[,1, drop = F]; X <- dat[,-1, drop = F]
HSIC(y,X)
HSIC(y,X[,1, drop = F])
HSIC(y,X[,2:4, drop = F])






