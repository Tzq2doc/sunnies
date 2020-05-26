n <- 2e3; d <- 4
X <- matrix(rnorm(n*d), nrow = n, ncol = d)
cats <- sample(1:20, n, replace = T)
feats <- cats %% d + 1

