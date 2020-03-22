source("utility_functions.R")
source("old/old_shapley_helpers.R")

######## TESTING CODE
d <- 4
n <- 50
X <- matrix(rnorm(n*d),n,d)
y <- X%*% (2*(0:(d-1))) + rnorm(n)

d <- 4
n <- 1000
X <- matrix(runif(n*d,-1,1),n,d)
y <- X^2 %*% (2*(0:(d-1)))

cor(y,X)
Rfast::dcor(y,X)$dcor

test_utility <- function(y, X, utility) {
  s <- old_shapley(y, X, utility = utility)
  if (all(all.equal(sum(s), utility(y,X)), s >= 0)) 
    {cat("pass")} else {cat("fail")}
}

test_utility(y, X, utility = R2)
test_utility(y, X, utility = DC)
test_utility(y, X, utility = BCDC)


##### UNUSED TESTING CODE
# R2 <- t(y) %*% X %*% solve(t(X) %*% X) %*% t(X) %*% y %*% solve(t(y) %*% y)
# summary(speedlm(y~X))$r.squared

