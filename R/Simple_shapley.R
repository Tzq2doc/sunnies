source("shapley_helpers.R")


#### TEST IT OUT
source("simple_shapley_helpers.R")
d <- 4
n <- 100
X <- matrix(runif(n*d,-1,1),n,d)
y <- X^2 %*% (2*(0:(d-1)))

CF <- estimate_characteristic_function(X, DC, y = y)
simple_shapley(CF, v = 1)
shapley(y,X,DC)

#### TEST WITH NON-RANDOM DATA
d <- 4
n <- 10
X <- matrix(rep(seq(-1,1,length.out = n),d),n,d)
y <- X^2 %*% (2*(0:(d-1)))

CF <- estimate_characteristic_function(X, DC, y = y)
simple_shapley(CF, v = 1)
shapley(y,X,DC)


###### QUICK BENCHMARKING
shapley2 <- function(y, X) {
  CF <- estimate_characteristic_function(X, utility = function(X){R2(y,X)})
  v <- 1:ncol(X)
  out <- vector(mode = "numeric", length = length(v))
  for (i in 1:length(v)) {
    out[i] <- simple_shapley(CF, v[i])
  }
  out
}

microbenchmark::microbenchmark(
  shapley(y,X, R2),
  shapley2(y, X)
)
