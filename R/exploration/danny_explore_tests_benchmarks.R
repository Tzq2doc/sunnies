source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

# Various tests and comparisons -------------------------------------------

#### TEST with HSIC
d <- 4
n <- 100
set.seed(0)
X <- matrix(runif(n*d,-1,1), n, d)
y <- X^2 %*% (2*(0:(d-1)))

plot(X[,1], y)
points(X[,2], y, col = "green")
points(X[,3], y, col = "blue")
points(X[,4], y, col = "red")

CF <- estimate_characteristic_function(X, HSIC, y = y)
for (i in 1:4) { print(shapley(CF, v = i)) }

shapley_(CF, 1:d)

#### TEST IT OUT VERSUS THE OLD VERSION
d <- 4
n <- 100
X <- matrix(runif(n*d,-1,1),n,d)
y <- X^2 %*% (2*(0:(d-1)))

CF <- estimate_characteristic_function(X, DC, y = y)
for (i in 1:4) { print(shapley(CF, v = i)) }
old_shapley(y,X,DC)

#### TEST WITH NON-RANDOM DATA
d <- 2
n <- 5
#X <- matrix(rep(seq(-1,1,length.out = n),d),n,d)
X <- matrix(seq(-1,1,length.out = n*d),n,d)
y <- X^2 %*% (2*(0:(d-1)))

HSIC(X,y)
AIDC(y,X)

CF <- estimate_characteristic_function(X, DC, y = y)
shapley(CF, v = 1)
old_shapley(y,X,DC)

###### QUICK BENCHMARKING
shapley2 <- function(y, X) {
  CF <- estimate_characteristic_function(X, utility = function(X){R2(y,X)})
  v <- 1:ncol(X)
  out <- vector(mode = "numeric", length = length(v))
  for (i in 1:length(v)) {
    out[i] <- shapley(CF, v[i])
  }
  out
}

microbenchmark::microbenchmark(
  old_shapley(y,X, R2),
  shapley2(y, X)
)