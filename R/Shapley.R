source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")

### Parameter names
n = 1000  # n: sample size (used 1000 for saved data)
d = 4     # d: number of features (used 4 for saved data)
N = 1000  # N: number of iterations simulating data (used 1000 for saved data)


#### Compare them all * 1000
results <- list()
utilities <- c("R2","DC","BCDC","AIDC","HSIC")
for (i in 1:length(utilities)) {
  results[[utilities[i]]] <- shapleyN(get(utilities[i]), N, n , d)
}
#saveRDS(results, "results/compare_them_all_1000.Rds")
results <- readRDS("results/compare_them_all_1000.Rds")
res_means <- lapply(results, function(r) {apply(r, FUN = mean, MARGIN = 2)})
for (u in utilities) { barplot(res_means[[u]], main = u) }

# y = 2*0*x_1^2 + 2*1*x_2^2 + 2*2*x_3^2 + ... + 2*(d-1)*x_{d-1}^2
plot(X[,1], y)
points(X[,2], y, col = "green")
points(X[,3], y, col = "blue")
points(X[,4], y, col = "red")

barplot(results[1,])
barplot(results[2,])


#### TEST with HSIC
d <- 4
n <- 100
set.seed(0)
X <- matrix(runif(n*d,-1,1), n, d)
y <- X^2 %*% (2*(0:(d-1)))

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
d <- 4
n <- 100
#X <- matrix(rep(seq(-1,1,length.out = n),d),n,d)
X <- matrix(seq(-1,1,length.out = n*d),n,d)
y <- X^2 %*% (2*(0:(d-1)))

HSIC(X,y)
AIDC(y,X)

CF <- estimate_characteristic_function(X, DC, y = y)
shapley(CF, v = 1)
old_shapley(y,X,DC)

# See equation (4) of paper [27] 
dterm1 <- sum(diag(k %*% l))
dterm2 <- 1/(len^4)*sum(k)*sum(l)
dterm3 <- 2/(len^3)*sum(k %*% l)
dHSIC <- 1/(len^2)*dterm1 + dterm2 - dterm3


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

