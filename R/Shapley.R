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
d <- 2
n <- 5
X <- matrix(rep(seq(-1,1,length.out = n),d),n,d)
y <- X^2 %*% (2*(0:(d-1)))

HSIC(X,y)

CF <- estimate_characteristic_function(X, DC, y = y)
shapley(CF, v = 1)
old_shapley(y,X,DC)


dhsic
dHSIC:::gaussian_grammat_rcpp()


median_bandwidth <- function(x) {
  bandwidth <- dHSIC:::median_bandwidth_rcpp(x[sample(1:len), 
                                            , drop = FALSE], len, ncol(x))
  if (bandwidth == 0) {
    bandwidth <- 0.001
  }
  return(bandwidth)
}

for (j in 1:d) {
  bandwidth[j] <- median_bandwidth(X[[j]])
  K[[j]] <- dHSIC:::gaussian_grammat_rcpp(X[[j]], bandwidth[j], 
                                          len, ncol(X[[j]]))
}

dHSIC:::gaussian_grammat_rcpp(X[[1]], bandwidth[1], len, ncol(X[[1]]))

term1 <- 1
term2 <- 1
term3 <- 2/len
for (j in 1:d) {
  term1 <- term1 * K[[j]]
  term2 <- 1/len^2 * term2 * sum(K[[j]])
  term3 <- 1/len * term3 * colSums(K[[j]])
}
term1 <- sum(term1)
term3 <- sum(term3)
dHSIC = 1/len^2 * term1 + term2 - term3

Xt <- X

dHSIC::dhsic(list(X,y))$dHSIC
dHSIC::dhsic(X,y)$dHSIC


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

