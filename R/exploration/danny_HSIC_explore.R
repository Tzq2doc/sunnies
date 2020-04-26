source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

## Various ways to calculate HSIC ------------------------------------------
## See equation (4) of paper [27]

# Source 1: https://github.com/xiao-he/HSIC/blob/master/HSIC.py
# Source 2: https://github.com/cran/dHSIC/blob/master/R/dhsic.R
# Paper [27] (2007): https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf
# Paper [28] (2017): https://arxiv.org/pdf/1603.00285.pdf

d <- 2; n <- 5
X <- matrix(seq(-1,1,length.out = n*d),n,d)
y <- X^2 %*% (2*(0:(d-1)))

# To compute the Gaussian kernels

# Equivalent form 0 (but doesn't return matrix!)
gaussianK0 <- function(X) {
  dists <- dist(X)^2
  sigma <- sqrt(median(dists[dists != 0])/2)
  exp(-dists/(2*sigma^2))
}
# Equivalent form 1
gaussianK1 <- function(X) {
  GX <- X %*% t(X)
  dGX <- diag(GX) - GX
  KX <- dGX + t(dGX)
  mdist <- median(KX[KX != 0])
  sigma <- sqrt(mdist/2)
  print(sigma)
  return( exp(-KX/(2*sigma^2)) )
}
# Equivalent form 2
gaussianK <- function(X) {
  n <- nrow(X); d <- ncol(X)
  bandwidth <- dHSIC:::median_bandwidth_rcpp(X, n, d)
  return(dHSIC:::gaussian_grammat_rcpp(X, bandwidth, n, d))
}
gaussianK0(X)
gaussianK1(X)
gaussianK(X)
microbenchmark::microbenchmark(gaussianK0(X), 
                               gaussianK1(X), 
                               gaussianK(X))


k <- gaussianK(X)
l <- gaussianK(y)
# When k and l are Gaussian kernals of X and Y
# Equivalent form 0
dHSIC0 <- function(k,l) {
  n <- nrow(k)
  dterm1 <- sum(k*l)
  dterm2 <- 1/(n^4)*sum(k)*sum(l)
  dterm3 <- 2/(n^3)*sum(k %*% l)
  return(1/(n^2)*dterm1 + dterm2 - dterm3)
}
# Equivalent form 1
dHSIC1 <- function(k,l) {
  n <- nrow(k)
  dterm1 <- sum(diag(k %*% l))
  dterm2 <- 1/(n^4)*sum(k)*sum(l)
  dterm3 <- 2/(n^3)*sum(k %*% l)
  return(1/(n^2)*dterm1 + dterm2 - dterm3)
}
# Equivalent form 2
dHSIC2 <- function(k,l) {
  n <- nrow(k); H <- diag(n) - 1/n
  return(1/n^2*sum(diag(k %*% H %*% l %*% H)))
}
# Equivalent form 3
dHSIC3 <- function(k,l) {
  n <- nrow(k)
  term1 <- sum(k*l)
  term2 <- 1/n^4 * sum(k)*sum(l)
  term3 <- sum(2/n^3 * colSums(k) * colSums(l))
  return(1/len^2 * term1 + term2 - term3)
}
# Check all equal
all.equal(dHSIC0(k,l), dHSIC1(k,l), dHSIC2(k,l), dHSIC3(k,l))
microbenchmark::microbenchmark(dHSIC0(k,l), dHSIC1(k,l), 
                               dHSIC2(k,l), dHSIC3(k,l))