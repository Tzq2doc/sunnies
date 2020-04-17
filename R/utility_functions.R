library(speedglm) # speedlm
library(Rfast) # dcor bcdcor
library(dHSIC) # dHSIC
library(expm) # sqrtm

# R squared
R2 <- function(y, X) {if (length(X) == 0) {0} else {summary(speedlm(y~X))$r.squared}}
# Distance correlation
DC <- function(y, X){if (length(X) == 0) {0} else {dcor(y,X)$dcor}}
# Bias corrected distance correlation
BCDC <- function(y, X){if (length(X) == 0) {0} else {bcdcor(y,X)}}
# Affine invariant distance correlation
AIDC <- function(y,X){if (length(X) == 0) {0} else {
  dcor(y %*% sqrtm(solve(cov(y))), X %*% sqrtm(solve(cov(X))))$dcor}
}
# Hilbert Schmidt Independence Criterion
HSIC <- function(y,X){if (length(X) == 0) {0} else {dhsic(X,y)$dHSIC}}