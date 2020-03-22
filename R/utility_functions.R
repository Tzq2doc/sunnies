# R squared
R2 <- function(y, X) {if (length(X) == 0) {0} else {summary(speedlm(y~X))$r.squared}}
# Distance correlation
DC <- function(y, X){if (length(X) == 0) {0} else {dcor(y,X)$dcor}}
# Bias corrected distance correlation
BCDC <- function(y, X){if (length(X) == 0) {0} else {bcdcor(y,X)}}