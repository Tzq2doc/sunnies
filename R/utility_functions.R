R2 <- function(y, X) {if (length(X) == 0) {0} else {summary(speedlm(y~X))$r.squared}}
DC <- function(y, X){if (length(X) == 0) {0} else {dcor(y,X)$dcor}}
BCDC <- function(y, X){if (length(X) == 0) {0} else {bcdcor(y,X)}}