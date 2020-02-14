library(speedglm)
library(Rfast)

R2 <- function(y,X) {summary(speedlm(y~X))$r.squared}
DC <- function(y,X){dcor(y,X)$dcor}
BCDC <- function(y,X){bcdcor(y,X)}

# This could definitely be faster
shapley <- function(y, X, utility, ...) {
  n <- nrow(X)
  d <- ncol(X)
  
  count <- 0
  R2 <- c()
  weights <- c()
  sign <- c()
  belong <- c()
  for (ii in 0:(d-1)) {
    comb <- combn(d-1,ii)
    for (jj in 1:d) {
      indep <- (1:d)[-jj]
      if (ii == 0) {
        count <- count + 1
        weights[count] <- factorial(ii)*factorial(d-ii-1)/factorial(d)
        sign[count] <- 1
        belong[count] <- jj
        R2[count] <- utility(y,X[,jj])
      } else {
        for (kk in 1:ncol(comb)) {
          count <- count + 1
          weights[count] <- factorial(ii)*factorial(d-ii-1)/factorial(d)
          sign[count] <- 1
          belong[count] <- jj
          R2[count] <- utility(y,X[,c(jj,indep[comb[,kk]])])
          
          count <- count + 1
          weights[count] <- factorial(ii)*factorial(d-ii-1)/factorial(d)
          sign[count] <- -1
          belong[count] <- jj
          R2[count] <- utility(y,X[,c(indep[comb[,kk]])])
        }
      }
    }
  }
  
  shapley <- numeric(d)
  for (ii in 1:d) {
    shapley[ii] <- sum(R2[which(belong==ii)]*weights[which(belong==ii)]*sign[which(belong==ii)])
  }
  
  shapley
}