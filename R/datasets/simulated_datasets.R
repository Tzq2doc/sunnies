## Data set list
# dat_continuous_XOR
# dat_unif_squared
# dat_nonrandom_squared
# dat_unif_squared_corr
# dat_unif_independent
# dat_unif_cos
# dat_unif_step
# dat_unif_XORlike
# dat_categorical_XOR
# dat_continuous_XOR

##############################
# y = 0*X_1^2 + 2*X_2^2 + ... + (d-1)X_d^2 where X_i ~ unif(-1,1) 
dat_unif_squared <- function(d = 4, n = 100, A = (2*(0:(d-1)))) {
  X <- matrix(runif(n*d,-1,1), n, d)
  y <- X^2 %*% A
  return(cbind(y,X))
}

##############################
# This is a modification of dat_unif_squared, where X is non-random
dat_nonrandom_squared <- function(d = 4, n = 100, A = (2*(0:(d-1)))) {
  X <- matrix(seq(-1,1,length.out = n*d),n,d)
  y <- X^2 %*% A
  return(cbind(y,X))
}

##############################
# This is almost non-random
dat_nonnoisy_squared <- function(d = 4, n = 100, A = (2*(0:(d-1))), sd = 0.001) {
  X <- matrix(seq(-1,1,length.out = n*d),n,d)
  for (i in 1:d) {X[,i] <- X[,i] + rnorm(n,0,sd)}
  y <- X^2 %*% A
  return(cbind(y,X))
}

##############################
# This is a modification of dat_unif_squared, where all
# the coefficients are set to be 1, and
# we set X_2 = X_1 + eps_1 and X_3 = X_1 + eps_2
# where eps_i ~ N(0, sigma^2)
dat_unif_squared_corr <- function(d = 4, n = 100, sigma = 0.2) {
  X <- matrix(runif(n*d,-1,1), n, d)
  X[,2] <- X[,1] + rnorm(n, sd = sigma)
  X[,3] <- X[,1] + rnorm(n, sd = sigma)
  y <- X^2 %*% rep(1,d)
  return(cbind(y,X))
}

##############################
# There is no relationship, just uniform on all variables
dat_unif_independent <- function(d = 4, n = 100) {
  return(matrix(runif(n*(d+1),-1,1), n, d+1))
}

##############################
# A cosine symmetric about 0
dat_unif_cos <- function(d = 4, n = 100, A = (2*(0:(d-1)))) {
  x <- matrix(runif(n*d,-pi,pi), n, d)
  y <- cos(x) %*% A
  return(cbind(y,x))
}

##############################
# A step function symmetric about 0
dat_unif_step <- function(d = 4, n = 100, A = (2*(0:(d-1)))) {
  x <- runif(n,-1,1)
  x <- matrix(runif(n*d,-1,1), n, d)
  y <- (-0.5 < x & x < 0.5) %*% A
  return(cbind(y,x))
}

###############################
# An XOR-like thing with continuity,
# Continuous features with continuous labels
dat_concon_XOR <- function(n = 100) {
  x1 <- runif(n,-1,1)
  x2 <- runif(n,-1,1)
  y <- x1*(x1 > 0 & x2 < 0) + x2*(x1 < 0 & x2 > 0) +
       x1*(x1 < 0 & x2 < 0) - x2*(x1 > 0 & x2 > 0)
  return(cbind(y,x1,x2))
}

###############################
# Discrete features with discrete labels XOR
dat_catcat_XOR <- function(n = 1e3) {
  x1 <- sample(c(rep(0,floor(n/2)),rep(1,floor(n/2))), n)
  x2 <- sample(c(rep(0,floor(n/2)),rep(1,floor(n/2))), n)
  y  <- as.integer(xor(x1,x2))
  return(cbind(y,x1,x2))
}

###############################
# Continuous features with discrete labels XOR
dat_concat_XOR <- function(n = 1e3) {
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  y  <- as.integer(xor(x1 > 0, x2 > 0))
  #plot(x1,x2, col = y + 1, main = "XOR")
  return(cbind(y,x1,x2))
}

###############################
# Counterexamples in Probability and Statistics 2.12
dat_tricky_gaussians <- function(n = 1e3) {
  x1 <- rnorm(n, 0, 1)
  x2 <- rnorm(n, 0, 1)
  x3a <- rnorm(n, 0, 2)
  x3 <- abs(x3a)*sign(x1*x2)
  y <- x1 + x2 + x3
  #plot(x2,y)
  #plot(x1,y)
  #plot(x3,y)
  return(cbind(y,x1,x2,x3))
}


#dat_variable_normal <- function(n = 1e3, d = 4, sd_vec = 1:d) {
#  X <- matrix(0, nrow = n, ncol = d)
#  Y <- vector(mode = "numeric", length = n)
#  for (i in 1:d) {
#    X[,i] <- rnorm(n, mean = 0, sd = sd_vec[i]) 
#  }
#  Y <- rowSums(X^2)
#  return(cbind(Y, X^2))
#}
#
#X <- dat_variable_normal()
#rowSums(X)
#
#YX <- dat_variable_normal()
#plot(YX[,1], YX[,2])
