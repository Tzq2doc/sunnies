############ NONLINEAR DEPENDENCE EXAMPLE
# Helper
cube_root <- Vectorize(function(x) {
  if (x >= 0) {x^(1/3)}
  else {-(-x)^(1/3)}
})

# X and Y are uncorrelated but dependent
x <- seq(-1,1,by = 0.01) # runif(1e5, -1,1)
y <- x^2
cor(x,y)

# The dependence is quadratic
plot(x,y)

# See that the pdf of XY ( = X^3) is symmetric, so E(XY) = 0 = E(X)E(Y)
plot(x, 1/6*1/cube_root(x)^2)
hist(x*y)
hist(x)


############ DISTANCE CORRELATION NOT MONOTONE IN NUMBER OF PLAYERS EXAMPLES
n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)
y <- x1
Rfast::dcor(y, cbind(x1,x2))$dcor
Rfast::dcor(y, cbind(x1))$dcor
shapley(y, cbind(x1,x2), utility = DC)

