# Sunnies :sunglasses:
**S**hapley values that **u**ncover **n**onli**n**ear dependenc**ies**


# Intro
Create some data (y,X) where there is some (possibly nonlinear) dependence of y on X
```r
d <- 4
n <- 1000
X <- matrix(runif(n*d,-1,1),n,d)
y <- X^2 %*% (2*(0:(d-1))) + rnorm(n)
```

Compare the linear (Pearson) pairwise correlations to the overall distance correlation of your data
```r
cor(y,X)
Rfast::dcor(y,X)$dcor
```

From the R folder, source the script `Shapley_helpers.R`. Choose the distance correlation as the utility function, and distribute the distance correlation amongst the features
```r
source("Shapley_helpers.R")
shapley( y, X, function(y,X){Rfast::dcor(y,X)$dcor} )
```
