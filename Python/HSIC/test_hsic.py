from HSIC import hsic_gam


import numpy
import sys

D = 4
N = 100

# --- Data
X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
#X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
TWO_D = 2*numpy.array(range(D))
Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
# ---


Y = Y.reshape((N,1))
print(Y.shape)
print(X.shape)
testStat, thresh = hsic_gam(X, Y, alph = 0.05)
print(testStat)
print(thresh)
