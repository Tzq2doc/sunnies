
import dcor
import matplotlib.pyplot as plt
import scipy.linalg
import numpy
import sys
from shapley_helpers import AIDC

D = 4
N = 100

# --- Data
#X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
TWO_D = 2*numpy.array(range(D))
Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
# ---

# --- Transform data
M = numpy.array([numpy.random.uniform(-10, 10, D) for _ in range(D)])
N = numpy.array([numpy.random.uniform(-10, 10, N) for _ in range(D)]).T
X_TRANS1 = numpy.matmul(X, M)
X_TRANS2 = numpy.matmul(X, M) + N

print("Distance correlation:")
print(dcor.distance_correlation(Y, X))
print("Unbiased dcor:")
print(numpy.sqrt(dcor.u_distance_correlation_sqr(Y, X)))
sys.exit()

#for _ in range(10000):
#    AIDC(X, Y)
#    dcor.distance_correlation_af_inv(Y, X)
#print("done")
#sys.exit()

print("AIDC original X:")
print(AIDC(X, Y))
print("AIDC built-in X:")
print(dcor.distance_correlation_af_inv(Y, X))
print("AIDC X = M*X:")
print(AIDC(X_TRANS1, Y))
print(dcor.distance_correlation_af_inv(Y, X_TRANS1))
print("AIDC X = M*X + N:")
print(AIDC(X_TRANS2, Y))
print(dcor.distance_correlation_af_inv(Y, X_TRANS2))

# AIDC
#cov_y = numpy.cov(Y)
#cov_x = numpy.cov(X.T)
#inv_cov_x = numpy.linalg.inv(cov_x)
#inv_cov_y = 1/cov_y
#X_trans = numpy.dot(X, scipy.linalg.sqrtm(inv_cov_x))
#Y_trans = numpy.dot(Y, numpy.sqrt(inv_cov_y))

# R2
#det_C_xy = numpy.linalg.det(numpy.corrcoef(X.T, Y))
#det_C_x = numpy.linalg.det(numpy.corrcoef(Y))
#R2 = 1 - det_C_xy/det_C_x


#print(dcor.distance_correlation(Y, X))
#print(dcor.distance_correlation(Y, X_TRANS))
