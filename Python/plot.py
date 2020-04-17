import matplotlib.pyplot as plt
import scipy.linalg
import numpy
import sys

D = 4
N = 100

X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T

#X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
TWO_D = 2*numpy.array(range(D))
Y = numpy.matmul(numpy.multiply(X, X), TWO_D)

cov_y = numpy.cov(Y)
cov_x = numpy.cov(X.T)
inv_cov_x = numpy.linalg.inv(cov_x)
inv_cov_y = 1/cov_y
X_trans = numpy.dot(X, scipy.linalg.sqrtm(inv_cov_x))
Y_trans = numpy.dot(Y, numpy.sqrt(inv_cov_y))
# Plug X_trans and Y_trans into the usual distance correlation CF

det_C_xy = numpy.linalg.det(numpy.corrcoef(X.T, Y))
det_C_x = numpy.linalg.det(numpy.corrcoef(Y))
R2 = 1 - det_C_xy/det_C_x
print(R2)


sys.exit()

#print(X)
#print(numpy.multiply(X,X))
#print(TWO_D)
#print(Y)

#print(numpy.array(range(D)).T)
#print(X[:,1])

plt.scatter(X[:,0], Y, label="X0", alpha=0.8)
plt.scatter(X[:,1], Y, label="X1", alpha=0.8)
plt.scatter(X[:,2], Y, label="X2", alpha=0.8)
plt.scatter(X[:,3], Y, label="X3", alpha=0.8)
plt.legend(loc="upper right")
plt.show()


#import dcor
#print(dcor.distance_correlation(Y, X))
#print(dcor.distance_correlation(Y, X[:,0]))
