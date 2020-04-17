import matplotlib.pyplot as plt
import numpy
import sys

D = 4
N = 100


X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T

#X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
TWO_D = 2*numpy.array(range(D))

Y = numpy.matmul(numpy.multiply(X, X), TWO_D)

det_C_xy = numpy.linalg.det(numpy.corrcoef(X.T, Y))
det_C_x = numpy.linalg.det(numpy.corrcoef(X.T))
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
