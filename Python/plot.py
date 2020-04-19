import matplotlib.pyplot as plt
import scipy.linalg
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





# -- Plot
plt.scatter(X[:,0], Y, label="X0", alpha=0.8)
plt.scatter(X[:,1], Y, label="X1", alpha=0.8)
plt.scatter(X[:,2], Y, label="X2", alpha=0.8)
plt.scatter(X[:,3], Y, label="X3", alpha=0.8)
plt.legend(loc="upper right")
plt.show()

# ---
