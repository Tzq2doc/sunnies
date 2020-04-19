import matplotlib.pyplot as plt
import scipy.linalg
import numpy
import math
import sys
from HSIC import HSIC

def centering(K):
    """
    Calculate the centering matrix (is a symmetric and idempotent matrix, which
    when multiplied with a vector has the same effect as subtracting the mean
    of the components of the vector from every component).
    """
    n = K.shape[0]
    unit = numpy.ones([n, n])
    I = numpy.eye(n)
    Q = I - unit/n

    #return numpy.dot(numpy.dot(Q, K), Q)
    return numpy.matmul(K, Q)

def rbf(x, sigma=None):
    N = x.shape[0]
    GX = numpy.dot(x, x.T)
    KX = numpy.diag(GX) - GX + (numpy.diag(GX) - GX).T
    #KX = numpy.diag(GX)*numpy.eye(N) - GX + (numpy.diag(GX)*numpy.eye(N) - GX).T
    if sigma is None:
        mdist = numpy.median(KX[KX != 0])
        #sigma = math.sqrt(mdist)
        sigma = math.sqrt(mdist*0.5)
    KX *= - 0.5 / sigma / sigma
    numpy.exp(KX, KX)
    return KX


def japanese(x):
    sigma=1
    n = x.shape[0]
    GX = numpy.dot(x, x.T)
    KX = numpy.diag(GX) - GX + (numpy.diag(GX) - GX).T
    # TODO: check mdist*0.5 thing
    #if sigma is None:
    #    mdist = numpy.median(KX[KX != 0])
    #    #sigma = math.sqrt(mdist)
    #    sigma = math.sqrt(mdist*0.5)
    KX *= - 0.5 / sigma / sigma
    numpy.exp(KX, KX)
    return KX

def cpp_xnorm(x):
    n = x.shape[0]
    d = x.shape[1]
    xnorm = 0
    K = numpy.zeros((n,n)) #n x n
    bw = 1
    # TODO: Bandwidth calculation

    for i in range(n+1):
        j=i
        while j<n:
            for l in range(0,d):
                xnorm += (x[i][l] - x[j][l])**2
            K[i][j] = numpy.exp(-xnorm/(2.0*bw**2))
            #K[i][j] = xnorm
            K[j][i] = K[i][j]
            xnorm = 0.0
            j += 1
    return K

def HSIC(x, Y):
    #return numpy.sum(centering(rbf(x)) * centering(rbf(Y)))
    return numpy.trace(numpy.matmul(centering(rbf(x)),centering(rbf(Y))))/(N-1)/(N-1)

if __name__ == "__main__":
    # --- Data
    D = 2#4
    N = 5#100

    X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
    #X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
    TWO_D = 2*numpy.array(range(D))
    Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
    # ---
    Y = numpy.reshape(Y, (N,1))

    print(cpp_xnorm(X) == japanese(X))
    sys.exit()
    #print(X.shape)
    #print(Y)
    #print(TWO_D)

    print(HSIC(X, Y))

