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
    try:
        x.shape[1]
    except IndexError:
        n = x.shape[0]
        x = x.reshape(n, 1)

    GX = numpy.dot(x, x.T)
    KX = numpy.diag(GX) - GX + (numpy.diag(GX) - GX).T
    if sigma is None:
        mdist = numpy.median(KX[KX != 0])
        #sigma = math.sqrt(mdist)
        sigma = math.sqrt(mdist*0.5)
    KX *= - 0.5 / sigma / sigma
    numpy.exp(KX, KX)
    return KX

def japanese_bw(x):
    try:
        x.shape[1]
    except IndexError:
        n = x.shape[0]
        x = x.reshape(n, 1)
    GX = numpy.dot(x, x.T)
    KX = numpy.diag(GX) - GX + (numpy.diag(GX) - GX).T
    mdist = numpy.median(KX[KX != 0])
    sigma = math.sqrt(mdist*0.5)
    return sigma

def HSIC(x, Y):
    #return numpy.sum(centering(rbf(x)) * centering(rbf(Y)))
    return numpy.trace(numpy.matmul(centering(rbf(x)),centering(rbf(Y))))/(N-1)/(N-1)


def cpp_bw(x):
    """
    starts at line 60 in
    https://github.com/cran/dHSIC/blob/master/src/rcpp_functions.cpp
    """
    length = x.shape[0]

    try:
        d = x.shape[1]
    except IndexError:
        d = 1
        x = numpy.matrix(x).T

    if(length > 1000):
        length = 1000

    lentot = length*(length+1)/2-length
    bandvec = numpy.zeros((int(lentot)))
    xnorm = 0.0
    count = 0
    for i in range(0, length):
        j = i+1;
        while(j < length):
            for l in range(0, d):
                xnorm += (x[i][l]-x[j][l])**2
            bandvec[count] = xnorm
            xnorm = 0.0
            j += 1
            count += 1

    bandwidth = numpy.median(bandvec)
    #v = numpy.copy(bandvec)
    #v.sort()
    #middle = int(lentot/2)
    #bandwidth = v[middle]
    #std::nth_element(v.begin(), v.begin() + middle, v.end());
    bandwidth = numpy.sqrt(bandwidth*0.5)
    return bandwidth


def cpp_K(x, bw=None):
    """
    starts at line 5 in
    https://github.com/cran/dHSIC/blob/master/src/rcpp_functions.cpp

    """
    n = x.shape[0]

    try:
        d = x.shape[1]
    except IndexError:
        d = 1
        x = numpy.matrix(x).T

    xnorm = 0
    K = numpy.zeros((n,n)) #n x n
    if bw is None:
        bw = cpp_bw(x)

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

def r_dhsic(K):
    """
    starts at line 112 in
    https://github.com/cran/dHSIC/blob/master/R/dhsic.R
    # Compute dHSIC
    """
    length, d = K.shape #TODO: check that K is always NxN
    term1 = 1
    term2 = 1
    term3 = 2/length
    for j in range(0, d):
        Kj = cpp_K(X[:,j])
        term1 = term1*Kj
        term2 = 1/length/length*numpy.sum(Kj)
        #term3 <- 1/len*term3*colSums(K[[j]])
        term1 = sum(term1)
        #term3 = sum(term3)
        dHSIC = 1/length**2*term1+term2#-term3
    return dHSIC


if __name__ == "__main__":
    # --- Data
    D = 2#4
    N = 5#100

    #X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
    X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
    TWO_D = 2*numpy.array(range(D))
    Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
    # ---

    # --- Test bandwidth calculations:
    print(cpp_bw(X))
    print(japanese_bw(X))
    #sys.exit()

    # --- Test rbf = cpp_K
    #print(cpp_K(Y, bw=1).round(decimals=2) == rbf(Y, sigma=1).round(decimals=2))
    print(cpp_K(X).round(decimals=2) == rbf(X).round(decimals=2))
    print(cpp_K(Y).round(decimals=2) == rbf(Y).round(decimals=2))
    #print(cpp_K(X, bw=1).round(decimals=2) == rbf(X, sigma=1).round(decimals=2))
    sys.exit()

    #Y = numpy.reshape(Y, (N,1))

    #print(HSIC(X, Y))

