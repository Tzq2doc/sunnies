import matplotlib.pyplot as plt
import scipy.linalg
import numpy
import math
import sys


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


def japanese_bw(x):
    try:
        x.shape[1]
    except IndexError:
        x = numpy.reshape
    GX = numpy.dot(x, x.T)
    KX = numpy.diag(GX) - GX + (numpy.diag(GX) - GX).T
    mdist = numpy.median(KX[KX != 0])
    sigma = math.sqrt(mdist*0.5)
    return sigma




japanese_bw(X)
japanese_bw(Y)
Y = numpy.matrix(Y).T
n = Y.shape[0]
GX = numpy.dot(Y, Y.T)
KX = numpy.diag(GX) - GX + (numpy.diag(GX) - GX).T
numpy.median(KX[KX != 0], axis = 1)
sigma = math.sqrt(mdist * 0.5)
return sigma

D = 2#4
N = 5#100

X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
#X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
TWO_D = 2*numpy.array(range(D))
Y = numpy.matmul(numpy.multiply(X, X), TWO_D)

cpp_K(X)