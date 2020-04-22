"""
Hilbert Schmidt Information Criterion with a Gaussian kernel, based on the
following references
[1]: http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf
[2]: https://arxiv.org/pdf/1603.00285.pdf

"""
import numpy

def centering(M):
    """
    Calculate the centering matrix
    """
    n = M.shape[0]
    unit = numpy.ones([n, n])
    identity = numpy.eye(n)
    H = identity - unit/n

    return numpy.matmul(M, H)

def gaussian_grammat(x, sigma=None):
    """
    Calculate the Gram matrix of x using a Gaussian kernel.
    If the bandwidth sigma is None, it is estimated using the median heuristic:
    ||x_i - x_j||**2 = 2 sigma**2
    """
    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(x.shape[0], 1)

    xxT = numpy.matmul(x, x.T)
    xnorm = numpy.diag(xxT) - xxT + (numpy.diag(xxT) - xxT).T
    if sigma is None:
        mdist = numpy.median(xnorm[xnorm!= 0])
        sigma = numpy.sqrt(mdist*0.5)
    KX = - 0.5 * xnorm / sigma / sigma
    numpy.exp(KX, KX)
    return KX

def dHSIC_calc(K_list):
    """
    Calculate the HSIC estimator in the general case d > 2, as in
    [2] Definition 2.6
    """
    if not isinstance(K_list, list):
        K_list = list(K_list)

    n_k = len(K_list)

    length = K_list[0].shape[0]
    term1 = 1.0
    term2 = 1.0
    term3 = 2.0/length

    for j in range(0, n_k):
        K_j = K_list[j]
        term1 = numpy.multiply(term1, K_j)
        term2 = 1.0/length/length*term2*numpy.sum(K_j)
        term3 = 1.0/length*term3*K_j.sum(axis=0)

    term1 = numpy.sum(term1)
    term3 = numpy.sum(term3)
    dHSIC = (1.0/length)**2*term1+term2-term3
    return dHSIC

def HSIC(x, y):
    """
    Calculate the HSIC estimator for d=2, as in [1] eq (9)
    """
    n = x.shape[0]
    return numpy.trace(numpy.matmul(centering(gaussian_grammat(x)),centering(gaussian_grammat(y))))/n/n

def dHSIC(*argv):
    assert len(argv) > 1, "dHSIC requires at least two arguments"

    if len(argv) == 2:
        x, y = argv
        return HSIC(x, y)

    K_list = [gaussian_grammat(_arg) for _arg in argv]
    return dHSIC_calc(K_list)
if __name__ == "__main__":
    # --- Data
    D = 5
    N = 100

    X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
    #X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
    TWO_D = 2*numpy.array(range(D))
    Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
    # ---

    # --- Test dHSIC calculation
    print(HSIC(X, Y))


