
import numpy
import sys

sys.path.insert(0, "tests")
from tests import test_hsic


if __name__ == "__main__":
    try:
        import calc_hsic as target
    except ModuleNotFoundError:
        sys.path.insert(0, "hsic")
        import calc_hsic as target

    test_hsic.test_centering(target.centering)

    print("Everything passed")

    ## --- Data
    #D = 3
    #N = 10
    #X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
    #TWO_D = 2*numpy.array(range(D))
    #Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
    ## ---

    ## --- Test dHSIC calculations
    #K_list = [gaussian_grammat(_x) for _x in [X, Y]]
    #print(dHSIC_calc(K_list))
    ##print(dHSIC_calc0(K_list))
    #print(dHSIC(X, Y))

    ##print(dHSIC(X, Y, X, Y))
    ##print(dHSIC(X))
