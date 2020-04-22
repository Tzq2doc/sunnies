
import numpy

def test_centering(func):
    m2 = numpy.ones(2) + numpy.eye(2)
    ans_2d = (numpy.eye(2)-1/2.0*numpy.ones(2))
    assert (ans_2d == func(m2)).all()

    m4 = numpy.ones(4) + numpy.eye(4)
    ans_4d = (numpy.eye(4)-1/4.0*numpy.ones(4))
    assert (ans_4d == func(m4)).all()

    return True

def test_gaussian_grammat(func):
    m = numpy.ones(2) + numpy.eye(2)
    ans = numpy.array([[1., 0.37], [0.37, 1.]])
    assert (func(m, sigma=1).round(decimals=2) == ans).all()

    ans = numpy.array([[1., 0.14], [0.14, 1.]])
    assert (func(m).round(decimals=2) == ans).all()

    return True

if __name__ == "__main__":
    try:
        import calc_hsic as target
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, "../hsic")
        import calc_hsic as target

    test_centering(target.centering)
    test_gaussian_grammat(target.gaussian_grammat)


    #D = 3
    #N = 10
    #X = numpy.ones(2) + numpy.eye(2)
    #X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
    #TWO_D = 2*numpy.array(range(D))
    #Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
    ## ---
    print("Everything passed")
