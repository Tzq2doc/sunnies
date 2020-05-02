import sys
import numpy
import scipy

def make_data_random(d, n):
    #x = numpy.array([numpy.linspace(-1, 1, n) for _ in range(d)]).T
    x = numpy.array([numpy.random.uniform(-1, 1, n) for _ in range(d)]).T
    two_d = 2*numpy.array(range(d))
    epsilon = numpy.random.uniform(-0.2, 0.2, n)
    y = numpy.matmul(numpy.multiply(x, x), two_d) + epsilon

    return x, y

def make_data_step(d, n):
    two_d = 2*numpy.array(range(d))

    x = numpy.array([numpy.random.uniform(-1, 1, n) for _ in range(d)]).T
    y = numpy.matmul([[(-0.5 < _xi and _xi < 0.5) for _xi in _x] for _x in x], two_d)

    return x, y

def make_data_xor(d, n):
    if d !=2:
        print("Please use only d=2 for XOR function")
        sys.exit()
    # no d currently implemented. d=2.
    x1 = numpy.random.uniform(-1, 1, n)
    x2 = numpy.random.uniform(-1, 1, n)
    y = [_x1*(_x1 > 0 and _x2 < 0) + _x2*(_x1 < 0 and _x2 > 0)
       + _x1*(_x1 < 0 and _x2 < 0) - _x2*(_x1 > 0 and _x2 > 0)
            for _x1, _x2 in zip(x1, x2)]

    x = numpy.vstack((x1, x2)).T

    return x, y

def make_data_harmonic(d, n):
    two_d = 2*numpy.array(range(d))

    x = numpy.array([numpy.random.uniform(-numpy.pi, numpy.pi, n) for _ in range(d)]).T
    y = numpy.matmul(scipy.cos(x), two_d)

    return x, y