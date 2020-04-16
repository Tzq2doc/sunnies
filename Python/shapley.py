"""
Shapley values with various model-agnostic measures of dependence as
utility functions.
Uses shapley_helpers.py
"""

import shapley_helpers as sh
import numpy
from itertools import combinations
import dcor
import sys

if __name__ == "__main__":
    N = 100
    D = 5

    #X = numpy.array([[numpy.random.uniform(-1, 1) for _ in range(D)]
    #                  for y in range(N)])
    #X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
    X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
    Y = numpy.matmul(numpy.multiply(X, X), 2*numpy.array(range(D)))

    PLAYERS = list(range(D))
    CF_DICT = sh.make_cf_dict(X, Y, PLAYERS, cf_name="dcor")
    #CF_DICT = sh.make_cf_dict(X, Y, PLAYERS, cf_name="hilbert_schmidt")
    for v in range(D):
        print(sh.calc_shap(X, Y, v, CF_DICT))

