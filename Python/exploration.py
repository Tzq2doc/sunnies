"""
Shapley values with various model-agnostic measures of dependence as
utility functions.
Package dependencies:
pip install dcor (from pypi.org/project/dcor/)


"""
import numpy

def make_y(x1, x2, x3, x4):
    a1, a2, a3, a4 = 0, 1, 3, 4
    return a1*x1*x1 + a2*2*x2 + a3*x3*x3 + a4*x4*x4

x = numpy.random.uniform(-1, 1, 100)
y = x*x
#cov(x,y) = 0

#n=1000, d=4
# Matrix with (n rows, d columns) uniformly distributed (-1, 1)  reshape into nxd matrix
#square element-wise
#

#import simple_shapley from simple_shapley_helpers
