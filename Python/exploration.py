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

# For calculating the shapley value of player v
value = 0
for s in team_sizes: # These are all possible team sizes excluding v, i.e., (0,1,...,d-1)
  value_s = 0
  # <do> Create teams_of_size_s which stores all teams of size s (none of which include v)
  for team in teams_of_size_s:
    value_in_team = CF(y, X, append(v,team)) - CF(y, X, team) # <do> write CF
    value_s += value_in_team
  average_value_s = value_s/len(teams_of_size_s)
  value += average_value_s
average_value = value/len(team_sizes) # len(team_sizes) = d



# You can also precalculate this for efficiency.
# Writing the characteristic function CF:
def CF(y, X, team):
  # distance correlation between y and X
