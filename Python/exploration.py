"""
Shapley values with various model-agnostic measures of dependence as
utility functions.
Package dependencies:
pip install dcor (from pypi.org/project/dcor/)


"""
import numpy
from itertools import combinations

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

def calc_shap(players, v):
    """
    Calculate the Shapley value for v given a list of players in the game
    """
    assert isinstance(players, list)
    if v in players:
        players.remove(v)

    num_players = len(players)
    team_sizes = list(range(1, num_players+1)) # Python stops one before last
    value = 0

    for _size in team_sizes:
        value_s = 0
        teams_of_size_s = combinations(players, _size)
        print(list(teams_of_size_s))
        for _team in teams_of_size_s:
            value_in_team = CF(y, X, append(v, _team)) - CF(y, X, _team) # <do> write CF
            value_s += value_in_team
        average_value_s = value_s/len(teams_of_size_s)
        value += average_value_s
    average_value = value/len(team_sizes) # len(team_sizes) = d

players = [1,2,3,4]
v =3
print(calc_shap(players, v))


# You can also precalculate this for efficiency.
# Writing the characteristic function CF:
def CF(y, X, team):
  # distance correlation between y and X
  #oh no, what are y and X in this context :S
  return
