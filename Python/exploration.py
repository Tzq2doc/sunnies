"""
Shapley values with various model-agnostic measures of dependence as
utility functions.
Package dependencies:
pip install dcor (from pypi.org/project/dcor/)


"""
import numpy
from itertools import combinations
import dcor
import sys

#def make_y(x1, x2, x3, x4):
#    a1, a2, a3, a4 = 0, 1, 3, 4
#    return a1*x1*x1 + a2*2*x2 + a3*x3*x3 + a4*x4*x4

n=10
d=4
X = numpy.array([[numpy.random.uniform(-1, 1) for x in range(d+1)]
                  for y in range(n)])
#y = X * (2*(0:(d-1))) + rnorm(n)
e = e = numpy.random.uniform(-1, 1, n).reshape(n, 1) # Check this!
y = X * 2*numpy.array(range(d+1)) + e # Square X at some point

# You can also precalculate this for efficiency.
def CF(y, X, team):
    """Distance correlation between y and X """
    X = X[:, team]
    y = y[:, team]
    return dcor.distance_correlation(y,X)

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
        teams_of_size_s = list(combinations(players, _size)) #NB: returns tuples
        for _team in teams_of_size_s:
            _team = list(_team) # cast tuple to list. probably a slow fix :(
            value_in_team = CF(y, X, _team.append(v)) - CF(y, X, _team)
            value_s += value_in_team
        average_value_s = value_s/len(teams_of_size_s)
        value += average_value_s
    average_value = value/len(team_sizes) # len(team_sizes) = d
    return average_value

players = [1,2,3,4]
v = 3
print(calc_shap(players, v))

