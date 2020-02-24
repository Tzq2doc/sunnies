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

n = 10
d = 4
#X = numpy.array([[numpy.random.uniform(-1, 1) for x in range(d)]
#                  for y in range(n)])
X = numpy.array([numpy.linspace(-1, 1, n) for x in range(d)]).T
e = numpy.random.normal(0, 1, n)
y = numpy.matmul(numpy.multiply(X, X), 2*numpy.array(range(d)))# + e
v = 3

# You can also precalculate this for efficiency.
def CF(y, X, team):
    """Distance correlation between y and X """
    X = X[:, team]
    return dcor.distance_correlation(y,X)

def calc_shap(X, v, CF):
    """
    Calculate the Shapley value for v given a list of players in the game
    """
    # todo: assert only unique values via set(players)
    players = list(range(X.shape[1]))
    assert isinstance(players, list)
    if v in players:
        players.remove(v)

    num_players = len(players)
    team_sizes = list(range(num_players+1))
    value = 0
    vt = (v,)

    for _size in team_sizes:
        value_s = 0
        teams_of_size_s = list(combinations(players, _size)) #NB: returns tuples
        for _team in teams_of_size_s:
            value_in_team = CF(y, X, _team + vt) - CF(y, X, _team)
            value_s += value_in_team
        average_value_s = value_s/len(teams_of_size_s)
        value += average_value_s
    average_value = value/len(team_sizes) # len(team_sizes) = d
    return average_value

players = list(range(d))
print(calc_shap(X, v, CF))

