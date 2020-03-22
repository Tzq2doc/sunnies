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
    Calculate the Shapley value for player indexed v,
    given X (todo explain) and the caracteristic function CF (todo explain)
    """
    # If list of players given as input:
    #todo assert only unique values via set(players)
    #assert isinstance(players, list)

    players = list(range(X.shape[1]))

    if v in players:
        players.remove(v)

    num_players = len(players)
    team_sizes = list(range(num_players+1))
    value = 0
    v_tuple = (v,)

    # TODO -------------------------------------------------
    #for all unique team combinations

    #value_in_team = CF(y, X, _team + v_tuple) - CF(y, X, _team)
    # CF dict
    # ------------------------------------------------------

    for _size in team_sizes:
        value_s = 0
        teams_of_size_s = list(combinations(players, _size)) #NB: returns tuples
        for _team in teams_of_size_s:
            #value_in_team = CF(y, X, _team + v_tuple) - CF(y, X, _team)
            value_in_team = cf_dict[_team]
            value_s += value_in_team
        average_value_s = value_s/len(teams_of_size_s)
        value += average_value_s
    average_value = value/len(team_sizes) # len(team_sizes) = d
    return average_value

players = list(range(d))
print(calc_shap(X, v, CF))

