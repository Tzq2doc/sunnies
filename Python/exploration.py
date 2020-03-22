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

N = 10
D = 5
#X = numpy.array([[numpy.random.uniform(-1, 1) for _ in range(D)]
#                  for y in range(N)])
X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
#ERROR = numpy.random.normal(0, 1, N)
Y = numpy.matmul(numpy.multiply(X, X), 2*numpy.array(range(D)))# + e
V = 2

def CF(x, y, team):
    """Distance correlation between y and X """
    x = x[:, team]
    return dcor.distance_correlation(y, x)

def make_cf_dict(x, y, players):
    """
    Creates dictionary with values of the characteristic function for each
    combination of the players.
    """
    cf_dict = {}
    num_players = len(players)
    team_sizes = list(range(num_players+1))

    for _size in team_sizes:
        value_s = 0
        teams_of_size_s = list(combinations(players, _size)) #NB: returns tuples
        for _team in teams_of_size_s:
            cf_dict[_team] = CF(x, y, _team)

    return cf_dict

def calc_shap(x, y, v, cf_dict, cf):
    """
    Calculate the Shapley value for player indexed v,
    given X (todo explain) and the caracteristic function cf (todo explain)
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

    for _size in team_sizes:
        value_s = 0
        teams_of_size_s = list(combinations(players, _size)) #NB: returns tuples
        for _team in teams_of_size_s:
            value_in_team = cf(x, y, _team + v_tuple) - cf(x, y, _team)
            #value_in_team = (cf_dict[tuple(sorted(_team+v_tuple))] - cf_dict[_team])
            value_s += value_in_team
        average_value_s = value_s/len(teams_of_size_s)
        value += average_value_s
    average_value = value/len(team_sizes) # len(team_sizes) = d
    return average_value

PLAYERS = list(range(D))
CF_DICT = make_cf_dict(X, Y, PLAYERS)
print(calc_shap(X, Y, V, CF_DICT, CF))

