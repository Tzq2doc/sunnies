"""
Helper functions for shalpey.py
Package dependencies:
pip install dcor (from pypi.org/project/dcor/)
"""

from itertools import combinations
import dcor

def CF(x, y, team, cf_name="dcor"):
    """
    Available characteristic functions:
        dcor: Distance correlation between y and x
    """
    x = x[:, team]

    if cf_name is "dcor":
        return dcor.distance_correlation(y, x)
    else:
        raise NameError("I don't know the characteristic function {0}".format(cf_name))
        return 0

def make_cf_dict(x, y, players, cf_name="dcor"):
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
            cf_dict[_team] = CF(x, y, _team, cf_name)

    return cf_dict

def calc_shap(x, y, v, cf_dict):
    """
    Calculate the Shapley value for player indexed v,
    given x (todo explain) and the caracteristic function cf (todo explain)
    """
    players = list(range(x.shape[1]))

    if v in players:
        players.remove(v)

    num_players = len(players)
    team_sizes = list(range(num_players+1))
    value = 0
    v_tuple = (v,)

    for _size in team_sizes:
        value_s = 0
        teams_of_size_s = list(combinations(players, _size))
        for _team in teams_of_size_s:
            #value_in_team = cf(x, y, _team + v_tuple) - cf(x, y, _team)
            value_in_team = (cf_dict[tuple(sorted(_team+v_tuple))] - cf_dict[_team])
            value_s += value_in_team
        average_value_s = value_s/len(teams_of_size_s)
        value += average_value_s
    average_value = value/len(team_sizes)
    return average_value

