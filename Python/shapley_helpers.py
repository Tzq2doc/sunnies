"""
Helper functions for shalpey.py
Package dependencies:
pip install dcor (from pypi.org/project/dcor/)
"""

import sys
from itertools import combinations
import dcor
import numpy
import scipy
from HSIC import dHSIC
from xgb_regressor import make_xgb_dict

def AIDC(X, Y):
    cov_y = numpy.cov(Y)
    cov_x = numpy.cov(X.T)

    if cov_x.shape is ():
        inv_cov_x = 1.0/cov_x
        X_trans = numpy.dot(X, numpy.sqrt(inv_cov_x))
    else:
        inv_cov_x = numpy.linalg.inv(cov_x)
        X_trans = numpy.dot(X, scipy.linalg.sqrtm(inv_cov_x))

    inv_cov_y = 1/cov_y
    Y_trans = numpy.dot(Y, numpy.sqrt(inv_cov_y))
    return dcor.distance_correlation(Y_trans, X_trans)

def CF(x, y, team, cf_name):
    """
    Available characteristic functions:
        dcor: Distance correlation between y and x
    """
    x = x[:, team]

    if len(team)==0:
        return 0.0

    if cf_name is "dcor":
        return dcor.distance_correlation(y, x)

    elif cf_name is "r2":
        det_C_xy = numpy.linalg.det(numpy.corrcoef(x.T, y))
        if len(team)==1:
            det_C_x = 1
        else:
            det_C_x = numpy.linalg.det(numpy.corrcoef(x.T))
        return (1 - det_C_xy/det_C_x)

    elif cf_name is "aidc":
        return dcor.distance_correlation_af_inv(y, x)
        #return AIDC(x, y)

    elif cf_name is "hsic":
        return dHSIC(x, y)

    else:
        raise NameError("I don't know the characteristic function {0}".format(cf_name))
        return 0

def make_cf_dict(x, y, players, cf_name):
    """
    Creates dictionary with values of the characteristic function for each
    combination of the players.
    """
    cf_dict = {}
    num_players = len(players)
    team_sizes = list(range(num_players+1))

    if cf_name is "xgb":
        return make_xgb_dict(x, y)

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

            #this sometimes gets negative when using cf=r^2
            #print(value_in_team)
            value_s += value_in_team
        average_value_s = value_s/len(teams_of_size_s)
        value += average_value_s
    average_value = value/len(team_sizes)
    return average_value

