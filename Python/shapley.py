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
import matplotlib.pyplot as plt

def calc_shapley_values(x, y, players, cf_name="dcor"):
    shapley_values = []
    cf_dict = sh.make_cf_dict(x, y, players, cf_name=cf_name)
    for _player in players:
        shapley_values.append(sh.calc_shap(x, y, _player, cf_dict))
    return shapley_values

if __name__ == "__main__":

    CF_NAME = "dcor"
    #CF_NAME = "r2"
    #CF_NAME = "aidc"

    N_SAMPLES = 100
    N_FEATS = 5

    N_ITER = 1000
    ALL_SHAPS = []
    PLAYERS = list(range(N_FEATS))

    for _i in range(N_ITER):
        #X = numpy.array([numpy.linspace(-1, 1, N_SAMPLES) for _ in range(N_FEATS)]).T
        X = numpy.array([numpy.random.uniform(-1, 1, N_SAMPLES) for _ in range(N_FEATS)]).T
        Y = numpy.matmul(numpy.multiply(X, X), 2*numpy.array(range(N_FEATS)))


        _shapley_values = calc_shapley_values(X, Y, PLAYERS, CF_NAME)
        ALL_SHAPS.append(_shapley_values)

    SHAPS_AVG = [sum(x)/len(x) for x in zip(*ALL_SHAPS)]

    # Axis formatting.
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    #plt.title("Distance correlation")
    #plt.title(r"$R^2$")
    #plt.xlabel("Player index")
    #plt.ylabel("Avg Shapley value")
    #plt.ylim([0,0.005])
    plt.bar(PLAYERS, SHAPS_AVG)
    plt.show()

