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
import os
import matplotlib.pyplot as plt

# --- My stuff
from plot import nice_axes, violinplot, boxplot, barplot_all
import data

CF_DICT = {
        "r2" : r"$R^2$",
        "dcor" : "Distance correlation",
        "aidc" : "Affine invariant dist. corr",
        "hsic" : "Hilbert-Schmidt indep.cr.",
        "xgb" : "XGBoost Regressor",
        }

def calc_shapley_values(x, y, players, cf_name="dcor"):
    shapley_values = []
    cf_dict = sh.make_cf_dict(x, y, players, cf_name=cf_name)
    for _player in players:
        shapley_values.append(sh.calc_shap(x, y, _player, cf_dict))
    return shapley_values

def calc_n_shapley_values(n_iter, cf_name, overwrite=False):
    """
    Returns a nested list of shapley values (per player) per iteration;
    [[v1... vn], [v1...vn], [v1...vn], ...]
    I.e. the length of the list is equal to n_iter
    """
    global N_FEATS, N_SAMPLES, PLAYERS

    filename = "{0}/{1}_feats_{2}_samples_{3}_iter_{4}.npy".format(DATA_DIR,
            N_FEATS, N_SAMPLES, N_ITER, cf_name)
    if not overwrite and os.path.exists(filename):
        return numpy.load(filename)

    all_shaps = []
    for _i in range(n_iter):
        x, y = data.make_data_step(N_FEATS, N_SAMPLES)
        #x, y = data.make_data_random(N_FEATS, N_SAMPLES)

        _shapley_values = calc_shapley_values(x, y, PLAYERS, cf_name)
        all_shaps.append(_shapley_values)

    numpy.save(filename, all_shaps)

    return all_shaps

def normalise(x):
    return (x - numpy.mean(x))/(numpy.std(x))


if __name__ == "__main__":

    #CF_NAME = "hsic"
    #CF_NAME = "dcor"
    #CF_NAME = "r2"
    CF_NAME = "aidc"
    #CF_NAME = "xgb"

    N_SAMPLES = 1000
    N_FEATS = 5
    N_ITER = 1000
    PLAYERS = list(range(N_FEATS))


    # --- Pick one data generating process
    #DATA_TYPE = "step" #ok
    #DATA_TYPE = "random" #ok
    #DATA_TYPE = "harmonic" #ok
    DATA_TYPE = "xor" #fix
    # ---

    DATA_DIR = os.path.join("result_data", "{0}".format(DATA_TYPE))
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    # --- Plot shapley decomposition per player for one cf:
    #all_shaps = calc_n_shapley_values(N_ITER, CF_NAME)
    # --- Non-normalised:
    #shaps_per_player = [numpy.array(all_shaps)[:,_player] for _player in PLAYERS]
    #violinplot(shaps_per_player, PLAYERS, multi=False)
    #boxplot(shaps_per_player, PLAYERS, multi=False)
    # --- Normalised:
    #shaps_per_player = [normalise(numpy.array(all_shaps))[:,_player] for _player in PLAYERS]
    #boxplot(shaps_per_player, PLAYERS, multi=False)
    #violinplot(shaps_per_player, PLAYERS, multi=False)
    #plt.title(CF_DICT.get(CF_NAME, CF_NAME))
    #plt.show()
    #sys.exit()
    # ---

    # --- Plot shapley decompositions for all cfs per player
    cfs = ["r2", "hsic", "dcor", "aidc"]
    all_cf_shaps_per_player = []
    for _cf in cfs:
        _all_player_shaps = []
        _cf_shaps = calc_n_shapley_values(N_ITER, _cf)

        # --- Group shapley decompositions per player
        # --- Normalised:
        all_cf_shaps_per_player.append([normalise(numpy.array(_cf_shaps))[:,_player] for _player in PLAYERS])
        # --- Non-normalized:
        #all_cf_shaps_per_player.append([numpy.array(_cf_shaps)[:,_player] for _player in PLAYERS])
        # ---
        print("Done with {0}.".format(_cf))
    # ---

    cf_labels = [CF_DICT.get(_cf, 0) for _cf in cfs]
    violinplot(all_cf_shaps_per_player, PLAYERS, labels=cf_labels, multi=True)
    boxplot(all_cf_shaps_per_player, PLAYERS, labels=cf_labels, multi=True)

    # --- Plot average Shapley decomposition per player
    #SHAPS_AVG = [sum(x)/len(x) for x in zip(*all_shaps)]
    #barplot_all(PLAYERS, SHAPS_AVG)
    # ---

    try:
        plt.show()
    except Exception:
        pass


# NOTES
    #plt.title("Distance correlation")
    #plt.title(r"$R^2$")
    #plt.xlabel("Player index")
    #plt.ylabel("Avg Shapley value")
    #plt.ylim([0,0.005])
