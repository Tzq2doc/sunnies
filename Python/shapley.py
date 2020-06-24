"""
Shapley values with various model-agnostic measures of dependence as
utility functions.
Uses shapley_helpers.py
"""

import numpy
from itertools import combinations
import dcor
import sys
import os
import matplotlib.pyplot as plt

# --- My stuff
from plot import nice_axes, violinplot, boxplot, barplot_all
import data
import shapley_helpers as sh

CF_DICT = {
        "r2" : r"$R^2$",
        "dcor" : "Distance correlation",
        "aidc" : "Affine invariant dist. corr",
        "hsic" : "Hilbert-Schmidt indep.cr.",
        "xgb" : "XGBoost Regressor",
        }

def calc_shapley_values(x, y, cf_name="dcor"):
    """
    Returns the shapley values for features x and labels y, given a
    characteristic function (default dcor)
    """
    players = list(range(x.shape[1]))
    shapley_values = []
    cf_dict = sh.make_cf_dict(x, y, players, cf_name=cf_name)
    for _player in players:
        shapley_values.append(sh.calc_shap(x, y, _player, cf_dict))
    return shapley_values

def calc_n_shapley_values(n_feats, n_samples, n_iter, data_type, cf_name, overwrite=False, data_dir="result_data_sunnies"):
    """
    Returns a nested list of shapley values (per player) per iteration;
    [[v1... vn], [v1...vn], [v1...vn], ...]
    I.e. the length of the list is equal to n_iter
    """
    players = list(range(n_feats))

    filename = f"{data_dir}/{n_feats}_feats_{n_samples}_samples_{n_iter}_iter_{cf_name}.npy"
    if not overwrite and os.path.exists(filename):
        return numpy.load(filename)

    all_shaps = []
    for _i in range(n_iter):
        x, y = data.make_data(n_feats, n_samples, data_type)

        _shapley_values = calc_shapley_values(x, y, cf_name)
        all_shaps.append(_shapley_values)

    numpy.save(filename, all_shaps)

    return all_shaps

def normalise(x):
    return (x - numpy.mean(x))/(numpy.std(x))


def make_paper_violin_plot():
    """
    Create violin plot as it appears in the paper
    """

    n_samples = 1000
    n_feats = 5
    n_iter = 1000
    players = list(range(n_feats))
    data_type = "random"
    data_dir = os.path.join("result_data", "{0}".format(data_type))
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # --- Plot shapley decompositions for all cfs per player
    cfs = ["r2", "hsic", "dcor", "aidc"]
    all_cf_shaps_per_player = []
    for _cf in cfs:
        _all_player_shaps = []
        _cf_shaps = calc_n_shapley_values(n_feats, n_samples, n_iter,
                data_type, _cf, overwrite=False, data_dir=data_dir)

        # --- Group shapley decompositions per player. Normalised.
        all_cf_shaps_per_player.append([normalise(numpy.array(_cf_shaps))[:,_player] for _player in players])
        print("Done with {0}.".format(_cf))
    # ---

    cf_labels = [CF_DICT.get(_cf, 0) for _cf in cfs]
    violinplot(all_cf_shaps_per_player, players, labels=cf_labels, multi=True)


if __name__ == "__main__":

    make_paper_violin_plot()

    #CF_NAME = "hsic"
    #CF_NAME = "dcor"
    #CF_NAME = "r2"
    #CF_NAME = "aidc"
    #CF_NAME = "xgb"

    N_SAMPLES = 100#1000
    N_FEATS = 2#5
    N_ITER = 100#1000
    PLAYERS = list(range(N_FEATS))


    # --- Pick one data generating process
    #DATA_TYPE = "step" #ok
    DATA_TYPE = "random" #ok
    #DATA_TYPE = "harmonic" #ok
    #DATA_TYPE = "xor_discrete" #ok
    #DATA_TYPE = "xor_discrete_discrete" #ok
    # ---

    DATA_DIR = os.path.join("result_data", "{0}".format(DATA_TYPE))
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    # --- Plot shapley decomposition per player for one cf:
    #all_shaps = calc_n_shapley_values(N_FEATS, N_SAMPLES, N_ITER, DATA_TYPE, CF_NAME, data_dir=DATA_DIR)
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
    cfs = ["hsic", "dcor", "aidc"]
    all_cf_shaps_per_player = []
    for _cf in cfs:
        _all_player_shaps = []
        _cf_shaps = calc_n_shapley_values(N_FEATS, N_SAMPLES, N_ITER,
                DATA_TYPE, _cf, data_dir=DATA_DIR)

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
