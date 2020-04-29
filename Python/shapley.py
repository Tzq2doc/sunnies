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

CF_DICT = {
        "r2" : r"$R^2$",
        "dcor" : "Distance correlation",
        "aidc" : "Affine invariant dist. corr",
        "hsic" : "Hilbert-Schmidt indep.cr."}

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
        #x = numpy.array([numpy.linspace(-1, 1, N_SAMPLES) for _ in range(N_FEATS)]).T
        x = numpy.array([numpy.random.uniform(-1, 1, N_SAMPLES) for _ in range(N_FEATS)]).T
        y = numpy.matmul(numpy.multiply(x, x), 2*numpy.array(range(N_FEATS)))

        _shapley_values = calc_shapley_values(x, y, PLAYERS, cf_name)
        all_shaps.append(_shapley_values)

    numpy.save(filename, all_shaps)

    return all_shaps

def normalise(x):
    return (x - numpy.mean(x))/(numpy.std(x))

def nice_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    return ax

def violinplot(values, positions, labels=None, multi=True):
    if not multi:
        assert len(values)==len(positions)
    # --- Axis formatting
    _, ax = plt.subplots()
    ax = nice_axes(ax)

    if multi:
        new_positions = [2*_p for _p in positions] # More x-axis space
        widths = [0.1 for _ in positions]

        for _n, _values in enumerate(values):
            _col = COLORS[_n]
            _positions = [1+_p + 0.2*_n for _p in new_positions]
            _bplot = ax.violinplot(_values,
                        positions=_positions,
                        #widths=widths,
                        quantiles=[[0.05, 0.95] for _ in range(5)],
                        showextrema=False,#
                        )

            if labels is not None:
                ax.plot([0], linestyle='-', label=labels[_n], c=_col)

            #[_bplot["bodies"][_p].set_color(_col) for _p in positions]
            [_bplot["bodies"][_p].set_edgecolor(_col) for _p in positions]
            [_bplot["bodies"][_p].set_facecolor(_col) for _p in positions]
        ax.set_xticks([_p+1 for _p in new_positions])
        ax.set_xticklabels([str(_p+1) for _p in positions])
        plt.legend()
        plt.ylabel("Normalised Shapley value")

    else:
        ax.violinplot(values, positions=positions)
        plt.ylabel("Shapley value")

    plt.xlabel("Player index")

def boxplot(values, positions, labels=None, multi=True):
    """
    If values is a list of numbers, you'll get a plot containing one box.
    If values contains lists of lists of numbers, you'll get a plot containing
    one box per list.
    If values is a list of lists which contain numbers and multi=True, you'll
    get a plot containing one box per sublist and list.
    """
    if not multi:
        assert len(values)==len(positions)
    # --- Axis formatting
    _, ax = plt.subplots()
    ax = nice_axes(ax)

    if multi:
        new_positions = [2*_p for _p in positions] # More x-axis space
        widths = [0.1 for _ in positions]

        for _n, _values in enumerate(values):
            _col = COLORS[_n]
            _positions = [1+_p + 0.2*_n for _p in new_positions]
            _bplot = ax.boxplot(_values,
                        positions=_positions,
                        showfliers=False,
                        widths=widths,
                        patch_artist=True,
                        medianprops=dict(color="black"),
                        )

            if labels is not None:
                ax.plot([0], linestyle='-', label=labels[_n], c=_col)

            [_bplot["boxes"][_p].set_facecolor(_col) for _p in positions]
        ax.set_xticks([_p+1 for _p in new_positions])
        ax.set_xticklabels([str(_p+1) for _p in positions])
        plt.legend()
        plt.ylabel("Normalised Shapley value")

    else:
        ax.boxplot(values, positions=positions, showfliers=False)
        plt.ylabel("Shapley value")

    plt.xlabel("Player index")
    plt.draw()

def barplot_all(xs, values):
    # --- Axis formatting
    _, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    plt.bar(xs, values)
    plt.draw()

if __name__ == "__main__":

    #CF_NAME = "hsic"
    #CF_NAME = "dcor"
    CF_NAME = "r2"
    #CF_NAME = "aidc"

    N_SAMPLES = 1000
    N_FEATS = 5

    DATA_DIR = "numpy_data"
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    N_ITER = 1000
    PLAYERS = list(range(N_FEATS))
    COLORS = ["orange", "blue", "green", "purple"]

    # --- Plot shapley decomposition per player for one cf:
    #all_shaps = calc_n_shapley_values(N_ITER, CF_NAME)
    # --- Non-normalised:
    #shaps_per_player = [numpy.array(all_shaps)[:,_player] for _player in PLAYERS]
    #violinplot(shaps_per_player, PLAYERS, multi=False)
    #boxplot(shaps_per_player, PLAYERS, multi=False)
    ## --- Normalised:
    #shaps_per_player = [normalise(numpy.array(all_shaps))[:,_player] for _player in PLAYERS]
    #boxplot(shaps_per_player, PLAYERS, multi=False)
    #violinplot(shaps_per_player, PLAYERS, multi=False)
    #plt.title(CF_DICT.get(CF_NAME, CF_NAME))
    #plt.show()
    #sys.exit()
    # ---

    # --- Plot shapley decompositions for all cfs per player
    cfs = ["r2", "dcor", "aidc", "hsic"]
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
    # ---

    cf_labels = [CF_DICT.get(_cf, 0) for _cf in cfs]
    violinplot(all_cf_shaps_per_player, PLAYERS, labels=cf_labels, multi=True)
    boxplot(all_cf_shaps_per_player, PLAYERS, labels=cf_labels, multi=True)

    # --- Plot average Shapley decomposition per player
    #SHAPS_AVG = [sum(x)/len(x) for x in zip(*all_shaps)]
    #barplot_all(PLAYERS, SHAPS_AVG)
    # ---

    #TODO: Save result to cleverly named file. Column headers with cf names

    try:
        plt.show()
    except Exception:
        pass
    sys.exit()



# NOTES
    #plt.title("Distance correlation")
    #plt.title(r"$R^2$")
    #plt.xlabel("Player index")
    #plt.ylabel("Avg Shapley value")
    #plt.ylim([0,0.005])
