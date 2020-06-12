import matplotlib.pyplot as plt
import xgboost
import os
import pickle
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler#, Imputer
from sklearn.impute import SimpleImputer as Imputer
import sklearn
import matplotlib.pyplot as pl
import numpy as np
from tqdm import tqdm
import keras
import pandas as pd
import lifelines
import scipy
import sys



sys.path.insert(0, "../../../sunnies/Python")
from xgb_regressor import display_shapley
import shapley

cf = "dcor" #"aidc" #"r2" #"dcor"
kind = ["expl", "pred", "res"]

_, ax = plt.subplots()
#modelname = "linreg_model.dat"
#modelname= "full_model.dat"
modelname = "small_xgb.dat"
with open(f"results/shapley_features_{modelname}.pickle", 'rb') as _f:
        labels = pickle.load(_f)
d = len(labels)
x_range = list(range(d))

for _n, _kind in enumerate(kind):
    _filename = f"results/shapley_{_kind}_{cf}_{modelname}.pickle"
    with open(_filename, 'rb') as _f:
        _shapley_values = pickle.load(_f)

    plt.bar(x_range + 0.1*_n*np.ones(d), _shapley_values, alpha=0.5,
            label=_kind, width=0.1)

    plt.title(f"Shapley decomp of {cf} on {modelname}")
    plt.legend()
    ax.set_xticks(x_range)
    ax.set_xticklabels(labels, rotation=30)
    plt.draw()

plt.show()

