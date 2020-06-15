import matplotlib.pyplot as plt
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
from sklearn.linear_model import LinearRegression

modelname = "linreg_model.dat"

load_data =True
Shapley = True
Pred = False

shapley_features = [
    "sex_isFemale",
    "age",
    "systolic_blood_pressure",
    "bmi",
    "white_blood_cells",
    ]
features_with_target = shapley_features + ["target"]

name_map = {
    "sex_isFemale": "Sex",
    "age": "Age",
    "systolic_blood_pressure": "Systolic blood pressure",
    "bmi": "BMI",
    "white_blood_cells": "White blood cells", # (mg/dL)
    "sedimentation_rate": "Sedimentation rate",
    "serum_albumin": "Blood albumin",
    "alkaline_phosphatase": "Alkaline phosphatase",
    "cholesterol": "Total cholesterol",
    "physical_activity": "Physical activity",
    "hematocrit": "Hematocrit",
    "uric_acid": "Uric acid",
    "red_blood_cells": "Red blood cells",
    "urine_albumin_isNegative": "Albumin present in urine",
    "serum_protein": "Blood protein"
}
drop = ["creatinine", "BUN", "potassium", "sodium", "total_bilirubin",
        "segmented_neutrophils", "lymphocytes", "monocytes", "eosinophils",
        "basophils", "band_neutrophils", "calcium", "SGOT",
        "alkaline_phosphatase", "uric_acid", "sedimentation_rate",
        "red_blood_cells", "serum_protein"]

# =============================================================================
# ---
if not load_data:
    import loadnhanes

    X,y = loadnhanes._load()

    # clean up a bit
    for c in X.columns:
        if c.endswith("_isBlank"):
            del X[c]
    X["bmi"] = 10000 * X["weight"].values.copy() / (X["height"].values.copy() * X["height"].values.copy())
    del X["weight"]
    del X["height"]
    del X["urine_hematest_isTrace"] # would have no variance in the strain set
    del X["SGOT_isBlankbutapplicable"] # would have no variance in the strain set
    del X["calcium_isBlankbutapplicable"] # would have no variance in the strain set
    del X["uric_acid_isBlankbutapplicable"] # would only have one true value in the train set
    del X["urine_hematest_isVerylarge"] # would only have one true value in the train set
    del X["total_bilirubin_isBlankbutapplicable"] # would only have one true value in the train set
    del X["alkaline_phosphatase_isBlankbutapplicable"] # would only have one true value in the train set
    del X["hemoglobin_isUnacceptable"] # redundant with hematocrit_isUnacceptable

    rows = np.where(np.invert(np.isnan(X["systolic_blood_pressure"]) | np.isnan(X["bmi"])))[0]

    X = X.iloc[rows,:]
    y = y[rows]

    data = X.drop(drop, axis=1)
    data["target"] = y
    data = data.dropna()

    #--- Data for Shapley calc
    shapley_data = data[features_with_target]
    X_shapley = shapley_data.drop(["target"], axis=1)
    X_shapley["sex_isFemale"] = [1 if _x else 0 for _x in X_shapley["sex_isFemale"]]
    labels = X_shapley.columns
    X_shapley = np.array(X_shapley[shapley_features])
    y_shapley = np.array(shapley_data["target"].astype('int'))
    # ---

    # --- Split by patient id
    pids = np.unique(X.index.values)
    if X.shape[0] == len(pids):
        print("Only unique patient ids")

    data_0 = data[data['sex_isFemale'] == 0]
    data_1 = data[data['sex_isFemale'] == 1]

    train_0 = data_0.sample(n=400, random_state=1)
    train_1 = data_1.sample(n=4000, random_state=1)
    train = pd.concat([train_0, train_1])
    train.to_csv("train_data_9010.csv")

    remainder_0 = data_0.drop(train_0.index)
    remainder_1 = data_1.drop(train_1.index)

    test_0 = remainder_0.sample(n=1142, random_state=1)
    test_1 = remainder_1.sample(n=1142, random_state=1)
    test = pd.concat([test_0, test_1])
    test.to_csv("test_data_5050.csv")

    print("Train data: {0}".format(train.shape))
    print("Class 0: {0}".format(train_0.shape))
    print("Class 1: {0}".format(train_1.shape))
    print("Test data: {0}".format(test.shape))
    print("Class 0: {0}".format(test_0.shape))
    print("Class 1: {0}".format(test_1.shape))

    X_train = train.drop(["target"], axis=1)
    X_test = test.drop(["target"], axis=1)
    y_train = train["target"].astype('int')
    y_test = test["target"].astype('int')
# ---
# =============================================================================

if load_data:
    train_file = "train_data_9010.csv"
    test_file = "test_data_5050.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train = train.dropna()
    test = test.dropna()

    #--- Data for Shapley calc
    shapley_data = pd.concat([train, test])
    shapley_data = shapley_data[features_with_target]
    X_shapley = shapley_data.drop(["target"], axis=1)
    X_shapley["sex_isFemale"] = [1 if _x else 0 for _x in X_shapley["sex_isFemale"]]
    labels = X_shapley.columns
    convert_dict = {"sex_isFemale": float,
            "age" : float
            }

    X_shapley = X_shapley.astype(convert_dict)
    X_shapley = np.array(X_shapley)
    y_shapley = np.array(shapley_data["target"].astype('float'))
    # ---

    X_train = train.drop(["target"], axis=1)
    X_test = test.drop(["target"], axis=1)
    y_train = train["target"].astype('int')
    y_test = test["target"].astype('int')

    #X = pd.concat([X_train, X_test])

X = X_train.copy()
mapped_feature_names = list(map(lambda x: name_map.get(x, x), X.columns))

def bce(truth,preds):
    return np.mean(-truth*np.log(preds)-(1-truth)*np.log(1-preds))

def c_statistic_harrell(pred, labels):
    total = 0
    matches = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[j] > 0 and abs(labels[i]) > labels[j]:
                total += 1
                if pred[j] > pred[i]:
                    matches += 1
    return matches/total

# === LINEAR REGRESSION:
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)

if Pred:
    r_sq = model.score(X_test, y_test)
    print(f"Preds = {preds}")
    #bces = [bce(_y, _p) for _y, _p in zip(y_test, preds)]
    plt.scatter(y_test, preds, label=f"R2={r_sq}")#, c=bces, cmap='viridis')
    #plt.colorbar()
    #cbar.set_label(BCE)
    plt.xlabel("y_test")
    plt.ylabel("Linreg preds")
    plt.show()

    #print(c_statistic_harrell(preds, y_test))


# === Shapley:
if Shapley:
    sys.path.insert(0, "../../../sunnies/Python")
    from xgb_regressor import display_shapley
    import shapley

    _sfilename = "results/shapley_features_{0}.pickle".format(modelname)
    if not os.path.isfile(_sfilename):
        with open(_sfilename, 'wb') as _f:
            pickle.dump(labels, _f)


    # --- On data
    d = X_shapley.shape[1]
    x_range = list(range(d))

    _, ax = plt.subplots()
    if np.isnan(X_shapley).any():
        print("Data contains nan. Exiting")
        sys.exit()

    for _n, _cf in enumerate(["dcor", "r2", "aidc"]):
        print(_cf)
        _sfilename = "results/shapley_expl_{0}_{1}.pickle".format(_cf, modelname)

        if os.path.isfile(_sfilename):
            with open(_sfilename, 'rb') as _f:
                _shapley_values = pickle.load(_f)
        else:
            _shapley_values = shapley.calc_shapley_values(X_shapley, y_shapley, x_range, _cf)
            with open(_sfilename, 'wb') as _f:
                pickle.dump(_shapley_values, _f)

        print(_shapley_values)
        plt.bar(x_range + 0.1*_n*np.ones(d), _shapley_values, alpha=0.5,
                label=_cf, width=0.1)

    plt.title("Target")
    plt.legend()
    ax.set_xticks(x_range)
    ax.set_xticklabels(labels, rotation=90)
    plt.draw()

    # --- On predictions
    X_shapley_pred = X_test.copy()
    X_shapley_pred["sex_isFemale"] = [1 if _x else 0 for _x in X_shapley_pred["sex_isFemale"]]
    labels = X_shapley_pred.columns
    X_shapley_pred = np.array(X_shapley_pred[shapley_features])

    _, ax = plt.subplots()
    if np.isnan(X_shapley_pred).any():
        print("Data contains nan. Exiting")
        sys.exit()

    for _n, _cf in enumerate(["dcor", "r2", "aidc"]):
        print(_cf)
        _sfilename = "results/shapley_pred_{0}_{1}.pickle".format(_cf, modelname)

        if os.path.isfile(_sfilename):
            with open(_sfilename, 'rb') as _f:
                _shapley_values = pickle.load(_f)
        else:
            _shapley_values = shapley.calc_shapley_values(X_shapley_pred, preds, x_range, _cf)
        with open(_sfilename, 'wb') as _f:
            pickle.dump(_shapley_values, _f)

        print(_shapley_values)
        plt.bar(x_range + 0.1*_n*np.ones(d), _shapley_values, alpha=0.5,
                label=_cf, width=0.1)

    plt.title("Predictions")
    plt.legend()
    ax.set_xticks(x_range)
    ax.set_xticklabels(labels, rotation=90)

    plt.draw()

    # --- On residuals
    _, ax = plt.subplots()
    residuals = y_test - preds
    print(X_shapley_pred.shape)
    print(residuals.shape)

    if np.isnan(X_shapley_pred).any():
        print("Data contains nan. Exiting")
        sys.exit()

    for _n, _cf in enumerate(["dcor", "r2", "aidc"]):
        print(_cf)
        _sfilename = "results/shapley_res_{0}_{1}.pickle".format(_cf, modelname)

        if os.path.isfile(_sfilename):
            with open(_sfilename, 'rb') as _f:
                _shapley_values = pickle.load(_f)
        else:
            _shapley_values = shapley.calc_shapley_values(X_shapley_pred,
                    residuals, x_range, _cf)
        with open(_sfilename, 'wb') as _f:
            pickle.dump(_shapley_values, _f)

        print(_shapley_values)
        plt.bar(x_range + 0.1*_n*np.ones(d), _shapley_values, alpha=0.5,
                label=_cf, width=0.1)

    plt.title("Residuals")
    plt.legend()
    ax.set_xticks(x_range)
    ax.set_xticklabels(labels, rotation=90)

    plt.draw()

plt.show()
