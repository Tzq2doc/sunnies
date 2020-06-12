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
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "../../../sunnies/Python")
from xgb_regressor import display_shapley
import shapley

load_data, Shapley, Shap, Pred, Linreg = [0, 0, 0, 0, 0]

load_data = True
Shapley = True
#Shap = True
#Pred = True
#Linreg = True

#MODELNAME = "slundberg_model.dat"
MODELNAME = "slundberg_small_xgb.dat"


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
if load_data:
    train_file = "train_data_slundberg.csv"
    test_file = "test_data_slundberg.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    #--- Data for Shapley calc
    shapley_data = pd.concat([train, test])
    shapley_data = shapley_data[features_with_target]
    shapley_data = shapley_data.dropna()
    X_shapley = shapley_data.drop(["target"], axis=1)
    X_shapley["sex_isFemale"] = [1 if _x else 0 for _x in X_shapley["sex_isFemale"]]
    LABELS = X_shapley.columns
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

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.1)

else:
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

    data = X.copy()
    data = X.drop(drop, axis=1)
    data["target"] = y
    data = data.dropna()

    # ---
    # Drop NaNs from data which goes into analysis: (unlike Slundberg)
    X = data.drop(["target"], axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

    test = X_test.copy()
    test["target"] = y_test
    train= X_train.copy()
    train["target"] = y_train
    train.to_csv("train_data_slundberg.csv")
    test.to_csv("test_data_slundberg.csv")

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.1)
    # ---

    #--- Data for Shapley calc
    shapley_data = data[features_with_target]
    X_shapley = shapley_data.drop(["target"], axis=1)
    X_shapley["sex_isFemale"] = [1 if _x else 0 for _x in X_shapley["sex_isFemale"]]
    LABELS = X_shapley.columns
    X_shapley = np.array(X_shapley[shapley_features])
    y_shapley = np.array(shapley_data["target"].astype('int'))
    # ---

# ---
# =============================================================================

def bce(truth, preds):
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


# === XGBOOST:
# --- Try to load model from file
if os.path.isfile(MODELNAME):
    xgb_model = xgboost.Booster()
    xgb_model.load_model(MODELNAME)
    print("Loaded model from file. Not training!")
    PREDS = xgb_model.predict(xgboost.DMatrix(X_test))

else:
    X_train, X_valid, y_train, y_valid= train_test_split(X_train, y_train, test_size=0.2, random_state=123)

    params = {
        "learning_rate": 0.001,
        "n_estimators": 6765,
        "max_depth": 4,
        "subsample": 0.5,
        "reg_lambda": 5.5,
        "reg_alpha": 0,
        "colsample_bytree": 1
    }

    xgb_model = xgboost.XGBRegressor()
    #xgb_model = xgboost.XGBRegressor(
    #    max_depth=params["max_depth"],
    #    n_estimators=params["n_estimators"],
    #    learning_rate=params["learning_rate"],#math.pow(10, params["learning_rate"]),
    #    subsample=params["subsample"],
    #    reg_lambda=params["reg_lambda"],
    #    colsample_bytree=params["colsample_bytree"],
    #    reg_alpha=params["reg_alpha"],
    #    n_jobs=16,
    #    random_state=1,
    #    objective="survival:cox",
    #    base_score=1
    #)

    xgb_model.fit(X_train, y_train,
            verbose=500,
            eval_set=[(X_valid, y_valid)],
            #eval_metric="logloss",
            early_stopping_rounds=10000
    )

    # --- Save model to file
    xgb_model.save_model(MODELNAME)
    print("Saved model to file {0}".format(MODELNAME))

    PREDS = xgb_model.predict(X_test)

if Shap:
    mapped_feature_names = list(map(lambda x: name_map.get(x, x), X_train.columns))
    # === SUMMARY PLOTS
    explainer = shap.TreeExplainer(xgb_model)
    xgb_shap = explainer.shap_values(X)
    xgb_shap_interaction = shap.TreeExplainer(xgb_model).shap_interaction_values(X)
    shap.dependence_plot(("Age", "Sex"), xgb_shap_interaction, X, feature_names=np.array(mapped_feature_names), show=False)
    #pl.savefig("raw_figures/nhanes_age_sex_interaction.pdf", dpi=400)
    pl.draw()

    shap.dependence_plot(("Age", "Sex"), xgb_shap_interaction, X, feature_names=np.array(mapped_feature_names), show=False)
    pl.draw()
    #pl.savefig("raw_figures/nhanes_age_sex_interaction.pdf", dpi=400)
    #pl.show()

    shap.dependence_plot(("Systolic blood pressure", "Age"), xgb_shap_interaction, X, feature_names=np.array(mapped_feature_names), show=False)
    pl.draw()
    #pl.savefig("raw_figures/nhanes_sbp_age_interaction.pdf", dpi=400)
    #pl.show()

    f = pl.figure(figsize=(4,6))
    shap.summary_plot(
        xgb_shap, X, feature_names=mapped_feature_names, plot_type="bar",
        max_display=15, auto_size_plot=False, show=False
    )
    pl.xlabel("mean(|SHAP value|)")
    #pl.savefig("raw_figures/nhanes_summary_bar.pdf", dpi=400)
    pl.draw()
    pl.show()


if Pred:
    # --- Try to load model from file
    if os.path.isfile(MODELNAME):
        xgb_model = xgboost.Booster()
        xgb_model.load_model(MODELNAME)
        print("Loaded model from file. Not training!")

    bces = [bce(_y, _p) for _y, _p in zip(y_test, PREDS)]
    plt.scatter(y_test, (np.log(PREDS)), c=bces, cmap='viridis')
    plt.colorbar()
    plt.xlabel("y_test")
    plt.ylabel("log preds")
    plt.show()
    #print(c_statistic_harrell(PREDS, y_test))

def do_shapley(modelname, preds, labels):
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

# === Shapley:
if Shapley:
    do_shapley(MODELNAME, PREDS, LABELS)
plt.show()

if Linreg:
    MODELNAME = "linreg_slundberg.dat"
    model = LinearRegression().fit(X_train, y_train)
    PREDS = model.predict(X_test)
    do_shapley(MODELNAME, PREDS, LABELS)
