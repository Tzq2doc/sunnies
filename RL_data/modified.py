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
import loadnhanes
import lifelines
import scipy
import sys


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
mapped_feature_names = list(map(lambda x: name_map.get(x, x), X.columns))

X = X.iloc[rows,:]
y = y[rows]

data = X.copy()
data["target"] = y

# --- Split by patient id
pids = np.unique(X.index.values)
if X.shape[0] == len(pids):
    print("Only unique patient ids")

data_0 = data[data['sex_isFemale'] == 0]
data_1 = data[data['sex_isFemale'] == 1]
#pids_0 = np.unique(data_0.index.values)
#pids_1 = np.unique(data_1.index.values)


# ==============================================
# TODO:
# calculate the shapley values of the predictions, the residuals, and the labels,
# both on the training set and the test set.
# hopefully find that blood pressure contributes to the residuals more in females (on the deployed model), when the model is trained only on males
# ==============================================

train_0 = data_0.sample(n=3500, random_state=1)
train_1 = data_1.sample(n=3500, random_state=1)
train = pd.concat([train_0, train_1])
train.to_csv("train_data_5050.csv")

remainder_0 = data_0.drop(train_0.index)
remainder_1 = data_1.drop(train_1.index)


test_0 = remainder_0.sample(n=400, random_state=1)
test_1 = remainder_1.sample(n=4000, random_state=1)
test = pd.concat([test_0, test_1])
test.to_csv("test_data_9010.csv")

# --- Sanity check
#List1 = test.index.values
#List2 = train.index.values
#print(any(item in List1 for item in List2))
#print(test.shape)
#print(train.shape)


X_train = train.drop(["target"], axis=1)
X_test = test.drop(["target"], axis=1)
y_train = train["target"].astype('int')
y_test = test["target"].astype('int')

# ----------------------------------------------------------------
# --- Save data
#np.save(X, "X_data.npy")
#np.save(y, "y_data.npy")
#header = "{0}\n".format(",".join(list(X.columns)))
#with open("X_data_with_header.csv", 'wb') as f:
#  f.write(bytes(header, "UTF-8"))
#  np.savetxt(f, X, delimiter=',')
#
#np.savetxt('X_data.csv', X, delimiter=',')
#np.savetxt('y_data.csv', y, delimiter=',')
# ----------------------------------------------------------------


# =====================================================
# === The following is implemented but as far as I can tell not used by slundberg
# Mean impute for linear and deep models
imp = Imputer()
imp.fit(X_train)
X_train_imp = imp.transform(X_train)
X_test_imp = imp.transform(X_test)
X_imp = imp.transform(X)

# standardize
scaler = StandardScaler()
scaler.fit(X_train_imp)
X_train_imp = scaler.transform(X_train_imp)
X_test_imp = scaler.transform(X_test_imp)
X_imp = scaler.transform(X_imp)
# =====================================================



X_train, X_valid, y_train, y_valid= train_test_split(X_train, y_train, test_size=0.2, random_state=123)

# === TRAIN XGBOOST
# these parameters were found using the Tune XGboost on NHANES notebook (coordinate decent)
params = {
    "learning_rate": 0.001,
    "n_estimators": 6765,
    "max_depth": 4,
    "subsample": 0.5,
    "reg_lambda": 5.5,
    "reg_alpha": 0,
    "colsample_bytree": 1
}

modelname = "full_model.pickle.dat"
#modelname = "test_model.dat"

# --- Try to load model from file
if os.path.isfile(modelname):
    xgb_model = xgboost.Booster()
    xgb_model.load_model(modelname)
    print("Loaded model from file. Not training!")

else:
    #xgb_model = xgboost.XGBRegressor()
    xgb_model = xgboost.XGBRegressor(
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],#math.pow(10, params["learning_rate"]),
        subsample=params["subsample"],
        reg_lambda=params["reg_lambda"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        n_jobs=16,
        random_state=1,
        objective="survival:cox",
        base_score=1
    )

    xgb_model.fit(X_train, y_train,
            verbose=500,
            eval_set=[(X_valid, y_valid)],
            #eval_metric="logloss",
            early_stopping_rounds=10000
    )

    # --- Save model to file
    xgb_model.save_model(modelname)
    print("Saved model to file {0}".format(modelname))



# === SUMMARY PLOTS
explainer = shap.TreeExplainer(xgb_model)
xgb_shap = explainer.shap_values(X)
xgb_shap_interaction = shap.TreeExplainer(xgb_model).shap_interaction_values(X)
shap.dependence_plot(("Age", "Sex"), xgb_shap_interaction, X, feature_names=np.array(mapped_feature_names), show=False)
#pl.savefig("raw_figures/nhanes_age_sex_interaction.pdf", dpi=400)
#pl.show()
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

