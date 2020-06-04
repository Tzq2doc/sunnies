import xgboost
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
X = X.iloc[rows,:]
y = y[rows]

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

# split by patient id
pids = np.unique(X.index.values)
if X.shape[0] == len(pids):
    print("Only unique patient ids")

#print(X.shape)
#print(y.shape)

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

train_pids, test_pids = train_test_split(pids, random_state=0, test_size=0.3)
strain_pids,valid_pids = train_test_split(train_pids, random_state=0)

# find the indexes of the samples from the patient ids
train_inds = np.where([p in train_pids for p in X.index.values])[0]
strain_inds = np.where([p in strain_pids for p in X.index.values])[0]
valid_inds = np.where([p in valid_pids for p in X.index.values])[0]
test_inds = np.where([p in test_pids for p in X.index.values])[0]

# create the split datasets
X_train = X.iloc[train_inds,:]
X_strain = X.iloc[strain_inds,:]
X_valid = X.iloc[valid_inds,:]
X_test = X.iloc[test_inds,:]
y_train = y[train_inds]
y_strain = y[strain_inds]
y_valid = y[valid_inds]
y_test = y[test_inds]

# mean impute for linear and deep models
imp = Imputer()
imp.fit(X_strain)
X_strain_imp = imp.transform(X_strain)
X_train_imp = imp.transform(X_train)
X_valid_imp = imp.transform(X_valid)
X_test_imp = imp.transform(X_test)
X_imp = imp.transform(X)

# standardize
scaler = StandardScaler()
scaler.fit(X_strain_imp)
X_strain_imp = scaler.transform(X_strain_imp)
X_train_imp = scaler.transform(X_train_imp)
X_valid_imp = scaler.transform(X_valid_imp)
X_test_imp = scaler.transform(X_test_imp)
X_imp = scaler.transform(X_imp)




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
xgb_model.fit(
    X_strain, y_strain, verbose=500,
    eval_set=[(X_valid, y_valid)],
    #eval_metric="logloss",
    early_stopping_rounds=10000
)

# save model to file
pickle.dump(xgb_model, open("model.pickle.dat", "wb"))
# load model from file
#loaded_model = pickle.load(open("model.pickle.dat", "rb"))

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
print(c_statistic_harrell(xgb_model.predict(X_test), y_test))
