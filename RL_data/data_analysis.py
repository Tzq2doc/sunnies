
import sys
from typing import Union
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Colormap
from xgboost import XGBRegressor, XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy
import matplotlib.pyplot as plt
import shap
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "../Python")
from xgb_regressor import (display_predictions, display_feature_importances,
    display_shapley, display_shapley_vs_xgb, display_shap,
    display_residuals_shapley)

#data = pd.read_csv("../RL_data/student-mat.csv")
data = pd.read_csv("student-mat.csv")
target = "G3"

# ----------------------------------------------------
#print(data.columns)


# ----------------------------------------------------
X_cat = data.select_dtypes(include=['object'])
X_enc = X_cat.copy()

# ----------------------------------------------------
# --- One-hot encoding non-numerical data for XGB
X_onehot = pd.get_dummies(X_cat, columns=X_cat.columns)
#print(X_onehot.head)
# ----------------------------------------------------

# ----------------------------------------------------
# --- Label encoding data for Sunnies
X_encoded = X_cat.apply(LabelEncoder().fit_transform) #
#print(X_encoded.head)
# --- End label encoding
# ----------------------------------------------------
data = data.drop(X_cat.columns, axis=1)
sunnies_data = pd.concat([data, X_encoded], axis=1)
xgb_data = pd.concat([data, X_onehot], axis=1)

#print(sunnies_data.columns)
#print(sunnies_data.head)
#print(xgb_data.columns)


y = sunnies_data[target].astype('int')
X = sunnies_data.drop([target], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_train = numpy.array(X_train)
y_train = numpy.array(y_train)
X_test = numpy.array(X_test)
y_test = numpy.array(y_test)
# ----------------------------------------------------

# --- Fit model and predict
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred
#print(accuracy_score(y_test, y_pred))
# ---

# --- Save predictions
#filename = "y_pred_xgb"
#numpy.save(filename, y_pred)
#print("Saved file {0}.npy".format(filename))
# ---

print(X.columns)
print(X.columns[14])#, 12, 13, 22, 4, 0, 6, 8, 27, 24])
sys.exit()

display_predictions(y_test, y_pred)
print("1")
display_feature_importances(model) # XGB built-in
#display_shapley(X_train, y_train, cf=["dcor"])#, "aidc"])#, "r2"])#"hsic"])
print("2")
#display_shapley_vs_xgb()
print("3")
#display_shap(X_test, model)
#display_residuals_shapley(X_test, residuals)
print("4")

plt.show()
