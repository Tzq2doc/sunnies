
import sys
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# --- My stuff
sys.path.insert(0, "../Python")
from xgb_regressor import (display_predictions, display_feature_importances,
    display_shapley, display_shapley_vs_xgb, display_shap,
    display_residuals_shapley)
import shapley



X_data = pd.read_csv("X_data_with_header.csv")
y_data = pd.read_csv("y_data.csv", header=None)
#print(X_data.columns)
#print(X_data.shape)
#print(y_data.shape)

feats = [
        "sex_isFemale",
        "age",
        "systolic_blood_pressure",
        "white_blood_cells", # has NaN
        "bmi",
        "cholesterol", # has NaN
        "serum_albumin", # has NaN
        "alkaline_phosphatase", # has NaN
        "physical_activity",
        "hematocrit", # has NaN
        "uric_acid", # has NaN
        "red_blood_cells", # has NaN
        "urine_albumin_isNegative",
        "serum_protein", # has NaN
]

X_data = X_data[feats]
# --- Check for NaNs:
X_data = X_data.dropna(axis=0)
#print(X_data.isnull().any())



X_data = np.array(X_data[feats])
y_data = np.array(y_data)
y_data = y_data.T[0]
print(X_data.shape)
print(y_data.shape)


# --- Plot all data
#sns.pairplot(data_cont, hue='sex', size=2.5)
#plt.show()

x_range = list(range(X_data.shape[1]))
shapley_values = shapley.calc_shapley_values(X_data, y_data, x_range, "r2")
print(shapley_values)
shapley_values = shapley.calc_shapley_values(X_data, y_data, x_range, "dcor")
print(shapley_values)
display_shapley(X_data, y_data, cf=["r2"])
display_shapley(X_data, y_data, cf=["dcor"])
#plt.show()
#display_shapley(X_data, y_data, cf=["dcor", "aidc", "r2", "hsic"])


















# NOTES
sys.exit()
data = pd.read_csv('processed.cleveland.data', sep=",", header=0)
data = data.iloc[:, [0,1,3,4,7,9,13]]
target = "num"
y = data[target].astype('int')
X = data.drop([target], axis=1)
y_test = np.array(y)
X_test = np.array(X)
print(X_test.shape)
print(y_test.shape)
display_shapley(X_test, y_test, cf=["dcor"])
