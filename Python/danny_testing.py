from typing import Union

from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Colormap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy
import matplotlib.pyplot as plt
import shapley as shapley
import shap
X,y = shap.datasets.nhanesi()


import shap

# --- Data with no relationship
D = 5
N = 1000
X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D + 1)]).T
Y, X = X[:, 0], X[:, 1:]
# ---

# --- Data with high correlations
D = 5
N = 1000
sigma = 0.2
X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
X[:, 1] = X[:, 0] + numpy.random.normal(0, sigma, N)
X[:, 2] = X[:, 0] + numpy.random.normal(0, sigma, N)
Y = numpy.matmul(numpy.multiply(X, X), numpy.ones(D))
# ---

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# --- Fit model
model = XGBRegressor()
model.fit(X_train, y_train)

# --- Predict
y_pred = model.predict(X_test)

# --- Feature importances
feature_importance = model.feature_importances_

shapley_values_actual = shapley.calc_shapley_values(X_test, y_test, "dcor")
shapley_values_xgb = shapley.calc_shapley_values(X_test, y_pred, "dcor")

print(shapley_values_actual)
print(shapley_values_xgb)
display_predictions()
display_feature_importances()
display_shapley()
