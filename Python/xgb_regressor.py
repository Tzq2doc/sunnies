from typing import Union

from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Colormap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy
import matplotlib.pyplot as plt
import shapley as shapley

import shap

## --- Data with independent quadratic features
#D = 5
#N = 1000
## X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
#X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
#TWO_D = 2 * numpy.array(range(D))
#Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
## ---
#
## --- Data with no relationship
#D = 5
#N = 1000
#X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D + 1)]).T
#Y, X = X[:, 0], X[:, 1:]
## ---
#
# --- Data with high correlations
D = 5
N = 1000
sigma = 0.2
X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
X[:, 1] = X[:, 0] + numpy.random.normal(0, sigma, N)
X[:, 2] = X[:, 0] + numpy.random.normal(0, sigma, N)
Y = numpy.matmul(numpy.multiply(X, X), numpy.ones(D))
# ---

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=7)

# --- Fit model
model = XGBRegressor()
model.fit(X_train, y_train)


# --- Predict
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# --- Save predictions
#filename = "y_pred_xgb"
#numpy.save(filename, y_pred)
#print("Saved file {0}.npy".format(filename))

# --- Feature importances
feature_importance = model.feature_importances_

shapley_values_actual = shapley.calc_shapley_values(X_test, y_test, list(range(D)), "dcor")
shapley_values_xgb = shapley.calc_shapley_values(X_test, y_pred, list(range(D)), "dcor")
shapley_values_residuals = shapley.calc_shapley_values(X_test, residuals, list(range(D)), "dcor")


print(shapley_values_actual)
print(shapley_values_xgb)


def display_shapley():
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    plt.bar(range(len(shapley_values_actual)), shapley_values_actual, color="red",
            alpha=0.5, label="True")
    plt.bar(range(len(shapley_values_xgb)), shapley_values_xgb, color="blue",
            alpha=0.5, label="Predicted")
    plt.legend()
    plt.show()


def display_predictions():
    plt.scatter(y_test, y_pred)
    plt.show()


def display_feature_importances():
    print("Feature importances:")
    for _n, _imp in enumerate(feature_importance):
        print("Feature {0}: {1}".format(_n, _imp))

    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    plt.bar(range(len(feature_importance)), feature_importance)
    plt.show()

def display_residuals_shapley():
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    plt.bar(range(len(shapley_values_residuals)), shapley_values_residuals, color="red",
            alpha=0.5)
    plt.show()

def display_shap():
    # --- SHAP package
    explainer = shap.TreeExplainer(model)
    expected_value = explainer.expected_value
    shap_values = explainer.shap_values(X_test)
    my_cmap = plt.get_cmap('viridis')
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    # Change the colormap of the artists
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(my_cmap)
display_predictions()
display_feature_importances()
display_shapley()
# display_shap()
display_residuals_shapley()
