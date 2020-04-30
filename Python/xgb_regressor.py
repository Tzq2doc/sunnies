from typing import Union
from itertools import combinations

from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Colormap
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy
import matplotlib.pyplot as plt
import shapley as shapley
import shap

def make_xgb_dict(x, y):
    rmse_dict = {}
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=7)
    players = list(range(x_train.shape[1]))
    num_players = len(players)
    team_sizes = list(range(num_players+1))
    for _size in team_sizes:
        teams_of_size_s = list(combinations(players, _size))
        for _team in teams_of_size_s:
            if len(_team)==0:
                rmse_dict[_team] = 10. #TODO check
                continue
            if len(_team)==1:
                _x_train = numpy.reshape(x_train[:, _team[0]], (x_train.shape[0], 1))
                _x_test = numpy.reshape(x_test[:, _team[0]], (x_test.shape[0], 1))
            else:
                _x_train = x_train[:, _team]
                _x_test = x_test[:, _team]

            _model = XGBRegressor()
            _model.fit(_x_train, y_train)
            _pred = _model.predict(_x_test)
            _rmse = numpy.sqrt(mean_squared_error(y_test, _pred))
            rmse_dict[_team] = _rmse
    return rmse_dict

def display_shapley(cf="dcor"):
    shapley_values_actual = shapley.calc_shapley_values(X_test, y_test, list(range(D)), cf)
    shapley_values_xgb = shapley.calc_shapley_values(X_test, y_pred, list(range(D)), cf)

    fig, ax = plt.subplots()
    ax = nice_axes(ax)

    plt.bar(range(len(shapley_values_actual)), shapley_values_actual, color="red",
            alpha=0.5, label="True")
    plt.bar(range(len(shapley_values_xgb)), shapley_values_xgb, color="blue",
            alpha=0.5, label="Predicted")
    plt.legend()
    plt.show()


def display_predictions(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.show()


def display_feature_importances(model):
    plot_importance(model)

    feature_importance = model.feature_importances_

    print("Feature importances:")
    for _n, _imp in enumerate(feature_importance):
        print("Feature {0}: {1}".format(_n, _imp))

    fig, ax = plt.subplots()
    ax = shapley.nice_axes(ax)

    plt.bar(range(len(feature_importance)), feature_importance)
    plt.show()

def display_residuals_shapley(x, residuals, cf="dcor"):
    d = x.shape[1]
    shapley_values_residuals = shapley.calc_shapley_values(x, residuals, list(range(d)), cf)
    fig, ax = plt.subplots()
    ax = shapley.nice_axes(ax)

    plt.bar(range(len(shapley_values_residuals)), shapley_values_residuals, color="red",
            alpha=0.5)
    plt.show()

def display_shap(x, model):
    # --- SHAP package
    explainer = shap.TreeExplainer(model)
    expected_value = explainer.expected_value
    shap_values = explainer.shap_values(x)

    plt.figure()
    shap.summary_plot(shap_values, x, show=False)

    # --- Change the colormap
    my_cmap = plt.get_cmap('viridis')
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(my_cmap)

if __name__ == "__main__":


    # --- Data with independent quadratic features
    D = 5
    N = 1000
    #X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
    X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
    TWO_D = 2 * numpy.array(range(D))
    Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
    # ---

    ## --- Data with no relationship
    #D = 5
    #N = 1000
    #X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D + 1)]).T
    #Y, X = X[:, 0], X[:, 1:]
    ## ---

    # --- Data with high correlations
    #D = 5
    #N = 1000
    #sigma = 0.2
    #X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
    #X[:, 1] = X[:, 0] + numpy.random.normal(0, sigma, N)
    #X[:, 2] = X[:, 0] + numpy.random.normal(0, sigma, N)
    #Y = numpy.matmul(numpy.multiply(X, X), numpy.ones(D))
    # ---

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=7)

    # --- Fit model and predict
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    # ---

    # --- Save predictions
    #filename = "y_pred_xgb"
    #numpy.save(filename, y_pred)
    #print("Saved file {0}.npy".format(filename))
    # ---

    #display_predictions(y_test, y_pred)
    display_feature_importances(model)
    #display_shapley()
    #display_shap(X_test, model)
    #display_residuals_shapley(X_test, residuals)
