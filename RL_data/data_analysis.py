
import pandas as pd
import numpy
import sys
sys.path.insert(0, "../Python")
from xgb_regressor import (display_predictions, display_feature_importances,
    display_shapley, display_shapley_vs_xgb, display_shap,
    display_residuals_shapley)

#data = pd.read_csv("../RL_data/student-mat.csv")
data = pd.read_csv("student-mat.csv")

# ----------------------------------------------------
print(data.columns)
print("Dropping most of the data here because it's type string.")
data = data.select_dtypes([numpy.number])
print(data.columns)
# ----------------------------------------------------

X, y = data.iloc[:, :-1], data.iloc[:, -1]

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

#display_predictions(y_test, y_pred)
display_feature_importances(model)
display_shapley(X_train, y_train, cf=["dcor", "aidc"])#, "r2"])#"hsic"])
#display_shapley_vs_xgb()
display_shap(X_test, model)
#display_residuals_shapley(X_test, residuals)

plt.show()
