import sys
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


#0, 3,4,7,9,14
data = pd.read_csv('processed.cleveland.data', sep=",", header=0)
#data = pd.read_csv("student-mat.csv")
#X_data = pd.read_csv("X_data.csv")

# ----------------------------------------------------
print(data.columns)
#print("Dropping most of the data here because it's type string.")
#data = data.select_dtypes([np.number])
#print(data.columns)
# ----------------------------------------------------
data_cont = data.iloc[:, [0,1,3,4,7,9,13]]

# --- Plot all data
#sns.pairplot(data_cont, hue='sex', size=2.5)
#plt.show()

data_0 = data[data['sex'] == 0]
data_1 = data[data['sex'] == 1]
print("Instances of sex 0:", str(data_0.shape[0]))
print("Instances of sex 1:", str(data_1.shape[0]))

X_0, y_0 = data_0.iloc[:, :-1], data_0.iloc[:, -1]
X_1, y_1 = data_1.iloc[:, :-1], data_1.iloc[:, -1]
X_0_train, X_0_test, y_0_train, y_0_test = train_test_split(X_0, y_0, test_size=0.2, random_state=123)
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1, test_size=0.2, random_state=123)


#X, y = data.iloc[:, :-1], data.iloc[:, -1]
#data_dmatrix = xgb.DMatrix(data=X,label=y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#print(type(X_train))
#X_train=np.array(X_train)
#print(type(X_train))

xg_0 = xgb.XGBRegressor()
xg_0.fit(X_0_train, y_0_train)

xg_1 = xgb.XGBRegressor()
xg_1.fit(X_1_train, y_1_train)

preds_0_0 = xg_0.predict(X_0_test)
rmse_0_0 = np.sqrt(mean_squared_error(y_0_test, preds_0_0))

preds_1_1 = xg_1.predict(X_1_test)
rmse_1_1 = np.sqrt(mean_squared_error(y_1_test, preds_1_1))

preds_1_0 = xg_1.predict(X_0)
rmse_1_0 = np.sqrt(mean_squared_error(y_0, preds_1_0))

preds_0_1 = xg_0.predict(X_1)
rmse_0_1 = np.sqrt(mean_squared_error(y_1, preds_0_1))

print("RMSE trained on 0, tested on 0: %f" % (rmse_0_0))
print("RMSE trained on 1, tested on 1: %f" % (rmse_1_1))
print("RMSE trained on 0, tested on 1: %f" % (rmse_0_1))
print("RMSE trained on 1, tested on 0: %f" % (rmse_1_0))
