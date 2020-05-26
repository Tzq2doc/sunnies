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

# ----------------------------------------------------
print(data.columns)
#print("Dropping most of the data here because it's type string.")
#data = data.select_dtypes([np.number])
#print(data.columns)
# ----------------------------------------------------
data_cont = data.iloc[:, [0,1,3,4,7,9,13]]

print(data_cont)
sns.pairplot(data_cont, hue='sex', size=2.5)
plt.show()

sys.exit()


X, y = data.iloc[:, :-1], data.iloc[:, -1]
data_dmatrix = xgb.DMatrix(data=X,label=y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(type(X_train))
X_train=np.array(X_train)
print(type(X_train))

xg_reg = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1,
        max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
