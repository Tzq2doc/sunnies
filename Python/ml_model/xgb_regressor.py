from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy


# --- Data
D = 5
N = 1000

#X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
TWO_D = 2*numpy.array(range(D))
Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
# ---

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
        random_state=7)

# --- Fit model
model = XGBRegressor()
model.fit(X_train, y_train)

# --- Predict
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.show()
