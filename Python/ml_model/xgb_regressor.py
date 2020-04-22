from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy
import matplotlib.pyplot as plt


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

plt.scatter(y_test, y_pred)
plt.show()

# --- Feature importances
feature_importance = model.feature_importances_
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
