import matplotlib.pyplot as plt
import scipy.linalg
import numpy
import sys
from shapley import nice_axes

D = 4
N = 1000
N_ITER = 5

def make_data(d, n):
    #x = numpy.array([numpy.linspace(-1, 1, n) for _ in range(d)]).T
    x = numpy.array([numpy.random.uniform(-1, 1, n) for _ in range(d)]).T
    two_d = 2*numpy.array(range(d))
    epsilon = numpy.random.uniform(-0.2,0.2,N)
    y = numpy.matmul(numpy.multiply(x, x), two_d) + epsilon

    return x, y

X, Y = make_data(D, N)

#corr = []
#for _ in range(N_ITER):
#    _x, _y = make_data(D, N)
#    corr.append([numpy.corrcoef(X[:, _d], Y) for _d in range(D)])

#fig, ax = plt.subplots()
#ax.set_title('Hide Outlier Points')
#ax.boxplot(data, showfliers=False)
#plt.boxplot(data, showfliers=False)
#plt.show()


def least_squares(x_data, y_data):
    _, ax = plt.subplots()
    ax = nice_axes(ax)

    A = numpy.vstack([x_data, numpy.ones(len(x_data))]).T
    solution = numpy.linalg.lstsq(A, y_data)
    m, c = solution[0]
    resid = solution[1][0]

    r2 = 1 - resid / (y_data.size * y_data.var())

    x = numpy.linspace(-1, 1, 100)
    plt.plot(x, m*x+c, 'r', label=r"$R^2=${0}".format(round(r2, 4)))
    plt.scatter(x_data, y_data, alpha=0.8)
    plt.xlabel(r"$X_4$", fontsize=16)
    plt.ylabel(r"$Y$", fontsize=16)
    plt.legend(fontsize=16)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.show()

# -- Plot
def plot_data():
    #plt.scatter(X[:,0], Y, label="X0", alpha=0.8)
    #plt.scatter(X[:,1], Y, label="X1", alpha=0.8)
    #plt.scatter(X[:,2], Y, label="X2", alpha=0.8)
    #plt.scatter(X[:,3], Y, label="X3", alpha=0.8)
    #plt.scatter(X[:,3], X[:,0], label="X3 vs X0", alpha=0.8)
    plt.legend(loc="upper right")
    plt.show()

#plot_data()
least_squares(X[:,3], Y)
# ---
