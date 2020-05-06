from matplotlib.collections import PatchCollection
import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw():
    x = numpy.random.uniform(-3, 3)
    y = x*x
    return x, y

def width(x0, x1):
    return x1- x0

def heigth(y0, y1):
    return y1- y0

x0, y0 = draw()
x1, y1 = draw()

#def one_rectangle(x0, x1):
#
#    rectangle = Rectangle((x0, y0), width(x0, x1), heigth(y0, y1),
#            linewidth=1,
#            color='red',
#            fill=False
#            )
#
#ax = plt.gca()
#for _ in range(10):
#    _rectangle = one_rectangle(x0, x1)
#    ax.add_patch(_rectangle)

def make_rectangles(n):

    pos_rectangles = []
    neg_rectangles = []

    for _ in range(n):
        _x0, _y0 = draw()
        _x1, _y1 = draw()
        _width = width(_x0, _x1)
        _height = heigth(_y0, _y1)
        if (_width > 0 and _height > 0) or (_width < 0 and _height < 0):
            pos_rectangles.append(Rectangle((_x0, _y0), _width, _height))
            plt.scatter(_x0, _y0, c='red')
            plt.scatter(_x1, _y1, c='red')
        else:
            neg_rectangles.append(Rectangle((_x0, _y0), _width, _height))
            plt.scatter(_x0, _y0, c='blue')
            plt.scatter(_x1, _y1, c='blue')

    # Create patch collection with specified colour/alpha
    pos_pc = PatchCollection(pos_rectangles, facecolor="None", edgecolor='red')
    neg_pc = PatchCollection(neg_rectangles, facecolor="None", edgecolor='blue')

    # Add collection to axes
    ax.add_collection(pos_pc)
    ax.add_collection(neg_pc)

ax = plt.gca()
make_rectangles(10)
plt.xlim([-4, 4])
plt.ylim([0, 10])
plt.show()
