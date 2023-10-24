import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from twoLayer import twoLayer
from ActivationFunctions import TanhFunc
from ErrLoss import PrimeMSE, meanSqureError
from Network import train, prediction

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    twoLayer(2, 3),
    TanhFunc(),
    twoLayer(3, 1),
    TanhFunc()
]

# train
train(network, meanSqureError, PrimeMSE, X, Y, epoch=10000, learningRate=0.1)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = prediction(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()