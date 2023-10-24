import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from twoLayer import twoLayer
from ConvolutionLayer import Convolution
from reshape import ReshapeInput
from ActivationFunctions import Sigmoid
from ErrLoss import BinaryClassifiction_entropy, PrimeBinaryClass
from Network import train, prediction

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y, num_classes=2)  # Use to_categorical for one-hot encoding
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolution((1, 28, 28), 3, 5),
    Sigmoid(),
    ReshapeInput((5, 26, 26), (5 * 26 * 26, 1)),
    twoLayer(5 * 26 * 26, 100),
    Sigmoid(),
    twoLayer(100, 2),
    Sigmoid()
]

# train
train(
    network,
    BinaryClassifiction_entropy,
    PrimeBinaryClass,
    x_train,
    y_train,
    epoch=20,
    learningRate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = prediction(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
