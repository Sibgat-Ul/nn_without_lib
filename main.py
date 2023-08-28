import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = 0
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
class Activation_Relu:
    def __init__(self):
        self.output = 0
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

def create_data(points, classes):
    x = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')

    for class_num in range(classes):
        ix = range(points*class_num, points*(class_num+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_num*4, (class_num+1)*4, points) + np.random.randn(points)*0.2
        x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_num
    return x, y

np.random.seed(0)

# x, y = create_data(100, 3)
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()
#
# plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
# plt.show()

data = np.random.rand(100, 784) # rows, columns

layer1 = layer_dense(784, 128)
layer2 = layer_dense(128, 10)

layer1.forward(data)
#print(layer1.output)
relu1 = Activation_Relu()
relu1.forward(layer1.output)
#print(relu1.output)

#softmax
#input dog,cat,human = [1, 2, 3] -> exponentiate = [e1, e2, e3] -> normalize = [e1/sum, e2/sum, e3/sum] -> out
layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]
E = 2.71828182846 # Euler's number
exp_values = np.exp(layer_outputs)

norm_values = []
layer_sums = []

for layer in exp_values:
    layer_sum = np.sum(layer)
    layer_sums.append(layer_sum)

    layer_norm = [value/layer_sum for value in layer]
    norm_values.append(layer_norm)

#using numpy
norm_values2 = exp_values / np.sum(exp_values, axis=1, keepdims=True)
