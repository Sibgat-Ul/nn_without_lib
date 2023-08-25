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


x, y = create_data(100, 3)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
plt.show()

data = np.random.rand(100, 784) # rows, columns

layer1 = layer_dense(784, 128)
layer2 = layer_dense(128, 10)

layer1.forward(data)
print(layer1.output)
relu1 = Activation_Relu()
relu1.forward(layer1.output)
print(relu1.output)

# #output = input*weight + bias
# output = (input[0]*weights[0] + input[1]*weights[1] + input[2]*weights[2]) + biases
#
# #3 neurons, 4 inputs
# input = (np.random.rand(4)*10).round(2)
# w1 = np.random.rand(4)
# w2 = np.random.rand(4)
# w3 = np.random.rand(4)
#
# b1 = np.random.rand(1)
# b2 = np.random.rand(1)
# b3 = np.random.rand(1)
#
# #weights variable is a list consisting of 3 lists (w1, w2, w3)
# weights = [w1, w2, w3]
# biases = [b1, b2, b3]
#
# output = 0
# nn_output_list = []
#
# # Dot product of input and weights
# for w_list, b in zip(weights, biases):
#     n_out = 0
#     for inp, w in zip(input, w_list):
#         n_out += inp*w
#     n_out += b
#     output += n_out
#     nn_output_list.append(n_out)
#
# #using numpy
# numpy_prod = 0
# for w_list, b in zip(weights, biases):
#     numpy_prod += np.dot(input, w_list) + b
#
# print(output, numpy_prod)
# print(nn_output_list)