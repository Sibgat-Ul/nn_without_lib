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

class Activation_softmax:
   def forward(self, inputs):
       exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
       probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
       self.output = probabilities

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

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(selfs, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

np.random.seed(0)

x, y = create_data(100, 3)

dense1 = layer_dense(2, 3)
activation1 = Activation_Relu()

dense2 = layer_dense(3, 3)
activation2 = Activation_softmax()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print('Loss: ', loss)




