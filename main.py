import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('./digit_set/train.csv')
test_data = pd.read_csv('./digit_set/test.csv')

input = [1.0, 2.0, 3.0]
weights = [0.2, 0.8, -0.5]
bias = 2.0

output = (input[0]*weights[0] + input[1]*weights[1] + input[2]*weights[2]) + bias
print(output)