import numpy as np

inputs_size = 2
hidden_size = 3
output_size = 1
wieght_2_input = np.random.randn(hidden_size, inputs_size)#3*2
inputs = np.array([
    [3],
    [2]
])
bias = np.random.randn(3, 1)#3*1
output_1 = np.dot(wieght_2_input,inputs) + bias #3*1
weight_final = np.random.randn(hidden_size, output_size)#3*1
bias_output = np.random.random((1, 1))#1*1
output = np.array([[1]])
final_output = np.dot(weight_final.transpose(),output_1) + bias_output # 1*1
print(final_output)
error = output - final_output#to calculate mistake percentage