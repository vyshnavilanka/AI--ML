import torch
import numpy as np

# From a Python list
data_list = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data_list)
print('List Tensor \n',tensor_from_list)

# From a NumPy array
data_np = np.array([[1, 2], [3, 4]])
tensor_from_np = torch.tensor(data_np)
print('Numpy Tensor \n',tensor_from_np)

# Using Built-in Functions:

# Creating a tensor of zeros
zeros_tensor = torch.zeros((2, 3))
print(zeros_tensor)

# Creating a tensor of ones
ones_tensor = torch.ones((2, 3))
print(ones_tensor)

# Creating a tensor with random values
rand_tensor = torch.rand((2, 3))
print(rand_tensor)