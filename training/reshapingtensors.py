import torch
import numpy as np
#Reshaping Tensors (view, reshape)

# Creating a tensor
tensor_data1 = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
#tensor_data2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# Reshaping with view (note: view requires the tensor to be contiguous in memory)
reshaped_view = tensor_data1.view(4, 2)
print("Reshaped with view:", reshaped_view)

# Reshaping with reshape (more flexible)
reshaped_reshape = tensor_data1.reshape(4, 2)
print("Reshaped with reshape:", reshaped_reshape)

# Slicing and Indexing Tensors

# Creating a tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Indexing
first_element = tensor[0, 0]
print("First element:", first_element)

# Slicing
first_row = tensor[0, :]
print("First row:", first_row)

second_column = tensor[:, 1]
print("Second column:", second_column)

# Advanced Operations (Concatenation, Stacking)
# concatenation and stacking are useful for combining tensors.

# Creating tensors
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])

# Concatenation along the first dimension (rows)
concat_result = torch.cat((tensor_a, tensor_b), dim=0)
print("Concatenated along rows:", concat_result)

# Concatenation along the second dimension (columns)
concat_result = torch.cat((tensor_a, tensor_b), dim=1)
print("Concatenated along columns:", concat_result)

# Stacking (creates a new dimension)
stacked_result = torch.stack((tensor_a, tensor_b), dim=0)
print("Stacked along new dimension:", stacked_result)