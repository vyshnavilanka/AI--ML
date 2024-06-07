import numpy as np

# Generate random 3x3 matrices A and B with integer values between 1 and 10
A = np.random.randint(1, 11, size=(3, 3))
B = np.random.randint(1, 11, size=(3, 3))

# Compute the sum of A and B
sum_AB = A + B
print("Sum of A and B:")
print(sum_AB)

# Compute the difference between A and B
diff_AB = A - B
print("\nDifference between A and B:")
print(diff_AB)

# Compute the element-wise product of A and B
elementwise_product_AB = A * B
print("\nElement-wise product of A and B:")
print(elementwise_product_AB)

# Compute the matrix product of A and B
matrix_product_AB = np.dot(A, B)
print("\nMatrix product of A and B:")
print(matrix_product_AB)

# Compute the transpose of matrix A
A_transpose = A.T
print("\nTranspose of matrix A:")
print(A_transpose)

# Compute the determinant of matrix A
determinant_A = np.linalg.det(A)
print("\nDeterminant of matrix A:")
print(determinant_A)