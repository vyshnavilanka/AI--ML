import numpy as np
from scipy import linalg

# Coefficient matrix
A = np.array([[2, 3],
              [3, 4]])

# Constant terms vector
B = np.array([8, 11])

# Solve the linear system
x = linalg.solve(A, B)

print("Solution:")
print("x =", x[0])
print("y =", x[1])