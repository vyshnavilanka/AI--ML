# First derivative
f_prime = derivative(f, 1, dx=1e-6)
print("First derivative of f(x) at x=1:", f_prime)

# Second derivative
f_double_prime = derivative(f, 1, n=2, dx=1e-6)
print("Second derivative of f(x) at x=1:", f_double_prime)