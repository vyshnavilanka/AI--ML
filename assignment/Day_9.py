#Day_9(1)

from sklearn.datasets import load_boston
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the dataset
boston = load_boston()
X = boston.data
y = boston.target

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # reshape to column vector


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# Instantiate the model
model = LinearRegression(X.shape[1], 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# Convert tensors to numpy arrays
X_np = X.detach().numpy()
y_np = y.detach().numpy()

# Prediction
model.eval()
with torch.no_grad():
    predicted = model(X).detach().numpy()

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(y_np, predicted, c='r', label='Predictions')
plt.plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], 'b--', lw=2, label='Actual')
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------------------------
#Day_9(2)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # reshape to column vector
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Add polynomial features up to degree 3
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train.numpy())
X_test_poly = poly.transform(X_test.numpy())

# Convert back to PyTorch tensors
X_train_poly = torch.tensor(X_train_poly, dtype=torch.float32)
X_test_poly = torch.tensor(X_test_poly, dtype=torch.float32)


class PolynomialRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolynomialRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# Instantiate the model
model_poly = PolynomialRegression(X_train_poly.shape[1], 1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model_poly.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model_poly(X_train_poly)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # Define linear regression model
        model_linear = nn.Linear(X_train.shape[1], 1)

        # Define loss function and optimizer
        criterion_linear = nn.MSELoss()
        optimizer_linear = optim.SGD(model_linear.parameters(), lr=0.001)

        # Training loop
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model_linear(X_train)
            loss = criterion_linear(outputs, y_train)

            # Backward pass and optimization
            optimizer_linear.zero_grad()
            loss.backward()
            optimizer_linear.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation on test set
with torch.no_grad():
    model_poly.eval()
    y_pred_poly = model_poly(X_test_poly)
    mse_poly = mean_squared_error(y_test.numpy(), y_pred_poly.numpy())
    print(f'Polynomial Regression Test MSE: {mse_poly:.4f}')

    model_linear.eval()
    y_pred_linear = model_linear(X_test)
    mse_linear = mean_squared_error(y_test.numpy(), y_pred_linear.numpy())
    print(f'Linear Regression Test MSE: {mse_linear:.4f}')

#-----------------------------------------------------------------------------------------------------------------
#Day_9(3)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define Ridge regression model
ridge = Ridge()

# Define grid of alpha values (regularization parameter)
alphas = np.logspace(-3, 3, 20)  # 20 alphas from 10^-3 to 10^3

# Perform GridSearchCV to find the best alpha
param_grid = {'alpha': alphas}
ridge_cv = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

# Get the best alpha
best_alpha_ridge = ridge_cv.best_params_['alpha']
print(f'Best alpha for Ridge regression: {best_alpha_ridge:.4f}')

# Train Ridge regression model with best alpha
ridge_best = Ridge(alpha=best_alpha_ridge)
ridge_best.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_ridge = ridge_best.predict(X_test_scaled)

# Calculate MSE
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f'Ridge Regression Test MSE: {mse_ridge:.4f}')
from sklearn.linear_model import Lasso

# Define Lasso regression model
lasso = Lasso(max_iter=10000)

# Perform GridSearchCV to find the best alpha
lasso_cv = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train_scaled, y_train)

# Get the best alpha
best_alpha_lasso = lasso_cv.best_params_['alpha']
print(f'Best alpha for Lasso regression: {best_alpha_lasso:.4f}')

# Train Lasso regression model with best alpha
lasso_best = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_best.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_lasso = lasso_best.predict(X_test_scaled)

# Calculate MSE
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f'Lasso Regression Test MSE: {mse_lasso:.4f}')
# Train standard linear regression model
from sklearn.linear_model import LinearRegression

model_linear = LinearRegression()
model_linear.fit(X_train_scaled, y_train)
y_pred_linear = model_linear.predict(X_test_scaled)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f'Linear Regression Test MSE: {mse_linear:.4f}')

# Compare MSE of Ridge, Lasso, and Linear regression
print(f'Ridge Regression Test MSE: {mse_ridge:.4f}')
print(f'Lasso Regression Test MSE: {mse_lasso:.4f}')
print(f'Linear Regression Test MSE: {mse_linear:.4f}')

# Interpretation
print("\nInterpretation:")
print("- Ridge regression typically performs better than simple linear regression when there is multicollinearity among the features.")
print("- Lasso regression is useful for feature selection because it tends to shrink coefficients of less important features to zero.")
print("- In this case, we observe that Ridge regression has slightly lower MSE than Lasso regression and linear regression.")
