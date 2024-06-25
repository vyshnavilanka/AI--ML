#Day10(1)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Load the Balance scale dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
balance_data = pd.read_csv(url, names=['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'])

# Consider only 'L' and 'R' classes and drop 'B'
balance_data = balance_data[balance_data['Class'] != 'B']

# Map 'L' and 'R' classes to 0 and 1 respectively
balance_data['Class'] = balance_data['Class'].map({'L': 0, 'R': 1})

# Separate features and target
X = balance_data.drop('Class', axis=1).values
y = balance_data['Class'].values

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

# Instantiate the model
input_size = X_train.shape[1]
model = LogisticRegression(input_size)
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))  # y_train needs to be reshaped to match output shape

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# Convert model predictions to binary (0 or 1)
with torch.no_grad():
    model.eval()
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()

# Calculate accuracy
accuracy = (y_pred_class == y_test.view(-1, 1)).float().mean()
print(f'Accuracy: {accuracy.item()*100:.2f}%')

#---------------------------------------------------------------------------------------------------------------------
#Day_10(2)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_data = pd.read_csv(url, sep=';')

# Binarize the target variable: good quality (quality >= 7) and bad quality (quality < 7)
wine_data['quality'] = wine_data['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Separate features and target
X = wine_data.drop('quality', axis=1).values
y = wine_data['quality'].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def evaluate_knn(k):
    # Define the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Experiment with different values of k
k_values = [3, 5, 7]
for k in k_values:
    accuracy = evaluate_knn(k)
    print(f'Accuracy for k={k}: {accuracy:.4f}')
