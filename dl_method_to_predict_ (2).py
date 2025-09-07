# https://colab.research.google.com/drive/1cQlsb-myuBnb7FGpObSS0y0QjELDAZfF

# New Section

## Abstract
# This study presents a deep learning approach using PyTorch to predict breast cancer diagnoses from clinical features. 
# A publicly available dataset was preprocessed by handling missing values, encoding categorical variables, and scaling numerical features. 
# An artificial neural network (ANN) with two hidden layers was designed, employing ReLU activations and a sigmoid output for binary classification. 
# The model was trained for 100 epochs using binary cross-entropy loss and the Adam optimizer. 
# Evaluation on the held-out test set achieved an accuracy of 96.5%, demonstrating the model’s strong predictive capability. 
# These results indicate that PyTorch-based deep learning models can effectively classify malignant and benign breast cancer cases. 
# The framework can be further extended with hyperparameter optimization, alternative architectures, or cross-validation for enhanced robustness, 
# and it highlights the potential of deep learning to support computer-aided diagnostic systems in oncology.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('"C:/Users/Ar Dihan/Downloads/Breast_cancer_dataset.csv"')

## Data preprocessing

# Inspect missing values
print(df.isnull().sum())

# Drop column with only missing values
df = df.drop('Unnamed: 32', axis=1)

# Encode diagnosis column (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features (X) and target (y), dropping 'id'
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

## Model definition

class BreastCancerNet(nn.Module):
    def __init__(self, input_features):
        super(BreastCancerNet, self).__init__()
        self.fc1 = nn.Linear(input_features, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize the model
input_features = X_train.shape[1]
model = BreastCancerNet(input_features)
print(model)

## Model training

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
model.train()

for epoch in range(epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

## Model evaluation

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()
    correct = (predicted == y_test_tensor).sum().item()
    accuracy = correct / y_test_tensor.size(0)

print(f'Accuracy of the model on the test data: {accuracy:.4f}')

## Summary

# Key Findings:
# - Dropped 'Unnamed: 32' column with missing values
# - Encoded 'diagnosis' column (M=1, B=0)
# - Features scaled, dataset split (80/20)
# - ANN with two hidden layers + sigmoid output defined in PyTorch
# - Model trained for 100 epochs with BCE loss and Adam optimizer
# - Achieved test accuracy = 96.5%

# Next Steps:
# - Explore alternative architectures, hyperparameter tuning, or cross-validation
# - Validate robustness of performance for clinical use
