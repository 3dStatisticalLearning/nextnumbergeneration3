#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Load your dataset
data = pd.read_excel("C:/Users/Admin/Documents/Mappe_nagelneu_new.xlsx", header=None)

# Assuming the dataset has n columns and m rows
n = data.shape[1]  # Number of columns
m = data.shape[0]  # Number of rows

# Prepare input (X) and target (y) data
X = data.iloc[:-1, :].values  # All rows except the last one
y = data.iloc[1:, :].values    # All rows except the first one

# Normalize target values to be within [0, num_classes - 1]
unique_values, y_mapped = np.unique(y, return_inverse=True)  # Ensure correct mapping
y = y_mapped.reshape(y.shape)  # Reshape to original shape
num_classes = len(unique_values)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Create a custom dataset
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_dataset = SequenceDataset(X[:train_size], y[:train_size])
test_dataset = SequenceDataset(X[train_size:], y[train_size:])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim * input_dim)  # Output for each column

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, input_dim, hidden_dim)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)    # Remove sequence dimension
        x = self.fc(x)      # Shape: (batch_size, input_dim * num_classes)
        x = x.view(-1, n, num_classes)  # Reshape to (batch_size, input_dim, num_classes)
        return x

# Hyperparameters
input_dim = n  # Number of columns
output_dim = num_classes  # Number of unique classes in the target data
num_heads = 4
num_layers = 2
hidden_dim = 64
learning_rate = 0.001
num_epochs = 20

# Initialize the model, loss function, and optimizer
model = TransformerModel(input_dim, output_dim, num_heads, num_layers, hidden_dim)
criterion = nn.CrossEntropyLoss()  # Expecting shape (batch_size, input_dim, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)  # Shape: (batch_size, input_dim, num_classes)
        
        # Compute loss for each column independently
        loss = criterion(outputs.permute(0, 2, 1), y_batch)  # Reshaped for CrossEntropyLoss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Save the model and unique values
torch.save(model.state_dict(), "model.pth")
np.save("unique_values.npy", unique_values)
print("Model and unique values saved successfully.")


# In[ ]:




