import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_excel("C:/Users/Admin/Documents/Mappe_nagelneu_new.xlsx", header=None)

# Dataset dimensions
n = data.shape[1]  # Number of columns
m = data.shape[0]  # Number of rows

# Prepare input (X) and target (y)
X = data.iloc[:-1, :].values  # All rows except last
y = data.iloc[1:, :].values   # All rows except first

# Normalize target values
unique_values, y_mapped = np.unique(y, return_inverse=True)
y = y_mapped.reshape(y.shape)
num_classes = len(unique_values)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Define Dataset
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split Data
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_dataset = SequenceDataset(X[:train_size], y[:train_size])
test_dataset = SequenceDataset(X[train_size:], y[train_size:])

# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim * input_dim)

    def forward(self, x):
        x = self.embedding(x)  
        x = x.unsqueeze(1)  
        x = self.transformer(x)
        x = x.squeeze(1)    
        x = self.fc(x)      
        x = x.view(-1, input_dim, num_classes)
        return x

# Hyperparameters
input_dim = n
output_dim = num_classes
num_heads = 4
num_layers = 2
hidden_dim = 64
learning_rate = 0.001
num_epochs = 20

# Initialize Model
model = TransformerModel(input_dim, output_dim, num_heads, num_layers, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.permute(0, 2, 1), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Save model and unique values
torch.save(model.state_dict(), "model.pth")
np.save("unique_values.npy", unique_values)
print("Model and unique values saved successfully.")
