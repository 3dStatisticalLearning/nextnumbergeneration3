#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

# Initialize FastAPI app
app = FastAPI()

# Constants
MODEL_PATH = "model.pth"
UNIQUE_VALUES_PATH = "unique_values.npy"

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
        x = x.view(-1, input_dim, output_dim)  # Reshape to (batch_size, input_dim, num_classes)
        return x

# Load pre-trained model and unique values
if os.path.exists(MODEL_PATH) and os.path.exists(UNIQUE_VALUES_PATH):
    unique_values = np.load(UNIQUE_VALUES_PATH)
    num_classes = len(unique_values)
    input_dim = 20  # Adjust based on your dataset
    num_heads = 4
    num_layers = 2
    hidden_dim = 64

    model = TransformerModel(input_dim, num_classes, num_heads, num_layers, hidden_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set the model to evaluation mode
else:
    raise RuntimeError("Model or unique values file not found. Train and save the model first.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Upload an Excel file for prediction.
    """
    try:
        # Load and process the Excel file
        contents = await file.read()
        df = pd.read_excel(contents, header=None)
        dataset = df.values

        # Use the last row as input for prediction
        last_row = dataset[-1, :].astype(np.float32)
        last_row_tensor = torch.tensor(last_row, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            outputs = model(last_row_tensor)  # Shape: (batch_size, input_dim, num_classes)
            preds = torch.argmax(outputs, dim=2).numpy()  # Get predicted class indices for each column

        # Map predicted indices back to original values
        index_to_value = {idx: value for idx, value in enumerate(unique_values)}
        next_row = np.vectorize(index_to_value.get)(preds)

        return {"predictions": next_row.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Basic health check for the API.
    """
    return {"message": "Transformer Model Prediction API is running."}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

