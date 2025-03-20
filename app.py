from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from io import BytesIO

# Initialize FastAPI
app = FastAPI()

# Model paths
MODEL_PATH = "model.pth"
UNIQUE_VALUES_PATH = "unique_values.npy"

# Load Unique Values
if os.path.exists(UNIQUE_VALUES_PATH):
    unique_values = np.load(UNIQUE_VALUES_PATH)
    num_classes = len(unique_values)
else:
    raise RuntimeError("Unique values file not found.")

# Model Definition
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

# Load Model
if os.path.exists(MODEL_PATH):
    input_dim = 20  # Adjust as needed
    num_heads = 4
    num_layers = 2
    hidden_dim = 64

    model = TransformerModel(input_dim, num_classes, num_heads, num_layers, hidden_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
else:
    raise RuntimeError("Model file not found.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load Excel file
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents), header=None)
        dataset = df.values

        # Extract last row
        last_row = dataset[-1, :].astype(np.float32)
        last_row_tensor = torch.tensor(last_row, dtype=torch.float32).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(last_row_tensor)
            preds = torch.argmax(outputs, dim=2).numpy()

        # Convert to original values
        index_to_value = {idx: value for idx, value in enumerate(unique_values)}
        next_row = np.vectorize(index_to_value.get)(preds)

        return next_row.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Transformer Model Prediction API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
