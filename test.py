# validate.py
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# 1. Load dataset
with open("clean_dataset.pkl", "rb") as f:
    data = pickle.load(f)

val_patients = data["val_patients"]
val_labels = data["val_labels"]

class PatientDataset(Dataset):
    def __init__(self, patients, labels):
        self.patients = patients
        self.labels = labels
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.patients[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y

val_ds = PatientDataset(val_patients, val_labels)
val_loader = DataLoader(val_ds, batch_size=1)

# 2. Define model (same architecture as training)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)  # *2 for bidirectional
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

# 3. Load trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel(input_size=val_patients[0].shape[1],
                 hidden_size=512, num_layers=4, output_size=1, dropout=0.5).to(device)
model.load_state_dict(torch.load("rnn_model.pth", map_location=device))
model.eval()
i=0
with torch.no_grad():
    for i, (X, y) in enumerate(val_loader):
        X, y = X.to(device), y.to(device)
        outputs = model(X).squeeze(-1)

        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int).ravel()
        actual = y.cpu().numpy().astype(int).ravel()

        # Skip patients with no sepsis events
        if actual.sum() == 0:
            continue

        df_debug = pd.DataFrame({
            "ICULOS": np.arange(1, len(actual)+1),
            "Actual_SepsisLabel": actual,
            "Predicted_SepsisLabel": preds
        })

        print(f"\n=== Patient {i} (Sepsis present) ===")
        print(df_debug.to_string())
        print(i)
        # Stop after showing a few sepsis patients
        if i >= 50:
            break
