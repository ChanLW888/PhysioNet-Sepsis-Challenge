import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Load the clean dataset
with open("clean_dataset.pkl", "rb") as f:
    data = pickle.load(f)

train_patients = data["train_patients"]
val_patients = data["val_patients"]
train_labels = data["train_labels"]
val_labels = data["val_labels"]
means = data["means"]
stds = data["stds"]

print("Train patients:", len(train_patients))
print("Validation patients:", len(val_patients))


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

train_ds = PatientDataset(train_patients, train_labels)
val_ds = PatientDataset(val_patients, val_labels)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)


for i, (p, l) in enumerate(zip(train_patients, train_labels)):
    if np.isnan(p).any():
        print(f"=== Patient {i} has NaNs ===")
        df_debug = pd.DataFrame(p)
        df_debug["SepsisLabel"] = l

        # Show which columns are problematic
        print("NaN counts per column:\n", df_debug.isna().sum())

        # Show first 10 timesteps
        print("First 10 rows:\n", df_debug.head(10))

        # If you want the entire patient history:
        # print(df_debug.to_string())

        break  # stop after first problematic patient

class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)  # *2 for bidirectional
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = BiGRUModel(input_size=train_patients[0].shape[1], hidden_size=256, num_layers=3, output_size=1, dropout= 0.3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    epoch_loss = 0.0
    
    # Training loop
    for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X).squeeze(-1)
        target = y.float()
        
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze(-1)
            target = y.float()
            
            loss = criterion(outputs, target)
            val_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int).ravel()
            actual = target.cpu().numpy().astype(int).ravel()
            
            all_preds.extend(preds)
            all_targets.extend(actual)
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Compute metrics
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_targets, all_preds)
    
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        
torch.save(model.state_dict(), "rnn_model.pth")

