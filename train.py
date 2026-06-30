import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
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

# Flatten all patient label sequences into one long array
all_labels_flat = np.concatenate(train_labels)

num_pos = (all_labels_flat == 1).sum()
num_neg = (all_labels_flat == 0).sum()

pos_weight_value = num_neg / num_pos
pos_weight_value = 25
print("pos_weight =", pos_weight_value)

print("pos_weight =", pos_weight_value)
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

def collate_fn(batch):
    Xs, ys = zip(*batch)
    Xs_padded = pad_sequence(Xs, batch_first=True)
    ys_padded = pad_sequence(ys, batch_first=True)
    mask = torch.isnan(Xs_padded)  # True where missing
    Xs_padded = torch.nan_to_num(Xs_padded, nan=0.0)  # temporary fill
    return Xs_padded, ys_padded, mask

train_loader = DataLoader(train_ds, batch_size=64, collate_fn=collate_fn, shuffle=True)

val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate_fn)


for i, (p, l) in enumerate(zip(train_patients, train_labels)):
    if np.isnan(p).any():
        print(f"=== Patient {i} has NaNs ===")
        df_debug = pd.DataFrame(p)
        df_debug["SepsisLabel"] = l

        # Show which columns are problematic
        print("NaN counts per column:\n", df_debug.isna().sum())

        # Show the entire patient history with headers
        print(df_debug.to_string())   

        break  # stop after first problematic patient

class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        # One trainable value per feature
        self.missing_embed = nn.Parameter(torch.randn(input_size))
        self.gru = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x, mask):
        # mask: [batch, seq_len, features]
        # broadcast missing_embed across batch and seq_len
        x = torch.where(mask, self.missing_embed, x)
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = BiGRUModel(input_size=train_patients[0].shape[1], hidden_size=256, num_layers=1, output_size=1, dropout= 0.5).to(device)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_value]).to(device)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
writer = SummaryWriter()

best_f1 = 0.0
best_auc = 0.0
best_val = 10000000
best_epoch = -1

for epoch in range(50):
    model.train()
    epoch_loss = 0.0
    
    # Training loop
    for X, y, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
        X, y, mask = X.to(device), y.to(device), mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(X, mask).squeeze(-1)   # ✅ pass mask
        target = y.float()
        
        loss = criterion(outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y, mask in val_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            outputs = model(X, mask).squeeze(-1)   # ✅ pass mask
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
          f"F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Log to TensorBoard
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Metrics/precision", precision, epoch)
    writer.add_scalar("Metrics/recall", recall, epoch)
    writer.add_scalar("Metrics/f1", f1, epoch)
    writer.add_scalar("Metrics/roc_auc", roc_auc, epoch)
    
    # Save best model (based on F1 or ROC-AUC)
    if f1 > best_f1:
        best_val = avg_val_loss
        best_f1 = f1
        best_auc = roc_auc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Saved new best model at epoch {best_epoch} with F1={best_f1:.4f}, AUC={best_auc:.4f}")
    
    scheduler.step()

writer.close()
        
torch.save(model.state_dict(), "rnn_model.pth")

