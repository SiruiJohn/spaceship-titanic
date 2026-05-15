
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS = 5
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3

print(f"Using device: {DEVICE}")

# --- Load Data ---
print("Loading preprocessed data...")
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "results" / "preds" / "dl"
MODEL_DIR = ROOT / "results" / "models" / "dl"
PRED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train_processed_dl.csv")
test_df = pd.read_csv(DATA_DIR / "test_processed_dl.csv")

# --- Feature Info ---
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
numerical_features = [
    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
    'Cabin_Num', 'GroupSize', 'IsAlone', 'TotalSpend', 'AnySpend', 'CryoSleepSpendConflict'
]

# Get embedding sizes
emb_dims = [train_df[col].nunique() for col in categorical_features]
emb_sizes = [(n_cat, min(50, (n_cat + 1) // 2)) for n_cat in emb_dims]

# --- PyTorch Dataset & DataLoader ---
def get_tensors(df):
    cats = torch.tensor(df[categorical_features].values, dtype=torch.long)
    nums = torch.tensor(df[numerical_features].values, dtype=torch.float)
    if 'Transported' in df.columns:
        target = torch.tensor(df['Transported'].values, dtype=torch.float).unsqueeze(1)
        return cats, nums, target
    return cats, nums

# --- Model Definition ---
class SpaceshipMLP(nn.Module):
    def __init__(self, emb_sizes, n_numeric):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n_cat, n_dim) for n_cat, n_dim in emb_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_numeric = n_emb, n_numeric
        
        self.bn_cont = nn.BatchNorm1d(self.n_numeric)
        
        self.seq = nn.Sequential(
            nn.Linear(self.n_emb + self.n_numeric, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        return self.seq(x)

# --- Training Loop ---
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for cats, nums, y in dataloader:
        cats, nums, y = cats.to(DEVICE), nums.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(cats, nums)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

def validate(model, dataloader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for cats, nums, _ in dataloader:
            cats, nums = cats.to(DEVICE), nums.to(DEVICE)
            outputs = model(cats, nums)
            all_preds.append(outputs.cpu().numpy())
    return np.concatenate(all_preds)

# --- Main K-Fold Training ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), 1))
test_preds = np.zeros((len(test_df), 1))

X_cats, X_nums, y_tensor = get_tensors(train_df)
X_test_cats, X_test_nums = get_tensors(test_df)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_nums, y_tensor.squeeze())):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    
    # Data for this fold
    train_dataset = TensorDataset(X_cats[train_idx], X_nums[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_cats[val_idx], X_nums[val_idx], y_tensor[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # Model, optimizer, criterion
    model = SpaceshipMLP(emb_sizes, len(numerical_features)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, optimizer, criterion)
        
        # Simple validation loss for scheduler and early stopping
        val_preds_for_loss = validate(model, val_loader)
        val_loss = criterion(torch.tensor(val_preds_for_loss), y_tensor[val_idx].cpu()).item()
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / f"best_model_fold_{fold}.pth")
        else:
            patience_counter += 1
            if patience_counter >= 7: # Early stopping
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and predict on validation and test sets
    model.load_state_dict(torch.load(MODEL_DIR / f"best_model_fold_{fold}.pth"))
    oof_preds[val_idx] = validate(model, val_loader)
    
    test_loader = DataLoader(TensorDataset(X_test_cats, X_test_nums, torch.zeros(len(test_df), 1)), batch_size=BATCH_SIZE * 2)
    test_preds += validate(model, test_loader) / N_SPLITS

# --- Save Predictions ---
print("Saving OOF and test predictions...")
np.save(PRED_DIR / "oof_preds_dl.npy", oof_preds)
np.save(PRED_DIR / "test_preds_dl.npy", test_preds)

# Evaluate OOF predictions
oof_accuracy = accuracy_score(y_tensor.squeeze().numpy(), (oof_preds.squeeze() >= 0.5).astype(int))
print(f"Overall OOF Accuracy: {oof_accuracy:.4f}")
