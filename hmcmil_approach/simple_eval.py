"""
Simple Evaluation - EXACT same preprocessing as training
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from data_preprocessing import SisFallPreprocessor
from model_hmcmil import HMCMIL

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load data - EXACT same as training
    print("\n📂 Loading data...")
    preprocessor = SisFallPreprocessor(
        data_dir='../SisFall_dataset',
        window_size=500,
        overlap=0.5,
        max_adl_per_subject=800
    )
    
    X, y, subjects = preprocessor.load_all_data()
    X, y, subjects = preprocessor.balance_dataset(X, y, subjects)
    
    # Split - EXACT same as training
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )
    
    # Normalize - EXACT same as training (THIS IS CRITICAL!)
    print("\n📊 Normalizing...")
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # CORRECT: flatten each sample
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    print(f"  Test: {X_test_norm.shape}")
    
    # Create dataset
    test_dataset = SimpleDataset(X_test_norm, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load model
    print("\n🏗️  Loading model...")
    model = HMCMIL(
        in_channels=9,
        timesteps=500,
        embed_dim=128,
        num_heads=8,
        per_scale_layers=6,
        fusion_layers=4,
        dropout=0.2,
        n_wavelets=8,
        projection_dim=128
    ).to(device)
    
    checkpoint = torch.load('results/best_hmcmil.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✅ Checkpoint loaded (epoch {checkpoint['epoch']})")
    print(f"  Stored accuracy: {checkpoint['best_acc']*100:.2f}%")
    
    # Evaluate
    print("\n📊 Evaluating...")
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            logits, _ = model(data, return_features=True)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds).astype(int)
    all_targets = np.array(all_targets).astype(int)
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    print("\n" + "="*60)
    print("🎯 RESULTS WITH CORRECT PREPROCESSING")
    print("="*60)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*60)
    
    if acc >= 0.950:
        print("\n✅ Correct! Matches the stored 95.4% accuracy!")
    else:
        print(f"\n⚠️  Still mismatch. Expected ~95%, got {acc*100:.2f}%")

if __name__ == '__main__':
    main()










