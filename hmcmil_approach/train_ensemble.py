"""
Ensemble Training for 98%+ Accuracy
====================================

Strategy: Train 5 HMC-MIL models with:
- Different random seeds (diversity)
- Slightly different augmentation strategies
- Same architecture (maintains novelty!)

Expected: 95.62% (single) → 97-98%+ (ensemble)

Research-backed: Ensemble typically adds 1-3% accuracy
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from data_preprocessing import SisFallPreprocessor
from data_augmentation import TimeSeriesAugmentation
from model_hmcmil import HMCMIL

# ============================================================
# Loss (same as original training)
# ============================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        
        return loss

class CombinedLoss(nn.Module):
    """Focal + SupCon Loss"""
    def __init__(self, alpha=0.7, gamma=2.5, temperature=0.07, supcon_weight=0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.supcon_weight = supcon_weight
        self.supcon = SupConLoss(temperature=temperature)
    
    def forward(self, logits, features, targets):
        targets = targets.view(-1).float()
        logits = logits.view(-1)
        
        # Focal loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, pt, 1 - pt)
        focal_weight = (1 - pt) ** self.gamma
        class_weight = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
        focal_loss = (focal_weight * class_weight * bce_loss).mean()
        
        # SupCon loss
        contrastive_loss = self.supcon(features, targets.long())
        
        total_loss = focal_loss + self.supcon_weight * contrastive_loss
        
        return total_loss, focal_loss, contrastive_loss

# ============================================================
# Dataset
# ============================================================

class AugmentedDataset(Dataset):
    def __init__(self, X, y, augment=False, aug_prob=0.5):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.augmenter = TimeSeriesAugmentation(aug_prob=aug_prob) if augment else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment and self.augmenter is not None:
            x_np = x.numpy()
            x_aug = self.augmenter(x_np)
            x = torch.FloatTensor(x_aug)
        
        return x, y

# ============================================================
# Ensemble Trainer
# ============================================================

def train_single_model(model_id, seed, aug_prob, X_train, y_train, X_val, y_val, X_test, y_test, device):
    """Train a single model in the ensemble"""
    
    print(f"\n{'='*60}")
    print(f"🔧 Training Model {model_id+1}/5 (Seed={seed})")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create datasets
    train_dataset = AugmentedDataset(X_train, y_train, augment=True, aug_prob=aug_prob)
    val_dataset = AugmentedDataset(X_val, y_val, augment=False)
    test_dataset = AugmentedDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
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
    
    # Loss, optimizer, scheduler
    criterion = CombinedLoss(alpha=0.7, gamma=2.5, temperature=0.07, supcon_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # AMP
    scaler = torch.amp.GradScaler('cuda')
    
    # Training
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    max_patience = 15
    
    for epoch in range(1, 51):  # Max 50 epochs
        # Train
        model.train()
        train_loss = 0
        
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits, features = model(data, return_features=True)
                loss, focal, supcon = criterion(logits, features, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device)
                targets = targets.to(device)
                
                with torch.amp.autocast('cuda'):
                    logits, _ = model(data, return_features=True)
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                val_preds.extend(preds.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
        
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Test
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device)
                targets = targets.to(device)
                
                with torch.amp.autocast('cuda'):
                    logits, _ = model(data, return_features=True)
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                test_preds.extend(preds.cpu().numpy().flatten())
                test_targets.extend(targets.cpu().numpy().flatten())
        
        test_acc = accuracy_score(test_targets, test_preds)
        
        scheduler.step()
        
        print(f"Epoch {epoch:2d}: Val={val_acc*100:.2f}% | Test={test_acc*100:.2f}%", end="")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience = 0
            
            # Save model
            os.makedirs('ensemble_models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
                'seed': seed,
                'model_id': model_id
            }, f'ensemble_models/model_{model_id}_seed{seed}.pth')
            
            print(f" ✅ NEW BEST!")
        else:
            patience += 1
            print(f" ({patience}/{max_patience})")
        
        if patience >= max_patience:
            print(f"\n⏸️  Early stopping at epoch {epoch}")
            break
    
    print(f"\n✅ Model {model_id+1} Complete: Best Test Acc = {best_test_acc*100:.2f}%")
    
    return best_test_acc

# ============================================================
# Main
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    print("\n" + "="*60)
    print("🎯 ENSEMBLE TRAINING FOR 98%+ ACCURACY")
    print("="*60)
    print("Strategy: Train 5 HMC-MIL models with different seeds")
    print("Expected: 95.62% (single) → 97-98%+ (ensemble)")
    print("Estimated time: 3-4 hours (all 5 models)")
    print("="*60 + "\n")
    
    # Load data
    print("📂 Loading SisFall dataset...")
    preprocessor = SisFallPreprocessor(
        data_dir='../SisFall_dataset',
        window_size=500,
        overlap=0.5,
        max_adl_per_subject=800
    )
    
    X, y, subjects = preprocessor.load_all_data()
    X, y, subjects = preprocessor.balance_dataset(X, y, subjects)
    
    # Split - EXACT same as original
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )
    
    # Normalize - CORRECT way
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    print(f"  Train: {X_train_norm.shape}")
    print(f"  Val: {X_val_norm.shape}")
    print(f"  Test: {X_test_norm.shape}")
    
    # Train 5 models
    seeds = [42, 123, 456, 789, 1024]  # Different seeds for diversity
    aug_probs = [0.5, 0.6, 0.55, 0.65, 0.6]  # Slight variation in augmentation
    
    test_accs = []
    
    for i, (seed, aug_prob) in enumerate(zip(seeds, aug_probs)):
        acc = train_single_model(
            model_id=i,
            seed=seed,
            aug_prob=aug_prob,
            X_train=X_train_norm,
            y_train=y_train,
            X_val=X_val_norm,
            y_val=y_val,
            X_test=X_test_norm,
            y_test=y_test,
            device=device
        )
        test_accs.append(acc)
    
    print("\n" + "="*60)
    print("🎉 ALL MODELS TRAINED!")
    print("="*60)
    print("\nIndividual Model Accuracies:")
    for i, acc in enumerate(test_accs):
        print(f"  Model {i+1}: {acc*100:.2f}%")
    print(f"\n  Mean: {np.mean(test_accs)*100:.2f}%")
    print(f"  Std:  {np.std(test_accs)*100:.2f}%")
    print("="*60)
    print("\n✅ Run evaluate_ensemble.py to get final ensemble accuracy!")

if __name__ == '__main__':
    main()








