"""
HMC-MIL Training Script with Transfer Learning and Supervised Contrastive Loss

Training Strategy (3 Phases):
    Phase 1 (20 epochs): Train fine/coarse scales, freeze medium (TimeMIL v2)
    Phase 2 (60 epochs): Unfreeze all, joint training
    Phase 3 (20 epochs): Fine-tuning with stronger SupCon

Target: 98%+ test accuracy
"""

import os
import sys

# Add parent directory to path to import shared modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from model_hmcmil import HMCMIL, count_parameters
from data_preprocessing import SisFallPreprocessor
from data_augmentation import TimeSeriesAugmentation
from utils import (
    compute_metrics,
    plot_training_curves, plot_confusion_matrix,
    plot_roc_curve, plot_pr_curve,
    print_classification_report, save_metrics_to_csv
)

import warnings
warnings.filterwarnings('ignore')


class AugmentedDataset(Dataset):
    """Dataset with on-the-fly augmentation"""
    def __init__(self, X, y, augment=False, aug_prob=0.6):
        self.X = X
        self.y = y
        self.augment = augment
        self.augmenter = TimeSeriesAugmentation(aug_prob=aug_prob) if augment else None
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]
        
        if self.augment and self.augmenter is not None:
            x = self.augmenter(x)
        
        return torch.FloatTensor(x), torch.FloatTensor([y])


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    
    Paper: "Supervised Contrastive Learning" (NeurIPS 2020)
    
    Pull samples of same class together, push different classes apart
    """
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
    
    def forward(self, features, labels):
        """
        Args:
            features: (batch, projection_dim) - normalized features
            labels: (batch,) - class labels
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Flatten labels
        labels = labels.view(-1)
        
        # Mask: positive pairs (same class)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Mask out self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_sum
        
        loss = mean_log_prob_pos.mean()
        return loss


class HMCMILLoss(nn.Module):
    """
    Combined loss: Focal Loss (classification) + SupCon Loss (representation)
    """
    def __init__(self, alpha=0.7, gamma=2.5, temperature=0.07, 
                 supcon_weight=0.3, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.supcon_weight = supcon_weight
        self.label_smoothing = label_smoothing
        
        self.supcon_loss = SupConLoss(temperature=temperature)
    
    def forward(self, logits, features, targets):
        """
        Args:
            logits: (batch, 1) - classification logits
            features: (batch, projection_dim) - normalized features
            targets: (batch, 1) - labels
        """
        targets = targets.view(-1).float()
        logits = logits.view(-1)
        
        # Focal Loss
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        
        pt = torch.sigmoid(logits)
        pt = torch.where(targets_smooth > 0.5, pt, 1 - pt)
        focal_weight = (1 - pt) ** self.gamma
        
        class_weight = torch.where(targets_smooth > 0.5, self.alpha, 1 - self.alpha)
        focal_loss = (focal_weight * class_weight * bce_loss).mean()
        
        # Supervised Contrastive Loss
        contrastive_loss = self.supcon_loss(features, targets)
        
        # Combined loss
        total_loss = focal_loss + self.supcon_weight * contrastive_loss
        
        return total_loss, focal_loss, contrastive_loss


class HMCMILTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 criterion, optimizer, scheduler, device, save_dir,
                 epochs=100, patience=25, grad_clip=1.0):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        
        self.use_amp = device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.counter = 0
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
        }
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_focal = 0
        total_contrastive = 0
        all_preds, all_labels, all_probs = [], [], []
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    logits, features = self.model(data, return_features=True)
                    loss, focal, contrastive = self.criterion(logits, features, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, features = self.model(data, return_features=True)
                loss, focal, contrastive = self.criterion(logits, features, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            total_focal += focal.item()
            total_contrastive += contrastive.item()
            
            probs = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
        
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        
        avg_loss = total_loss / len(self.train_loader)
        avg_focal = total_focal / len(self.train_loader)
        avg_contrastive = total_contrastive / len(self.train_loader)
        
        return avg_loss, avg_focal, avg_contrastive, metrics
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                
                logits, features = self.model(data, return_features=True)
                loss, _, _ = self.criterion(logits, features, target)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(target.cpu().numpy())
        
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        return total_loss / len(loader), metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def train(self, phase_name="Training"):
        print(f"\n{'='*160}")
        print(f"{phase_name}")
        print(f"{'='*160}")
        print(f"Device: {self.device} | Epochs: {self.epochs} | Patience: {self.patience}")
        print(f"{'='*160}")
        
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Train F1':>9} | "
              f"{'Val Loss':>10} | {'Val Acc':>9} | {'Val F1':>9} | {'Val AUC':>9}")
        print(f"{'='*160}")
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_focal, train_contrastive, train_metrics = self.train_epoch()
            val_loss, val_metrics, _, _, _ = self.validate(self.val_loader)
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['train_auc'].append(train_metrics['auc'])
            
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            print(f"{epoch:5d} | "
                  f"{train_loss:10.4f} | "
                  f"{train_metrics['accuracy']:9.4f} | "
                  f"{train_metrics['f1']:9.4f} | "
                  f"{val_loss:10.4f} | "
                  f"{val_metrics['accuracy']:9.4f} | "
                  f"{val_metrics['f1']:9.4f} | "
                  f"{val_metrics['auc']:9.4f}")
            
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_val_acc,
                    'best_f1': self.best_val_f1
                }, os.path.join(self.save_dir, 'best_hmcmil.pth'))
                
                print(f"       → ✓ BEST MODEL (Acc: {self.best_val_acc:.4f}, F1: {self.best_val_f1:.4f})")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"\n       → ⚠ EARLY STOPPING at epoch {epoch}")
                    break
        
        print(f"{'='*160}")
        print(f"✅ {phase_name} COMPLETED!")
        print(f"   Best Val Accuracy: {self.best_val_acc:.4f} at Epoch {self.best_epoch}")
        print(f"   Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*160}")
        
        return self.best_val_acc
    
    def evaluate_final(self):
        print(f"\n{'='*160}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'='*160}")
        
        checkpoint = torch.load(os.path.join(self.save_dir, 'best_hmcmil.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_metrics, y_true, y_pred, y_prob = self.validate(self.test_loader)
        
        print(f"\n[Test Results]")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"  Precision:   {test_metrics['precision']:.4f}")
        print(f"  Recall:      {test_metrics['recall']:.4f}")
        print(f"  F1 Score:    {test_metrics['f1']:.4f}")
        print(f"  Specificity: {test_metrics['specificity']:.4f}")
        print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
        print(f"  AUC:         {test_metrics['auc']:.4f}")
        
        print_classification_report(y_true, y_pred)
        
        # Generate visualizations
        print("\n[Generating Visualizations]")
        viz_dir = os.path.join(self.save_dir, 'visualizations')
        
        plot_training_curves(self.history, os.path.join(viz_dir, 'training_curves.png'))
        plot_confusion_matrix(y_true, y_pred, os.path.join(viz_dir, 'confusion_matrix.png'))
        plot_roc_curve(y_true, y_prob, os.path.join(viz_dir, 'roc_curve.png'))
        plot_pr_curve(y_true, y_prob, os.path.join(viz_dir, 'pr_curve.png'))
        
        save_metrics_to_csv(self.history, os.path.join(self.save_dir, 'training_history.csv'))
        
        print(f"\n✅ ALL RESULTS SAVED to {self.save_dir}")
        print(f"{'='*160}\n")
        
        return test_metrics['accuracy']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'results'
    timemil_v2_checkpoint = '../timemil_v2_archive/timemil_v2_improved/best_timemil_v2.pth'
    
    print(f"\n[Device] {device}")
    
    # === 1. LOAD DATA ===
    print("\n[1/5] Loading ALL 38 Subjects...")
    preprocessor = SisFallPreprocessor(
        data_dir='../SisFall_dataset',
        window_size=500,
        overlap=0.5,
        max_adl_per_subject=800
    )
    
    X, y, subjects = preprocessor.load_all_data()
    X, y, subjects = preprocessor.balance_dataset(X, y, subjects)
    
    print(f"\n[Using ALL 38 subjects]")
    print(f"  Total samples: {len(X)}")
    print(f"  Falls: {y.sum()}/{len(y)}")
    
    # Standard train/val/test split (70/15/15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    print(f"\n[Data Split - 70/15/15]")
    print(f"  Train: {X_train_norm.shape}, Falls: {y_train.sum()}/{len(y_train)}")
    print(f"  Val: {X_val_norm.shape}, Falls: {y_val.sum()}/{len(y_val)}")
    print(f"  Test: {X_test_norm.shape}, Falls: {y_test.sum()}/{len(y_test)}")
    
    # === 2. CREATE DATALOADERS ===
    print(f"\n[2/5] Creating DataLoaders...")
    train_dataset = AugmentedDataset(X_train_norm, y_train, augment=True, aug_prob=0.6)
    val_dataset = AugmentedDataset(X_val_norm, y_val, augment=False)
    test_dataset = AugmentedDataset(X_test_norm, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    # === 3. CREATE HMC-MIL MODEL ===
    print(f"\n[3/5] Creating HMC-MIL Model with Transfer Learning...")
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
    
    # Load TimeMIL v2 weights into medium scale
    model.load_timemil_v2_weights(timemil_v2_checkpoint)
    
    n_params = count_parameters(model)
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # === 4. PHASED TRAINING STRATEGY ===
    print(f"\n[4/5] Setting up Phased Training Strategy...")
    print(f"   Phase 1: Train fine/coarse (medium frozen) - 20 epochs")
    print(f"   Phase 2: Joint training (all scales) - 60 epochs")
    print(f"   Phase 3: Fine-tuning with stronger SupCon - 20 epochs")
    print(f"   Total: 100 epochs, Expected: 98%+ accuracy! 🚀")
    
    # === 5. PHASE 1: Train Fine/Coarse, Freeze Medium ===
    print(f"\n[5/5] Starting PHASE 1: Warm-up Fine/Coarse Scales...")
    
    # Freeze medium scale
    for param in model.medium_scale.parameters():
        param.requires_grad = False
    
    criterion = HMCMILLoss(alpha=0.7, gamma=2.5, temperature=0.07, supcon_weight=0.2)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                           lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    trainer = HMCMILTrainer(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler, device, save_dir,
        epochs=20, patience=15, grad_clip=1.0
    )
    
    phase1_acc = trainer.train("PHASE 1: Warm-up Fine/Coarse (Medium Frozen)")
    
    # === PHASE 2: Unfreeze All, Joint Training ===
    print(f"\n[PHASE 2] Unfreezing All Scales - Joint Training...")
    
    # Unfreeze medium scale
    for param in model.medium_scale.parameters():
        param.requires_grad = True
    
    # Load best from phase 1
    checkpoint = torch.load(os.path.join(save_dir, 'best_hmcmil.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    
    trainer = HMCMILTrainer(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler, device, save_dir,
        epochs=60, patience=20, grad_clip=1.0
    )
    
    phase2_acc = trainer.train("PHASE 2: Joint Multi-Scale Training")
    
    # === PHASE 3: Fine-tuning with Stronger SupCon ===
    print(f"\n[PHASE 3] Fine-Tuning with Stronger Contrastive Loss...")
    
    # Load best from phase 2
    checkpoint = torch.load(os.path.join(save_dir, 'best_hmcmil.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Stronger contrastive learning
    criterion = HMCMILLoss(alpha=0.7, gamma=2.5, temperature=0.07, supcon_weight=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    
    trainer = HMCMILTrainer(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler, device, save_dir,
        epochs=20, patience=15, grad_clip=1.0
    )
    
    phase3_acc = trainer.train("PHASE 3: Fine-Tuning with Stronger SupCon")
    
    # === FINAL EVALUATION ===
    test_acc = trainer.evaluate_final()
    
    # === FINAL MESSAGE ===
    print(f"\n{'='*160}")
    print(f"🎯 HMC-MIL TRAINING COMPLETE!")
    print(f"{'='*160}")
    print(f"  Phase 1 (Fine/Coarse Warm-up): {phase1_acc:.2%}")
    print(f"  Phase 2 (Joint Training):      {phase2_acc:.2%}")
    print(f"  Phase 3 (Fine-Tuning):         {phase3_acc:.2%}")
    print(f"  Final Test Accuracy:           {test_acc:.2%}")
    print(f"{'='*160}")
    
    if test_acc >= 0.98:
        print(f"🎉 OUTSTANDING! Test Acc: {test_acc:.2%} → TARGET ACHIEVED!")
        print(f"   Ready for top-tier publication (IEEE JBHI, Nature SR)!")
    elif test_acc >= 0.97:
        print(f"🏆 EXCELLENT! Test Acc: {test_acc:.2%}")
        print(f"   Very close to 98% target - strong results!")
    elif test_acc >= 0.95:
        print(f"✅ VERY GOOD! Test Acc: {test_acc:.2%}")
        print(f"   Significant improvement over baseline (94.76%)!")
    else:
        print(f"👍 GOOD! Test Acc: {test_acc:.2%}")
    print(f"{'='*160}\n")


if __name__ == '__main__':
    main()

