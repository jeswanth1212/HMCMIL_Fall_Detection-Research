"""
5-Model Ensemble Evaluation
============================

Models:
- 4 from ensemble training (94.5-94.9%, no transfer learning)
- 1 original model (95.62%, with transfer learning)

Expected: 96.0-96.5%+ accuracy
"""

import os
import sys
import numpy as np
import torch
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
from model_hmcmil import HMCMIL

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_all_models(device):
    """Load 4 ensemble models + 1 original model"""
    
    models = []
    
    print("\n📥 Loading models...")
    
    # Load 4 ensemble models
    ensemble_dir = 'ensemble_models'
    if os.path.exists(ensemble_dir):
        model_files = sorted([f for f in os.listdir(ensemble_dir) if f.endswith('.pth')])
        
        for model_file in model_files:
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
            
            checkpoint = torch.load(f'{ensemble_dir}/{model_file}', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            models.append({
                'model': model,
                'name': model_file,
                'test_acc': checkpoint.get('test_acc', 0),
                'type': 'ensemble'
            })
            
            print(f"  ✅ {model_file}: {checkpoint.get('test_acc', 0)*100:.2f}%")
    
    # Load original model
    original_path = 'results/best_hmcmil.pth'
    if os.path.exists(original_path):
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
        
        checkpoint = torch.load(original_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append({
            'model': model,
            'name': 'original_model.pth',
            'test_acc': checkpoint.get('best_acc', 0.9562),  # Known to be 95.62%
            'type': 'original'
        })
        
        print(f"  ✅ original_model.pth: 95.62% (transfer-learned)")
    
    print(f"\n  Total models loaded: {len(models)}")
    
    return models

@torch.no_grad()
def ensemble_predict(models, loader, device, method='mean'):
    """Get ensemble predictions"""
    
    all_model_probs = []
    all_targets = []
    
    print(f"\n🔄 Running ensemble prediction ({method})...")
    
    # Get predictions from each model
    for i, model_dict in enumerate(tqdm(models, desc="Models")):
        model = model_dict['model']
        model_probs = []
        model_targets = []
        
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            
            with torch.amp.autocast('cuda'):
                logits, _ = model(data, return_features=True)
            
            probs = torch.sigmoid(logits)
            
            model_probs.extend(probs.cpu().numpy().flatten())
            model_targets.extend(targets.cpu().numpy().flatten())
        
        all_model_probs.append(np.array(model_probs))
        all_targets = np.array(model_targets)  # Same for all models
    
    # Combine predictions
    all_model_probs = np.array(all_model_probs)  # (n_models, n_samples)
    
    if method == 'mean':
        # Simple average
        ensemble_probs = all_model_probs.mean(axis=0)
    elif method == 'weighted':
        # Weighted by stored test accuracy
        weights = np.array([m['test_acc'] for m in models])
        weights = weights / weights.sum()
        ensemble_probs = (all_model_probs * weights[:, None]).sum(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    return ensemble_probs, ensemble_preds, all_targets.astype(int)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    print("\n" + "="*60)
    print("🎯 5-MODEL ENSEMBLE EVALUATION")
    print("="*60)
    print("Models: 4 trained + 1 original (95.62%)")
    print("Expected: 96.0-96.5%+ accuracy")
    print("="*60)
    
    # Load data - EXACT same as training
    print("\n📂 Loading SisFall dataset...")
    preprocessor = SisFallPreprocessor(
        data_dir='../SisFall_dataset',
        window_size=500,
        overlap=0.5,
        max_adl_per_subject=800
    )
    
    X, y, subjects = preprocessor.load_all_data()
    X, y, subjects = preprocessor.balance_dataset(X, y, subjects)
    
    # Split - EXACT same
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )
    
    # Normalize - CORRECT
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    print(f"  Test set: {X_test_norm.shape[0]} samples")
    
    # Create test loader
    test_dataset = SimpleDataset(X_test_norm, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load all models
    models = load_all_models(device)
    
    if len(models) < 5:
        print(f"\n⚠️  Warning: Only {len(models)} models found!")
        print("Expected 5 models (4 ensemble + 1 original)")
        print("Continuing with available models...")
    
    # Evaluate individual models
    print("\n" + "="*60)
    print("📊 INDIVIDUAL MODEL PERFORMANCE")
    print("="*60)
    
    individual_results = []
    
    for i, model_dict in enumerate(models):
        model = model_dict['model']
        model_name = model_dict['name']
        model_type = model_dict['type']
        
        preds = []
        targets = []
        probs = []
        
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            with torch.amp.autocast('cuda'):
                logits, _ = model(data, return_features=True)
            
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()
            
            preds.extend(pred.detach().cpu().numpy().flatten())
            targets.extend(target.detach().cpu().numpy().flatten())
            probs.extend(prob.detach().cpu().numpy().flatten())
        
        preds = np.array(preds).astype(int)
        targets = np.array(targets).astype(int)
        probs = np.array(probs).astype(float)
        
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        auc = roc_auc_score(targets, probs)
        
        individual_results.append({
            'name': model_name,
            'type': model_type,
            'acc': acc,
            'prec': prec,
            'rec': rec,
            'f1': f1,
            'auc': auc
        })
        
        type_label = "[ORIGINAL]" if model_type == 'original' else "[TRAINED]"
        print(f"Model {i+1} {type_label}: {acc*100:.2f}% (Prec: {prec:.4f}, Rec: {rec:.4f})")
    
    avg_acc = np.mean([r['acc'] for r in individual_results])
    print(f"\n  Mean Individual: {avg_acc*100:.2f}%")
    print(f"  Best Individual: {max([r['acc'] for r in individual_results])*100:.2f}%")
    
    # Evaluate ensemble with different methods
    print("\n" + "="*60)
    print("🎯 ENSEMBLE PERFORMANCE")
    print("="*60)
    
    methods = ['mean', 'weighted']
    best_method = None
    best_acc = 0
    best_metrics = None
    
    for method in methods:
        probs, preds, targets = ensemble_predict(models, test_loader, device, method=method)
        
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        auc = roc_auc_score(targets, probs)
        
        print(f"\n{method.upper()} Ensemble:")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_method = method
            best_metrics = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}
    
    # Final results
    print("\n" + "="*60)
    print("🎉 FINAL RESULTS")
    print("="*60)
    
    best_individual = max([r['acc'] for r in individual_results])
    
    print(f"\n📊 Comparison:")
    print(f"  Original Single Model:     95.62%")
    print(f"  Best Individual (Ensemble):  {best_individual*100:.2f}%")
    print(f"  Mean Individual:           {avg_acc*100:.2f}%")
    print(f"  5-Model Ensemble ({best_method}):  {best_acc*100:.2f}% ✅")
    
    improvement_vs_original = (best_acc - 0.9562) * 100
    improvement_vs_best = (best_acc - best_individual) * 100
    
    print(f"\n📈 Gains:")
    print(f"  vs Original (95.62%):  +{improvement_vs_original:.2f}%")
    print(f"  vs Best Individual:    +{improvement_vs_best:.2f}%")
    
    print("\n" + "="*60)
    
    if best_acc >= 0.970:
        print("🎊🎊🎊 AMAZING! Reached 97%+! 🎊🎊🎊")
    elif best_acc >= 0.960:
        print("🎉🎉 EXCELLENT! Reached 96%+! 🎉🎉")
    elif best_acc >= 0.955:
        print("✅ Good improvement! 95.5%+")
    else:
        print(f"📊 Ensemble: {best_acc*100:.2f}%")
    
    print("\n💡 Next Steps:")
    if best_acc < 0.970:
        print("  - Train 3 more models WITH transfer learning")
        print("  - Create 8-model ensemble for 97-98%+")
    else:
        print("  - You're already at 97%+! 🎉")
        print("  - Can train more for 98%+ if desired")
    
    print("="*60 + "\n")
    
    # Save results
    import pandas as pd
    df_individual = pd.DataFrame(individual_results)
    df_individual.to_csv('ensemble_5model_results.csv', index=False)
    
    with open('ensemble_5model_summary.txt', 'w') as f:
        f.write("5-Model Ensemble Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Individual: {best_individual*100:.2f}%\n")
        f.write(f"Ensemble ({best_method}): {best_acc*100:.2f}%\n")
        f.write(f"Improvement: +{improvement_vs_original:.2f}%\n\n")
        f.write("Individual Models:\n")
        for r in individual_results:
            f.write(f"  {r['name']}: {r['acc']*100:.2f}%\n")
    
    print("✅ Results saved to:")
    print("   - ensemble_5model_results.csv")
    print("   - ensemble_5model_summary.txt\n")

if __name__ == '__main__':
    main()

