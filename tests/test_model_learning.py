import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.process.model import TripleViewNet
from src.data import datamanager as data


def test_data_quality():
    """Check if positive and negative samples are distinguishable."""
    print("\n" + "="*60)
    print("TEST: Data Quality - Are Samples Distinguishable?")
    print("="*60)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
    
    pos_samples = train_df[train_df['target'] == 1]
    neg_samples = train_df[train_df['target'] == 0]
    
    print(f"\nPositive samples: {len(pos_samples)}")
    print(f"Negative samples: {len(neg_samples)}")
    
    pos_data = pos_samples['input'].iloc[0]
    neg_data = neg_samples['input'].iloc[0]
    
    print(f"\nPositive sample - nodes: {pos_data.x.shape[0]}, dims: {pos_data.x.shape[1]}")
    print(f"Negative sample - nodes: {neg_data.x.shape[0]}, dims: {neg_data.x.shape[1]}")
    
    pos_x = pos_data.x.cpu().numpy()
    neg_x = neg_data.x.cpu().numpy()
    
    pos_mean = pos_x.mean()
    neg_mean = neg_x.mean()
    pos_std = pos_x.std()
    neg_std = neg_x.std()
    
    print(f"\nPositive X - mean: {pos_mean:.6f}, std: {pos_std:.6f}")
    print(f"Negative X - mean: {neg_mean:.6f}, std: {neg_std:.6f}")
    
    euclidean_dist = np.sqrt(np.sum((pos_x.mean(axis=0) - neg_x.mean(axis=0))**2))
    print(f"\nEuclidean distance between sample means: {euclidean_dist:.6f}")
    
    if euclidean_dist < 0.1:
        print("⚠️ ERROR: Positive and negative samples are INDISTINGUISHABLE!")
        print("   Data is mislabeled or identical - model CAN'T learn")
        return False
    
    print("✅ Samples are distinguishable")
    return True


def test_edge_indices():
    """Check if edge indices are present and non-empty."""
    print("\n" + "="*60)
    print("TEST: Edge Indices - Graph Structure Present?")
    print("="*60)
    
    train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
    
    pos_samples = train_df[train_df['target'] == 1]
    neg_samples = train_df[train_df['target'] == 0]
    
    pos_data = pos_samples['input'].iloc[0]
    neg_data = neg_samples['input'].iloc[0]
    
    print(f"\nPositive sample edges:")
    print(f"  AST: {pos_data.edge_index_ast.shape[1]} edges")
    print(f"  CFG: {pos_data.edge_index_cfg.shape[1]} edges")
    print(f"  PDG: {pos_data.edge_index_pdg.shape[1]} edges")
    
    print(f"\nNegative sample edges:")
    print(f"  AST: {neg_data.edge_index_ast.shape[1]} edges")
    print(f"  CFG: {neg_data.edge_index_cfg.shape[1]} edges")
    print(f"  PDG: {neg_data.edge_index_pdg.shape[1]} edges")
    
    if pos_data.edge_index_ast.shape[1] == 0 and neg_data.edge_index_ast.shape[1] == 0:
        print("⚠️ ERROR: No edges in any graphs!")
        return False
    
    print("✅ Edge indices present")
    return True


def test_model_overfit_single_sample():
    """Test if model can overfit to a single sample."""
    print("\n" + "="*60)
    print("TEST: Model Overfitting - Can It Learn a Single Sample?")
    print("="*60)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
    
    pos_samples = train_df[train_df['target'] == 1]
    pos_data = pos_samples['input'].iloc[0]
    pos_data = pos_data.to(DEVICE)
    pos_data.batch = torch.zeros(pos_data.x.shape[0], dtype=torch.long).to(DEVICE)
    
    model = TripleViewNet(feature_dim=769, device=DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"\nTraining model to memorize ONE positive sample (target=1)...")
    print(f"Sample has {pos_data.x.shape[0]} nodes\n")
    
    losses = []
    probs = []
    
    for step in range(100):
        optimizer.zero_grad()
        
        logits = model(pos_data).view(-1)
        target = torch.tensor([1.0], device=DEVICE)
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        optimizer.step()
        
        prob = torch.sigmoid(logits).item()
        losses.append(loss.item())
        probs.append(prob)
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.6f}, logits={logits.item():.6f}, prob={prob:.4f}, grad_norm={grad_norm:.4f}")
    
    with torch.no_grad():
        logits_final = model(pos_data).view(-1)
        prob_final = torch.sigmoid(logits_final).item()
    
    print(f"\nAfter 100 steps:")
    print(f"  Loss change: {losses[0]:.6f} → {losses[-1]:.6f}")
    print(f"  Prob change: {probs[0]:.4f} → {prob_final:.4f}")
    print(f"  Final predicted class: {1 if prob_final > 0.5 else 0}")
    
    if prob_final < 0.6:
        print(f"\n❌ CRITICAL: Model CANNOT overfit to a single positive sample!")
        print(f"   Expected prob > 0.6, got {prob_final:.4f}")
        print(f"   This means:")
        print(f"   - Model architecture is broken")
        print(f"   - Gradients not flowing properly")
        print(f"   - Loss function has a bug")
        return False
    
    if losses[0] - losses[-1] < 0.01:
        print(f"\n❌ ERROR: Loss did NOT decrease!")
        print(f"   Model is not learning at all")
        return False
    
    print(f"\n✅ Model CAN overfit - issue is elsewhere (data/training)")
    return True


def test_model_overfit_negative_sample():
    """Test if model can push negative samples to class 0."""
    print("\n" + "="*60)
    print("TEST: Model Overfitting - Can It Push Negative to Class 0?")
    print("="*60)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
    
    neg_samples = train_df[train_df['target'] == 0]
    neg_data = neg_samples['input'].iloc[0]
    neg_data = neg_data.to(DEVICE)
    neg_data.batch = torch.zeros(neg_data.x.shape[0], dtype=torch.long).to(DEVICE)
    
    model = TripleViewNet(feature_dim=769, device=DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"\nTraining model to memorize ONE negative sample (target=0)...")
    print(f"Sample has {neg_data.x.shape[0]} nodes\n")
    
    losses = []
    probs = []
    
    for step in range(100):
        optimizer.zero_grad()
        
        logits = model(neg_data).view(-1)
        target = torch.tensor([0.0], device=DEVICE)
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()
        
        prob = torch.sigmoid(logits).item()
        losses.append(loss.item())
        probs.append(prob)
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.6f}, logits={logits.item():.6f}, prob={prob:.4f}")
    
    with torch.no_grad():
        logits_final = model(neg_data).view(-1)
        prob_final = torch.sigmoid(logits_final).item()
    
    print(f"\nAfter 100 steps:")
    print(f"  Loss change: {losses[0]:.6f} → {losses[-1]:.6f}")
    print(f"  Prob change: {probs[0]:.4f} → {prob_final:.4f}")
    print(f"  Final predicted class: {1 if prob_final > 0.5 else 0}")
    
    if prob_final > 0.4:
        print(f"\n❌ CRITICAL: Model CANNOT push negative to class 0!")
        print(f"   Expected prob < 0.4, got {prob_final:.4f}")
        return False
    
    if losses[0] - losses[-1] < 0.01:
        print(f"\n❌ ERROR: Loss did NOT decrease!")
        return False
    
    print(f"\n✅ Model can push negatives to class 0")
    return True


def test_batch_predictions():
    """Test model predictions on a batch."""
    print("\n" + "="*60)
    print("TEST: Batch Predictions - Diverse Output?")
    print("="*60)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
    
    model = TripleViewNet(feature_dim=769, device=DEVICE)
    model.eval()
    
    probs_list = []
    
    with torch.no_grad():
        for idx in range(min(50, len(train_df))):
            graph = train_df['input'].iloc[idx]
            graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            graph = graph.to(DEVICE)
            
            logits = model(graph).view(-1)
            prob = torch.sigmoid(logits).item()
            probs_list.append(prob)
    
    probs = np.array(probs_list)
    
    print(f"\nPredictions from 50 random samples:")
    print(f"  Mean: {probs.mean():.4f}")
    print(f"  Std: {probs.std():.4f}")
    print(f"  Min: {probs.min():.4f}")
    print(f"  Max: {probs.max():.4f}")
    print(f"  Class 0 count: {(probs < 0.5).sum()}")
    print(f"  Class 1 count: {(probs >= 0.5).sum()}")
    
    if probs.std() < 0.01:
        print(f"\n⚠️ WARNING: All predictions are near {probs.mean():.4f} (no diversity)")
        return False
    
    if (probs < 0.5).sum() == 0 or (probs >= 0.5).sum() == 0:
        print(f"\n❌ ERROR: Model predicts only ONE class!")
        return False
    
    print(f"\n✅ Predictions are diverse")
    return True


if __name__ == "__main__":
    results = []
    
    results.append(("Data Quality", test_data_quality()))
    results.append(("Edge Indices", test_edge_indices()))
    results.append(("Model Overfit Positive", test_model_overfit_single_sample()))
    results.append(("Model Overfit Negative", test_model_overfit_negative_sample()))
    results.append(("Batch Predictions", test_batch_predictions()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + ("="*60))
    if all_passed:
        print("ALL TESTS PASSED ✅")
        print("Model should be training correctly")
    else:
        print("SOME TESTS FAILED ❌")
        print("Review failures above to find the issue")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)
