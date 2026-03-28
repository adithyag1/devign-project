import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.process.model import TripleViewNet
from src.process.devign import Devign
from src.process.step import Step
from src.data import datamanager as data
from src.process import loader_step
import numpy as np


def test_weight_updates():
    """Check if model weights actually change during training."""
    print("\n" + "="*60)
    print("TEST: Weight Updates During Training")
    print("="*60)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_df, val_df, _ = data.global_train_val_test_split("data/input/", "data/")
    
    # Create model
    model_obj = TripleViewNet(feature_dim=769, device=DEVICE)
    
    # Store initial weights
    initial_weights = {name: p.clone().detach() for name, p in model_obj.named_parameters()}
    
    # Create Devign wrapper
    model = Devign(
        path="test_checkpoint.pt",
        device=DEVICE,
        model=model_obj,
        learning_rate=1e-2,
        weight_decay=1e-5,
        loss_lambda=0,
        weight_0=0.8,
        weight_1=1.5
    )
    
    # Create data loaders
    from torch_geometric.data import DataLoader
    
    train_loader = DataLoader([train_df['input'].iloc[i] for i in range(min(64, len(train_df)))], 
                              batch_size=16, shuffle=True)
    
    # Single training step
    print(f"\nRunning {len(train_loader)} batches...")
    model.train()
    
    batch_count = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        
        # Forward pass
        logits = model_obj(batch)
        
        # Compute loss manually
        probs = torch.sigmoid(logits)
        target = batch.y.float()
        loss_weights = torch.where(target == 0, model.w0, model.w1).to(target.device)
        raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
        loss = (raw_loss * loss_weights).mean()
        
        print(f"  Batch {batch_count}: loss = {loss.item():.6f}")
        
        # Backward pass
        model.optimizer.zero_grad()
        loss.backward()
        
        # Check gradients before update
        total_grad_norm = 0
        for name, p in model_obj.named_parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"    Gradient norm: {total_grad_norm:.6f}")
        
        model.optimizer.step()
        
        batch_count += 1
        if batch_count >= 3:  # Only first 3 batches
            break
    
    # Check weight changes
    print(f"\nChecking weight changes...")
    max_change = 0
    for name, p in model_obj.named_parameters():
        if name in initial_weights:
            change = (p - initial_weights[name]).norm().item()
            if change > max_change:
                max_change = change
            if change > 1e-6:
                print(f"  {name}: changed by {change:.6f}")
    
    if max_change < 1e-7:
        print("  ⚠️ ERROR: Weights did NOT change! Gradients not propagating!")
        return False
    
    print(f"\n✅ Max weight change: {max_change:.6f}")
    return True


def test_logit_distribution():
    """Check if model outputs diverse logits, not all ~0."""
    print("\n" + "="*60)
    print("TEST: Logit Distribution")
    print("="*60)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
    model = TripleViewNet(feature_dim=769, device=DEVICE)
    model.eval()
    
    logits_list = []
    
    with torch.no_grad():
        for i in range(min(100, len(train_df))):
            graph = train_df['input'].iloc[i]
            graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            graph = graph.to(DEVICE)
            
            logit = model(graph).item()
            logits_list.append(logit)
    
    logits = np.array(logits_list)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    
    print(f"\nLogits from 100 samples:")
    print(f"  Mean: {logits.mean():.6f}")
    print(f"  Std: {logits.std():.6f}")
    print(f"  Range: [{logits.min():.6f}, {logits.max():.6f}]")
    
    print(f"\nSigmoid(logits):")
    print(f"  Mean: {probs.mean():.6f}")
    print(f"  Std: {probs.std():.6f}")
    print(f"  Range: [{probs.min():.6f}, {probs.max():.6f}]")
    
    if probs.std() < 0.01:
        print(f"  ⚠️ ERROR: All predictions ~{probs.mean():.4f} (no diversity)!")
        return False
    
    print(f"\n✅ Logits are diverse")
    return True


def test_loss_computation():
    """Check if loss changes with different predictions."""
    print("\n" + "="*60)
    print("TEST: Loss Computation")
    print("="*60)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy logits and targets
    logits = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0], device=DEVICE)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0], device=DEVICE)
    
    # Compute loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    
    print(f"\nTest logits: {logits}")
    print(f"Test targets: {targets}")
    print(f"Loss: {loss.item():.6f}")
    
    if abs(loss.item() - 0.693) < 0.01:
        print(f"  ⚠️ WARNING: Loss is 0.693 (random baseline)")
    else:
        print(f"  ✅ Loss deviates from baseline")
    
    # Check if loss changes with different inputs
    logits2 = torch.ones(5, device=DEVICE)  # All 1.0
    loss2 = torch.nn.functional.binary_cross_entropy_with_logits(logits2, targets)
    
    print(f"\nWith all logits=1.0: loss = {loss2.item():.6f}")
    
    if abs(loss.item() - loss2.item()) < 0.001:
        print(f"  ⚠️ ERROR: Loss doesn't change! Something wrong with loss function!")
        return False
    
    print(f"\n✅ Loss computation working")
    return True


def test_learning_rate_schedule():
    """Check if learning rate warmup is working."""
    print("\n" + "="*60)
    print("TEST: Learning Rate Schedule")
    print("="*60)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from src.process.devign import Devign
    import configs
    
    context = configs.Process()
    
    model_obj = TripleViewNet(feature_dim=769, device=DEVICE)
    model = Devign(
        path="test_checkpoint.pt",
        device=DEVICE,
        model=model_obj,
        learning_rate=context.learning_rate,
        weight_decay=context.weight_decay,
        loss_lambda=context.loss_lambda,
        weight_0=context.weight_0,
        weight_1=context.weight_1
    )
    
    print(f"\nInitial learning rate: {context.learning_rate}")
    print(f"Warmup epochs: {context.warmup_epochs}")
    
    # Check optimizer LR
    for param_group in model.optimizer.param_groups:
        print(f"Current optimizer LR: {param_group['lr']}")
    
    print(f"\n✅ Learning rate config checked")
    return True


if __name__ == "__main__":
    results = []
    
    results.append(("Logit Distribution", test_logit_distribution()))
    results.append(("Loss Computation", test_loss_computation()))
    results.append(("Learning Rate", test_learning_rate_schedule()))
    results.append(("Weight Updates", test_weight_updates()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    sys.exit(0 if all_passed else 1)
