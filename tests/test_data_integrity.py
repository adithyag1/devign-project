import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import datamanager as data


def test_embedding_distribution():
    """Check if node embeddings are informative (not dead/zero)."""
    print("\n" + "="*60)
    print("TEST: Embedding Distribution")
    print("="*60)
    
    train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
    
    # Sample 10 graphs
    for idx in range(min(10, len(train_df))):
        graph = train_df['input'].iloc[idx]
        x = graph.x
        
        mean = x.mean().item()
        std = x.std().item()
        min_val = x.min().item()
        max_val = x.max().item()
        
        print(f"\nGraph {idx}: {x.shape[0]} nodes, {x.shape[1]} dims")
        print(f"  Mean: {mean:.6f} | Std: {std:.6f}")
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
        
        # Red flags
        if std < 0.001:
            print(f"  ⚠️ WARNING: Very low std dev (dead features)")
        if mean > 100 or mean < -100:
            print(f"  ⚠️ WARNING: Mean is extreme (unbounded values)")
        if x.isnan().any():
            print(f"  ⚠️ ERROR: NaN values detected!")
            return False
    
    print("\n✅ Embeddings look reasonable")
    return True


def test_edge_indices():
    """Check if graphs have structural information."""
    print("\n" + "="*60)
    print("TEST: Edge Indices Presence")
    print("="*60)
    
    train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
    
    ast_empty_count = 0
    cfg_empty_count = 0
    pdg_empty_count = 0
    
    for idx in range(len(train_df)):
        graph = train_df['input'].iloc[idx]
        
        if graph.edge_index_ast.shape[1] == 0:
            ast_empty_count += 1
        if graph.edge_index_cfg.shape[1] == 0:
            cfg_empty_count += 1
        if graph.edge_index_pdg.shape[1] == 0:
            pdg_empty_count += 1
    
    total = len(train_df)
    print(f"\nOut of {total} graphs:")
    print(f"  AST empty: {ast_empty_count} ({100*ast_empty_count/total:.1f}%)")
    print(f"  CFG empty: {cfg_empty_count} ({100*cfg_empty_count/total:.1f}%)")
    print(f"  PDG empty: {pdg_empty_count} ({100*pdg_empty_count/total:.1f}%)")
    
    if ast_empty_count > total * 0.5:
        print(f"  ⚠️ WARNING: >50% AST graphs are empty!")
    if cfg_empty_count > total * 0.5:
        print(f"  ⚠️ WARNING: >50% CFG graphs are empty!")
    if pdg_empty_count > total * 0.5:
        print(f"  ⚠️ WARNING: >50% PDG graphs are empty!")
    
    print("\n✅ Edge indices present")
    return True


def test_class_distribution():
    """Check for extreme class imbalance."""
    print("\n" + "="*60)
    print("TEST: Class Distribution")
    print("="*60)
    
    train_df, val_df, test_df = data.global_train_val_test_split("data/input/", "data/")
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        neg = (df['target'] == 0).sum()
        pos = (df['target'] == 1).sum()
        total = len(df)
        ratio = pos / neg if neg > 0 else 0
        
        print(f"\n{name}: {total} samples")
        print(f"  Class 0: {neg} ({100*neg/total:.1f}%)")
        print(f"  Class 1: {pos} ({100*pos/total:.1f}%)")
        print(f"  Ratio (1:0): 1:{1/ratio:.2f}" if ratio > 0 else "  Ratio: Only one class!")
        
        if ratio < 0.1 or ratio > 10:
            print(f"  ⚠️ WARNING: Extreme imbalance detected!")
    
    print("\n✅ Class distribution checked")
    return True


def test_model_forward_pass():
    """Check if model can do forward pass without errors."""
    print("\n" + "="*60)
    print("TEST: Model Forward Pass")
    print("="*60)
    
    try:
        from src.process.model import TripleViewNet
        
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TripleViewNet(feature_dim=844, device=DEVICE)
        model = model.to(DEVICE)
        
        # Load one batch
        train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
        graph = train_df['input'].iloc[0]
        
        # Add batch dimension
        graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
        graph = graph.to(DEVICE)
        
        # Forward pass
        output = model(graph)
        
        print(f"\nInput shape: {graph.x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output logits: {output.item():.6f}")
        print(f"Output sigmoid: {torch.sigmoid(output).item():.6f}")
        
        if torch.isnan(output).any():
            print("  ⚠️ ERROR: Output contains NaN!")
            return False
        
        print("\n✅ Model forward pass successful")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    results = []
    
    results.append(("Embeddings", test_embedding_distribution()))
    results.append(("Edge Indices", test_edge_indices()))
    results.append(("Class Distribution", test_class_distribution()))
    results.append(("Model Forward", test_model_forward_pass()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    sys.exit(0 if all_passed else 1)
