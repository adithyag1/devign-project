import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.process.model import TripleViewNet
from src.data import datamanager as data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")
model = TripleViewNet(feature_dim=844, device=DEVICE)
model = model.to(DEVICE)

# Hook to inspect intermediate outputs
activations = {}

def hook(name):
    def h(model, input, output):
        activations[name] = output.detach()
    return h

model.ast_pool.register_forward_hook(hook("ast_pool"))
model.cfg_pool.register_forward_hook(hook("cfg_pool"))
model.pdg_pool.register_forward_hook(hook("pdg_pool"))
model.fusion.register_forward_hook(hook("fusion"))

# Test on 5 graphs
print("\n" + "="*60)
print("INTERMEDIATE OUTPUT ANALYSIS")
print("="*60)

for graph_idx in range(5):
    print(f"\nGraph {graph_idx}:")
    
    graph = train_df['input'].iloc[graph_idx]
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    graph = graph.to(DEVICE)

    with torch.no_grad():
        output = model(graph)

    print(f"  AST pool:  shape={activations['ast_pool'].shape}, range=[{activations['ast_pool'].min():.6f}, {activations['ast_pool'].max():.6f}], std={activations['ast_pool'].std():.6f}")
    print(f"  CFG pool:  shape={activations['cfg_pool'].shape}, range=[{activations['cfg_pool'].min():.6f}, {activations['cfg_pool'].max():.6f}], std={activations['cfg_pool'].std():.6f}")
    print(f"  PDG pool:  shape={activations['pdg_pool'].shape}, range=[{activations['pdg_pool'].min():.6f}, {activations['pdg_pool'].max():.6f}], std={activations['pdg_pool'].std():.6f}")
    print(f"  Fusion:    shape={activations['fusion'].shape}, range=[{activations['fusion'].min():.6f}, {activations['fusion'].max():.6f}], std={activations['fusion'].std():.6f}")
    print(f"  Final output (logits): {output.item():.6f}")

print("\n✅ Diagnosis complete - check std/range values above")
