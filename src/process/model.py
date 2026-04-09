import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

class TripleViewNet(nn.Module):
    def __init__(self, feature_dim, device):
        super(TripleViewNet, self).__init__()
        self.device = device
        hidden_dim = 64
        
        # 1. Input Normalization
        self.input_norm = nn.BatchNorm1d(feature_dim)

        # 2. Optimized GNN Branches (3-Layer GAT with 2 heads as per your repo)
        # We keep the 64-dim output (32 channels * 2 heads)
        self.ast_gnn = GATConv(feature_dim, 32, heads=2, add_self_loops=True)
        self.cfg_gnn = GATConv(feature_dim, 32, heads=2, add_self_loops=True)
        self.pdg_gnn = GATConv(feature_dim, 32, heads=2, add_self_loops=True)

        # 3. Gated Attention Fusion (THE UPGRADE)
        # This layer learns to weight the 3 views (AST, CFG, PDG)
        self.gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3), # Outputs 3 weights: one for each view
            nn.Softmax(dim=-1)
        )
        
        # 4. Final Deep Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    
    def _encode_view(self, x, edge_index, batch, gnn1, gnn2, gnn3, norm, drop, pool):
        """Encode with 3 GAT layers + JK + Mixed Max/Mean Pooling."""
        h1 = F.elu(gnn1(x, edge_index))
        h2 = F.elu(gnn2(h1, edge_index))
        h3 = F.elu(gnn3(h2, edge_index))
        
        # Jumping Knowledge: retains info from all layers
        h = torch.cat([h1, h2, h3], dim=-1)
        h = norm(h)
        h = drop(h)

        # Mixed Pooling: Max captures sharp bugs, Mean captures overall context
        from torch_geometric.nn import global_mean_pool
        p_max = global_max_pool(h, batch)
        p_mean = global_mean_pool(h, batch)

        return torch.cat([p_max, p_mean], dim=1)

    def forward(self, data):
        x = data.x
        if x.size(0) > 1:
            x = self.input_norm(x)
            
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        # Encode 3 Views using Mixed Pooling (Max + Mean)
        # We use a shortcut to 64-dim for the gate
        h_ast = global_max_pool(F.elu(self.ast_gnn(x, data.edge_index_ast)), batch)
        h_cfg = global_max_pool(F.elu(self.cfg_gnn(x, data.edge_index_cfg)), batch)
        h_pdg = global_max_pool(F.elu(self.pdg_gnn(x, data.edge_index_pdg)), batch)

        # Calculate dynamic weights for this specific code snippet
        # We use the average of the views to decide the importance
        avg_view = (h_ast + h_cfg + h_pdg) / 3.0
        weights = self.gate(avg_view) # [batch, 3]

        # Apply Gated Fusion: Weighted Sum of Views
        fused = (weights[:, 0:1] * h_ast + 
                 weights[:, 1:2] * h_cfg + 
                 weights[:, 2:3] * h_pdg)

        logits = self.classifier(fused).view(-1)
        return logits

    def get_optimizer_groups(self, base_weight_decay: float):
        """Apply stronger L2 regularization (Weight Decay) to the classifier to force generalization."""
        # Find parameters belonging to the fusion and classifier layers
        classifier_ids = {id(p) for p in self.classifier.parameters()}
        classifier_ids |= {id(p) for p in self.fusion.parameters()}
        
        gnn_params, head_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in classifier_ids:
                head_params.append(p) # The 'brain'
            else:
                gnn_params.append(p) # The 'eyes' (GNN)
                
        return [
            {"params": gnn_params, "weight_decay": base_weight_decay},
            {"params": head_params, "weight_decay": base_weight_decay * 10}, # 10x penalty
        ]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))