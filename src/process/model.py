import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

class TripleViewNet(nn.Module):
    def __init__(self, feature_dim, device):
        super(TripleViewNet, self).__init__()
        self.device = device

        # 1. Input Normalization
        self.input_norm = nn.BatchNorm1d(feature_dim)

        # 2. 3-Layer GAT Branches (2 heads each), dropout=0.2 in each GATConv
        def make_gat_branch():
            return nn.ModuleList([
                GATConv(feature_dim, 32, heads=2, dropout=0.2, add_self_loops=True),
                GATConv(64, 32, heads=2, dropout=0.2, add_self_loops=True),
                GATConv(64, 32, heads=2, dropout=0.2, add_self_loops=True)
            ])

        self.ast_branch = make_gat_branch()
        self.cfg_branch = make_gat_branch()
        self.pdg_branch = make_gat_branch()

        # 3. Jumping Knowledge & View Normalization
        # JK concatenates 3 layers of 64-dim = 192-dim
        self.jk_norm = nn.LayerNorm(192)
        self.jk_drop = nn.Dropout(0.3)

        # 4. Per-view projection after dual pooling (max+mean => 384 -> 128)
        self.view_proj = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 5. Gated Attention Fusion
        # Learns which view (AST, CFG, PDG) is most important for a given sample
        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        # 6. Final Deep Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def _encode_view(self, x, edge_index, batch, branch):
        """Encode with 3 GAT layers + Jumping Knowledge (JK) + Dual Global Pooling."""
        h1 = F.elu(branch[0](x, edge_index))
        h2 = F.elu(branch[1](h1, edge_index))
        h3 = F.elu(branch[2](h2, edge_index))

        # Jumping Knowledge (JK): Concatenate features from all layers
        h_combined = torch.cat([h1, h2, h3], dim=-1)  # [Nodes, 192]
        h_combined = self.jk_norm(h_combined)
        h_combined = self.jk_drop(h_combined)

        # Dual Pooling: concatenate max and mean for richer graph representation
        h_max = global_max_pool(h_combined, batch)   # [Batch, 192]
        h_mean = global_mean_pool(h_combined, batch)  # [Batch, 192]
        return torch.cat([h_max, h_mean], dim=-1)     # [Batch, 384]

    def forward(self, data):
        x = data.x
        if x.size(0) > 1:
            x = self.input_norm(x)

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        # Step 1: Encode 3 Views (AST, CFG, PDG) using 3-layer GATs + JK + dual pool
        h_ast = self._encode_view(x, data.edge_index_ast, batch, self.ast_branch)
        h_cfg = self._encode_view(x, data.edge_index_cfg, batch, self.cfg_branch)
        h_pdg = self._encode_view(x, data.edge_index_pdg, batch, self.pdg_branch)

        # Step 2: Per-view projection (384 -> 128)
        h_ast = self.view_proj(h_ast)
        h_cfg = self.view_proj(h_cfg)
        h_pdg = self.view_proj(h_pdg)

        # Step 3: Gated Fusion
        # Stack views: [Batch, 3, 128]
        stacked_views = torch.stack([h_ast, h_cfg, h_pdg], dim=1)

        # Calculate dynamic weights based on the average projected features
        avg_features = torch.mean(stacked_views, dim=1)  # [Batch, 128]
        weights = self.gate(avg_features)                # [Batch, 3]

        # Weighted sum: [Batch, 1, 3] @ [Batch, 3, 128] -> [Batch, 128]
        fused = torch.bmm(weights.unsqueeze(1), stacked_views).squeeze(1)

        # Step 4: Classification — return logits shape [B]
        return self.classifier(fused).view(-1)

    def get_optimizer_groups(self, base_weight_decay: float):
        """Apply stronger regularization to the classifier to prevent memorization."""
        head_ids = {id(p) for p in self.classifier.parameters()}
        head_ids |= {id(p) for p in self.gate.parameters()}
        head_ids |= {id(p) for p in self.view_proj.parameters()}

        gnn_params, head_params = [], []
        for p in self.parameters():
            if not p.requires_grad: continue
            if id(p) in head_ids:
                head_params.append(p)
            else:
                gnn_params.append(p)

        return [
            {"params": gnn_params, "weight_decay": base_weight_decay},
            {"params": head_params, "weight_decay": base_weight_decay * 10},
        ]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))