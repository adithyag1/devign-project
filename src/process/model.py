import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch_geometric.utils import dropout_edge

class TripleViewNet(nn.Module):
    def __init__(self, feature_dim, device):
        super(TripleViewNet, self).__init__()
        self.device = device

        self.input_norm = nn.BatchNorm1d(feature_dim)

        def make_gat_branch():
            return nn.ModuleList([
                GATConv(feature_dim, 32, heads=2, add_self_loops=True, dropout=0.2),  # 64
                GATConv(64, 32, heads=2, add_self_loops=True, dropout=0.2),           # 64
                GATConv(64, 32, heads=2, add_self_loops=True, dropout=0.2),           # 64
            ])

        self.ast_branch = make_gat_branch()
        self.cfg_branch = make_gat_branch()
        self.pdg_branch = make_gat_branch()

        # JK: 64*3 = 192
        self.jk_norm = nn.LayerNorm(192)
        self.jk_drop = nn.Dropout(0.3)

        # pooled view = max(192)+mean(192)=384
        self.view_proj = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

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
        h1 = F.elu(branch[0](x, edge_index))
        h2 = F.elu(branch[1](h1, edge_index))
        h3 = F.elu(branch[2](h2, edge_index))

        h = torch.cat([h1, h2, h3], dim=-1)   # [N, 192]  (JK)
        h = self.jk_norm(h)
        h = self.jk_drop(h)

        h_max = global_max_pool(h, batch)     # [B, 192]
        h_mean = global_mean_pool(h, batch)   # [B, 192]
        return torch.cat([h_max, h_mean], dim=-1)  # [B, 384]

    def forward(self, data):
        x = data.x
        if x.size(0) > 1:
            x = self.input_norm(x)

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        # edge dropout regularization (train only)
        edge_ast, _ = dropout_edge(data.edge_index_ast, p=0.10, training=self.training)
        edge_cfg, _ = dropout_edge(data.edge_index_cfg, p=0.10, training=self.training)
        edge_pdg, _ = dropout_edge(data.edge_index_pdg, p=0.10, training=self.training)

        h_ast = self._encode_view(x, edge_ast, batch, self.ast_branch)  # [B,384]
        h_cfg = self._encode_view(x, edge_cfg, batch, self.cfg_branch)  # [B,384]
        h_pdg = self._encode_view(x, edge_pdg, batch, self.pdg_branch)  # [B,384]

        h_ast = self.view_proj(h_ast)  # [B,128]
        h_cfg = self.view_proj(h_cfg)  # [B,128]
        h_pdg = self.view_proj(h_pdg)  # [B,128]

        views = torch.stack([h_ast, h_cfg, h_pdg], dim=1)   # [B,3,128]
        gate_in = (h_ast + h_cfg + h_pdg) / 3.0            # [B,128]
        alpha = self.gate(gate_in).unsqueeze(-1)           # [B,3,1]
        fused = (views * alpha).sum(dim=1)                 # [B,128]

        logits = self.classifier(fused).squeeze(-1)        # [B]
        return logits

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