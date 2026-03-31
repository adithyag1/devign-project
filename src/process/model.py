import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GlobalAttention, global_max_pool

torch.manual_seed(2020)


class TripleViewNet(nn.Module):
    def __init__(self, feature_dim, device):
        super(TripleViewNet, self).__init__()
        self.device = device
        hidden_dim = 64
        fusion_dim = 96

        # ─── Input Normalization ───
        self.input_norm = nn.BatchNorm1d(feature_dim)

        # ─── AST Branch (3 Layers) ───
        self.ast_gnn1 = GATConv(feature_dim, 32, heads=2, add_self_loops=True)
        self.ast_gnn2 = GATConv(64, 32, heads=2, add_self_loops=True)
        self.ast_gnn3 = GATConv(64, 32, heads=2, add_self_loops=True)
        # JK: cat([h1, h2, h3]) = 3 * hidden_dim = 192 (each h_i is 64-dim: 32 channels * 2 heads)
        self.jk_dim = 3 * hidden_dim
        jk_dim = self.jk_dim
        self.ast_norm = nn.LayerNorm(jk_dim)
        self.ast_drop = nn.Dropout(0.3)
        self.ast_pool = GlobalAttention(gate_nn=nn.Linear(jk_dim, 1))

        # ─── CFG Branch (3 Layers) ───
        self.cfg_gnn1 = GATConv(feature_dim, 32, heads=2, add_self_loops=True)
        self.cfg_gnn2 = GATConv(64, 32, heads=2, add_self_loops=True)
        self.cfg_gnn3 = GATConv(64, 32, heads=2, add_self_loops=True)
        self.cfg_norm = nn.LayerNorm(jk_dim)
        self.cfg_drop = nn.Dropout(0.3)
        self.cfg_pool = GlobalAttention(gate_nn=nn.Linear(jk_dim, 1))

        # ─── PDG Branch (3 Layers) ───
        self.pdg_gnn1 = GATConv(feature_dim, 32, heads=2, add_self_loops=True)
        self.pdg_gnn2 = GATConv(64, 32, heads=2, add_self_loops=True)
        self.pdg_gnn3 = GATConv(64, 32, heads=2, add_self_loops=True)
        self.pdg_norm = nn.LayerNorm(jk_dim)
        self.pdg_drop = nn.Dropout(0.3)
        self.pdg_pool = GlobalAttention(gate_nn=nn.Linear(jk_dim, 1))

        # ─── Fusion with LayerNorm (not BatchNorm - more stable for variable batch sizes) ───
        # Each view outputs 2 * jk_dim (att pool + max pool); 3 views total = 6 * jk_dim
        self.fusion_norm = nn.LayerNorm(6 * jk_dim)
        self.fusion = nn.Sequential(
            nn.Linear(6 * jk_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased dropout to 0.5
        )

        # ─── Classifier ───
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

        # ✅ Initialize all weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform to prevent gradient explosion."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _encode_view(self, x, edge_index, batch, gnn1, gnn2, gnn3, norm, drop, pool):
        """Encode a single graph view with three GATConv layers + JK concat + GlobalAttention/Max pooling."""
        h1 = F.elu(gnn1(x, edge_index))
        h2 = F.elu(gnn2(h1, edge_index))
        h3 = F.elu(gnn3(h2, edge_index))
        # Jumping Knowledge: concatenate features from all neighbourhood scales
        h = torch.cat([h1, h2, h3], dim=-1)
        h = norm(h)
        h = drop(h)

        # Capture BOTH the global context (attention) and the sharpest local signal (max)
        pool_att = pool(h, batch)
        pool_max = global_max_pool(h, batch)

        return torch.cat([pool_att, pool_max], dim=1)

    def forward(self, data):
        x = data.x
        # Normalize inputs to prevent gradient saturation from large graph feature values
        if x.size(0) > 1:
            x = self.input_norm(x)
        # Handle both batched and single graph cases
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            # Single graph: create batch tensor (all zeros = same graph)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        h_ast = self._encode_view(x, data.edge_index_ast, batch,
                                  self.ast_gnn1, self.ast_gnn2, self.ast_gnn3, self.ast_norm, self.ast_drop, self.ast_pool)
        h_cfg = self._encode_view(x, data.edge_index_cfg, batch,
                                  self.cfg_gnn1, self.cfg_gnn2, self.cfg_gnn3, self.cfg_norm, self.cfg_drop, self.cfg_pool)
        h_pdg = self._encode_view(x, data.edge_index_pdg, batch,
                                  self.pdg_gnn1, self.pdg_gnn2, self.pdg_gnn3, self.pdg_norm, self.pdg_drop, self.pdg_pool)

        combined = torch.cat([h_ast, h_cfg, h_pdg], dim=1)  # [batch, 3 views * 2 pools * jk_dim = 1152]
        
        # ✅ Normalize combined features before fusion
        combined = self.fusion_norm(combined)
        
        fused = self.fusion(combined)                         # [batch, 96]
        logits = self.classifier(fused).view(-1)              # raw logits

        # Removed the self.output_scale multiplier to prevent gradient vanishing

        return logits
        
    def get_optimizer_groups(self, base_weight_decay: float):
        """Return parameter groups with stronger L2 regularization on classifier layers."""
        classifier_ids = {id(p) for p in self.classifier.parameters()}
        classifier_ids |= {id(p) for p in self.fusion.parameters()}
        gnn_params, head_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in classifier_ids:
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