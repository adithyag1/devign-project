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
        self.jk_dim = 3 * hidden_dim
        jk_dim = self.jk_dim
        self.ast_norm = nn.LayerNorm(jk_dim)
        self.ast_drop = nn.Dropout(0.5)  # ⬆️ Increased from 0.3
        self.ast_pool = GlobalAttention(gate_nn=nn.Linear(jk_dim, 1))

        # ─── CFG Branch (3 Layers) ───
        self.cfg_gnn1 = GATConv(feature_dim, 32, heads=2, add_self_loops=True)
        self.cfg_gnn2 = GATConv(64, 32, heads=2, add_self_loops=True)
        self.cfg_gnn3 = GATConv(64, 32, heads=2, add_self_loops=True)
        self.cfg_norm = nn.LayerNorm(jk_dim)
        self.cfg_drop = nn.Dropout(0.5)  # ⬆️ Increased from 0.3
        self.cfg_pool = GlobalAttention(gate_nn=nn.Linear(jk_dim, 1))

        # ─── PDG Branch (3 Layers) ───
        self.pdg_gnn1 = GATConv(feature_dim, 32, heads=2, add_self_loops=True)
        self.pdg_gnn2 = GATConv(64, 32, heads=2, add_self_loops=True)
        self.pdg_gnn3 = GATConv(64, 32, heads=2, add_self_loops=True)
        self.pdg_norm = nn.LayerNorm(jk_dim)
        self.pdg_drop = nn.Dropout(0.5)  # ⬆️ Increased from 0.3
        self.pdg_pool = GlobalAttention(gate_nn=nn.Linear(jk_dim, 1))

        # ─── Fusion ───
        self.fusion_norm = nn.LayerNorm(6 * jk_dim)
        self.fusion = nn.Sequential(
            nn.Linear(6 * jk_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # ⬆️ Increased from 0.3
        )

        # ─── Classifier ───
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            # This is the last line of defense against overfitting
            nn.Dropout(0.4), # ⬆️ Increased from 0.2
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
        """Encode a single graph view with three GATConv layers + JK concat + Mixed Pooling."""
        h1 = F.elu(gnn1(x, edge_index))
        h2 = F.elu(gnn2(h1, edge_index))
        h3 = F.elu(gnn3(h2, edge_index))
        
        # Jumping Knowledge: concatenate features from all scales
        h = torch.cat([h1, h2, h3], dim=-1)
        h = norm(h)
        h = drop(h)

        # ─── IMPROVED POOLING ───
        # Capture the 'sharpest' signal (Max) and the 'overall' context (Attention)
        pool_att = pool(h, batch)
        pool_max = global_max_pool(h, batch)

        # Concatenate both poolings (Total dim = 2 * jk_dim)
        return torch.cat([pool_att, pool_max], dim=1)

    def forward(self, data):
        x = data.x
        
        # 1. Input Normalization: Prevents large feature values from saturating gradients
        if x.size(0) > 1:
            x = self.input_norm(x)
            
        # 2. Handle batching
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        # 3. Encode the 3 Graph Views
        h_ast = self._encode_view(x, data.edge_index_ast, batch,
                                  self.ast_gnn1, self.ast_gnn2, self.ast_gnn3, self.ast_norm, self.ast_drop, self.ast_pool)
        h_cfg = self._encode_view(x, data.edge_index_cfg, batch,
                                  self.cfg_gnn1, self.cfg_gnn2, self.cfg_gnn3, self.cfg_norm, self.cfg_drop, self.cfg_pool)
        h_pdg = self._encode_view(x, data.edge_index_pdg, batch,
                                  self.pdg_gnn1, self.pdg_gnn2, self.pdg_gnn3, self.pdg_norm, self.pdg_drop, self.pdg_pool)

        # 4. Fusion
        combined = torch.cat([h_ast, h_cfg, h_pdg], dim=1)  # [batch, 1152]
        
        # Apply the fusion normalization defined in __init__
        combined = self.fusion_norm(combined)
        
        fused = self.fusion(combined)                         # [batch, fusion_dim]
        logits = self.classifier(fused).view(-1)              # Final prediction
        
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