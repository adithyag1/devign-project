import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GlobalAttention

torch.manual_seed(2020)


class TripleViewNet(nn.Module):
    def __init__(self, feature_dim, device):
        super(TripleViewNet, self).__init__()
        self.device = device
        hidden_dim = 64
        fusion_dim = 96

        # ─── AST Branch ───
        self.ast_gnn = GATConv(feature_dim, 32, heads=2, add_self_loops=True).to(device)
        self.ast_norm = nn.LayerNorm(hidden_dim).to(device)
        self.ast_drop = nn.Dropout(0.1).to(device)
        self.ast_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1)).to(device)

        # ─── CFG Branch ───
        self.cfg_gnn = GATConv(feature_dim, 32, heads=2, add_self_loops=True).to(device)
        self.cfg_norm = nn.LayerNorm(hidden_dim).to(device)
        self.cfg_drop = nn.Dropout(0.1).to(device)
        self.cfg_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1)).to(device)

        # ─── PDG Branch ───
        self.pdg_gnn = GATConv(feature_dim, 32, heads=2, add_self_loops=True).to(device)
        self.pdg_norm = nn.LayerNorm(hidden_dim).to(device)
        self.pdg_drop = nn.Dropout(0.1).to(device)
        self.pdg_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1)).to(device)

        # ─── Fusion with LayerNorm (not BatchNorm - more stable for variable batch sizes) ───
        self.fusion_norm = nn.LayerNorm(3 * hidden_dim).to(device)
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        ).to(device)

        # ─── Classifier ───
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        ).to(device)
        
        # ✅ Temperature scaling (learned during training)
        self.register_parameter('temperature', nn.Parameter(torch.tensor(1.0, device=device)))
    
    def _encode_view(self, x, edge_index, batch, gnn, norm, drop, pool):
        """Encode a single graph view with one GATConv layer + GlobalAttention pooling."""
        h = F.elu(norm(gnn(x, edge_index)))
        h = drop(h)
        return pool(h, batch)

        def forward(self, data):
        x = data.x
        batch = data.batch

        h_ast = self._encode_view(x, data.edge_index_ast, batch,
                                  self.ast_gnn, self.ast_norm, self.ast_drop, self.ast_pool)
        h_cfg = self._encode_view(x, data.edge_index_cfg, batch,
                                  self.cfg_gnn, self.cfg_norm, self.cfg_drop, self.cfg_pool)
        h_pdg = self._encode_view(x, data.edge_index_pdg, batch,
                                  self.pdg_gnn, self.pdg_norm, self.pdg_drop, self.pdg_pool)

        combined = torch.cat([h_ast, h_cfg, h_pdg], dim=1)  # [batch, 192]
        
        # ✅ Normalize combined features before fusion
        combined = self.fusion_norm(combined)
        
        fused = self.fusion(combined)                         # [batch, 96]
        logits = self.classifier(fused).view(-1)              # raw logits
        
        # ✅ Scale by temperature
        logits = logits / self.temperature.clamp(min=0.1)
        
        return logits
        
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
