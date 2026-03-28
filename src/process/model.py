import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import GATConv, GlobalAttention, SAGPooling

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Conv(nn.Module):

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(Conv, self).__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)

        fc1_size = get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])

        # Dense layers
        self.fc1 = nn.Linear(fc1_size, 1)
        self.fc2 = nn.Linear(fc2_size, 1)

        # Dropout
        self.drop = nn.Dropout(p=0.2)

        self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        Z = self.mp_1(F.relu(self.conv1d_1(concat)))
        Z = self.mp_2(self.conv1d_2(Z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        res = self.drop(res)
        # res = res.mean(1)
        # print(res, mean)
        sig = torch.sigmoid(torch.flatten(res))
        return sig


class Net(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
        super(Net, self).__init__()
        # 1. Replace Convolution with Attention (GAT)
        # Using Multi-head attention (e.g., 4 heads)
        self.gat = GATConv(gated_graph_conv_args["out_channels"], 
                           gated_graph_conv_args["out_channels"] // 4, 
                           heads=4).to(device)
        
        # 2. Attention-based Pooling instead of MaxPool1d
        self.pooling = GlobalAttention(gate_nn=nn.Linear(gated_graph_conv_args["out_channels"], 1)).to(device)
        
        self.classifier = nn.Sequential(
            nn.Linear(gated_graph_conv_args["out_channels"], 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        ).to(device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GAT layer
        x = self.gat(x, edge_index)
        x = F.elu(x)
        
        # Global Attention Pooling (Extracts features using attention)
        x = self.pooling(x, data.batch)
        
        return self.classifier(x)

class TripleViewNet(nn.Module):
    def __init__(self, feature_dim, device):
        super(TripleViewNet, self).__init__()
        self.device = device
        hidden_dim = 128  # 32 out_channels * 4 heads

        # Input normalization and projection
        self.feature_norm = nn.LayerNorm(feature_dim).to(device)
        self.feature_proj = nn.Linear(feature_dim, feature_dim).to(device)
        self.input_dropout = nn.Dropout(0.1).to(device)

        # --- AST branch: 3-layer GAT with residual connections + BatchNorm ---
        self.ast_gnn1 = GATConv(feature_dim, 32, heads=4, add_self_loops=True).to(device)
        self.ast_bn1 = nn.BatchNorm1d(hidden_dim).to(device)
        self.ast_gnn2 = GATConv(hidden_dim, 32, heads=4, add_self_loops=True).to(device)
        self.ast_bn2 = nn.BatchNorm1d(hidden_dim).to(device)
        self.ast_gnn3 = GATConv(hidden_dim, 32, heads=4, add_self_loops=True).to(device)
        self.ast_bn3 = nn.BatchNorm1d(hidden_dim).to(device)
        self.ast_drop = nn.Dropout(0.1).to(device)
        self.ast_sag = SAGPooling(hidden_dim, ratio=0.5).to(device)
        self.ast_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1)).to(device)

        # --- CFG branch: 3-layer GAT with residual connections + BatchNorm ---
        self.cfg_gnn1 = GATConv(feature_dim, 32, heads=4, add_self_loops=True).to(device)
        self.cfg_bn1 = nn.BatchNorm1d(hidden_dim).to(device)
        self.cfg_gnn2 = GATConv(hidden_dim, 32, heads=4, add_self_loops=True).to(device)
        self.cfg_bn2 = nn.BatchNorm1d(hidden_dim).to(device)
        self.cfg_gnn3 = GATConv(hidden_dim, 32, heads=4, add_self_loops=True).to(device)
        self.cfg_bn3 = nn.BatchNorm1d(hidden_dim).to(device)
        self.cfg_drop = nn.Dropout(0.1).to(device)
        self.cfg_sag = SAGPooling(hidden_dim, ratio=0.5).to(device)
        self.cfg_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1)).to(device)

        # --- PDG branch: 3-layer GAT with residual connections + BatchNorm ---
        self.pdg_gnn1 = GATConv(feature_dim, 32, heads=4, add_self_loops=True).to(device)
        self.pdg_bn1 = nn.BatchNorm1d(hidden_dim).to(device)
        self.pdg_gnn2 = GATConv(hidden_dim, 32, heads=4, add_self_loops=True).to(device)
        self.pdg_bn2 = nn.BatchNorm1d(hidden_dim).to(device)
        self.pdg_gnn3 = GATConv(hidden_dim, 32, heads=4, add_self_loops=True).to(device)
        self.pdg_bn3 = nn.BatchNorm1d(hidden_dim).to(device)
        self.pdg_drop = nn.Dropout(0.1).to(device)
        self.pdg_sag = SAGPooling(hidden_dim, ratio=0.5).to(device)
        self.pdg_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1)).to(device)

        # Multi-head attention cross-view fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        ).to(device)

        # 4-layer classifier with BatchNorm (input: 3 * 128 = 384)
    
    self.classifier = nn.Sequential(
        nn.Linear(3 * hidden_dim, 512),
        nn.GroupNorm(8, 512),  # Changed from BatchNorm1d
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.GroupNorm(8, 256),  # Changed from BatchNorm1d
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.GroupNorm(8, 128),  # Changed from BatchNorm1d
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    ).to(device)
    def _encode_view(self, x, edge_index, batch,
                     gnn1, bn1, gnn2, bn2, gnn3, bn3, drop, sag_pool, global_pool):
        """Encode a single graph view with 3-layer GAT + SAGPooling + GlobalAttention."""
        # Layer 1
        h = F.elu(bn1(gnn1(x, edge_index)))
        h = drop(h)

        # Layer 2 with residual connection
        h = F.elu(bn2(gnn2(h, edge_index)) + 0.5 * h)

        # Layer 3 with residual connection
        h = F.elu(bn3(gnn3(h, edge_index)) + 0.5 * h)

        # SAGPooling + GlobalAttention hybrid pooling
        try:
            h_pool, edge_idx_pool, _, batch_pool, _, _ = sag_pool(h, edge_index, batch=batch)
            h_graph = global_pool(h_pool, batch_pool)
        except (RuntimeError, ValueError):
            # Fall back to GlobalAttention only if SAGPooling fails (e.g. too few nodes)
            h_graph = global_pool(h, batch)

        return h_graph

    def forward(self, data):
        x = data.x
        batch = data.batch

        # Input preprocessing
        x = self.feature_norm(x)
        x = F.leaky_relu(self.feature_proj(x), 0.01)
        x = self.input_dropout(x)

        # Encode each view independently
        h_ast = self._encode_view(
            x, data.edge_index_ast, batch,
            self.ast_gnn1, self.ast_bn1, self.ast_gnn2, self.ast_bn2,
            self.ast_gnn3, self.ast_bn3, self.ast_drop, self.ast_sag, self.ast_pool,
        )
        h_cfg = self._encode_view(
            x, data.edge_index_cfg, batch,
            self.cfg_gnn1, self.cfg_bn1, self.cfg_gnn2, self.cfg_bn2,
            self.cfg_gnn3, self.cfg_bn3, self.cfg_drop, self.cfg_sag, self.cfg_pool,
        )
        h_pdg = self._encode_view(
            x, data.edge_index_pdg, batch,
            self.pdg_gnn1, self.pdg_bn1, self.pdg_gnn2, self.pdg_bn2,
            self.pdg_gnn3, self.pdg_bn3, self.pdg_drop, self.pdg_sag, self.pdg_pool,
        )

        # Cross-view attention fusion: [batch, 3, hidden_dim]
        views = torch.stack([h_ast, h_cfg, h_pdg], dim=1)
        attn_out, _ = self.cross_attn(views, views, views)

        # Flatten fused views: [batch, 3 * hidden_dim]
        combined = attn_out.reshape(attn_out.size(0), -1)

        return self.classifier(combined).view(-1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
