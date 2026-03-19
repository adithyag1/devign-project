import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import GATConv, GlobalAttention

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

        # CHANGE 1: Use a stronger Normalization (LayerNorm often works better than BatchNorm for GNNs)
        self.feature_norm = nn.LayerNorm(feature_dim).to(device) 
        self.feature_proj = nn.Linear(feature_dim, feature_dim).to(device)
        
        # CHANGE 2: Add a Dropout layer for the input features
        self.input_dropout = nn.Dropout(0.3).to(device) 

        self.ast_gnn = GATConv(feature_dim, 32, heads=4, add_self_loops=True).to(device)
        self.cfg_gnn = GATConv(feature_dim, 32, heads=4, add_self_loops=True).to(device)
        self.pdg_gnn = GATConv(feature_dim, 32, heads=4, add_self_loops=True).to(device)
        
        self.pool = GlobalAttention(gate_nn=nn.Linear(128, 1)).to(device)
        
        # CHANGE 3: Update Classifier to use LeakyReLU and more Dropout
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.LeakyReLU(0.01), # Changed from ReLU
            nn.Dropout(0.4),    # Increased dropout
            nn.Linear(128, 1),
        ).to(device)

    def forward(self, data):
        x = data.x
        
        # Apply normalization and dropout early
        x = self.feature_norm(x)
        x = self.feature_proj(x)
        x = F.leaky_relu(x, 0.01) # Use LeakyReLU here too
        x = self.input_dropout(x)

        # Branch 1 (AST)
        h_ast = F.elu(self.ast_gnn(x, data.edge_index_ast))
        h_ast = self.pool(h_ast, data.batch)
        
        # Branch 2 (CFG)
        h_cfg = F.elu(self.cfg_gnn(x, data.edge_index_cfg))
        h_cfg = self.pool(h_cfg, data.batch)
        
        # Branch 3 (PDG)
        h_pdg = F.elu(self.pdg_gnn(x, data.edge_index_pdg))
        h_pdg = self.pool(h_pdg, data.batch)
        
        combined = torch.cat([h_ast, h_cfg, h_pdg], dim=1)
        
        return self.classifier(combined).view(-1)
            
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))