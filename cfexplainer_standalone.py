import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch_geometric.nn import MessagePassing


class StandaloneCFExplainer:
    
    def __init__(self, model, epochs=100, lr=0.01, alpha=0.9, L1_dist=False):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.L1_dist = L1_dist
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_feat_mask = None
        self.edge_mask_ast = None
        self.edge_mask_cfg = None
        self.edge_mask_pdg = None

    def __call__(self, data, target_label=None):
        return self.forward(data, target_label)

    def __set_masks__(self, data):
        """Initialize learnable masks for each graph view."""
        x = data.x
        (N, F) = x.size()
        self.node_feat_mask = nn.Parameter(
            torch.randn(F, device=self.device) * 0.1
        )
        
        def init_edge_mask(E):
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            return nn.Parameter(torch.randn(E, device=self.device) * std)
        self.edge_mask_ast = init_edge_mask(data.edge_index_ast.size(1))
        self.edge_mask_cfg = init_edge_mask(data.edge_index_cfg.size(1))
        self.edge_mask_pdg = init_edge_mask(data.edge_index_pdg.size(1))

    def __clear_masks__(self):
        self.node_feat_mask = None
        self.edge_mask_ast = None
        self.edge_mask_cfg = None
        self.edge_mask_pdg = None

    def __loss__(self, pred, target):
        pred = pred.view(-1)[0]
        if target == 1:
            pred_loss = -torch.log(pred + 1e-8)
        else:
            pred_loss = -torch.log(1 - pred + 1e-8)
        def edge_loss(mask):
            m = mask.sigmoid()
            if self.L1_dist:
                return torch.norm(1 - m, p=1)
            else:
                return F.binary_cross_entropy(m, torch.ones_like(m))
        
        edge_dist_loss = (
            edge_loss(self.edge_mask_ast) +
            edge_loss(self.edge_mask_cfg) +
            edge_loss(self.edge_mask_pdg)
        )
        loss = self.alpha * pred_loss + (1 - self.alpha) * edge_dist_loss
        return loss

    def gnn_explainer_alg(self, data, target_label):
        optimizer = torch.optim.Adam(
            [self.node_feat_mask, self.edge_mask_ast, self.edge_mask_cfg, self.edge_mask_pdg],
            lr=self.lr
        )
        
        for epoch in range(1, self.epochs + 1):
            h = data.x * self.node_feat_mask.sigmoid().view(1, -1)
            data_masked = data.clone()
            data_masked.x = h
            pred = self.model(data_masked)
            loss = self.__loss__(pred, target_label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 2.0)
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f'  CFExplainer Epoch {epoch}/{self.epochs} Loss: {loss.item():.4f}')
        return (
            self.edge_mask_ast.sigmoid().detach(),
            self.edge_mask_cfg.sigmoid().detach(),
            self.edge_mask_pdg.sigmoid().detach()
        )

    def forward(self, data, target_label=None):
        """
        Extract explanation: which edges are important for the prediction?
        
        Args:
            data: torch_geometric.data.Data with edge_index_ast/cfg/pdg
            target_label: Integer label (0=clean, 1=vulnerable) to explain
                         If None, use model's prediction
        
        Returns:
            dict with edge_mask_ast, edge_mask_cfg, edge_mask_pdg, target_label
        """
        self.model.eval()
        if target_label is None:
            with torch.no_grad():
                pred = self.model(data)
                target_label = int((pred > 0.5).long().item())
        self.__clear_masks__()
        self.__set_masks__(data)
        edge_masks = self.gnn_explainer_alg(data, target_label)
        self.__clear_masks__()
        
        return {
            "edge_mask_ast": edge_masks[0],
            "edge_mask_cfg": edge_masks[1],
            "edge_mask_pdg": edge_masks[2],
            "target_label": target_label
        }

    def __repr__(self):
        return f'{self.__class__.__name__}(TripleView)'   
