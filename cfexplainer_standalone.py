"""
cfexplainer_standalone.py

Standalone implementation of CFExplainer that extracts explanatory subgraphs
from TripleViewNet without external dependency on 'dig' library.

Explanation mechanism:
- Creates three learnable edge masks (AST, CFG, PDG)
- Optimizes masks to minimize prediction loss while maximizing sparsity
- Masks with values close to 1.0 indicate important edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch_geometric.nn import MessagePassing


class StandaloneCFExplainer:
    """
    Extracts explanatory subgraphs showing which edges cause vulnerability predictions.
    
    Algorithm:
    1. Initialize learnable masks for AST, CFG, PDG edges
    2. For N epochs, optimize masks via gradient descent to:
       - Minimize prediction loss (push model to same prediction)
       - Maximize sparsity (minimize number of important edges)
    3. Return sigmoid(masks) as importance scores per edge
    """
    
    def __init__(self, model, epochs=100, lr=0.01, alpha=0.9, L1_dist=False):
        """
        Args:
            model: TripleViewNet instance
            epochs: Number of optimization iterations
            lr: Learning rate for Adam optimizer
            alpha: Weight for prediction loss vs sparsity (0=sparsity, 1=prediction)
            L1_dist: Use L1 norm for sparsity (vs binary cross-entropy)
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.L1_dist = L1_dist
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Masks will be created in forward()
        self.node_feat_mask = None
        self.edge_mask_ast = None
        self.edge_mask_cfg = None
        self.edge_mask_pdg = None

    def __set_masks__(self, data):
        """Initialize learnable masks for each graph view."""
        x = data.x
        (N, F) = x.size()
        
        # Node feature mask - learn which node features are important
        self.node_feat_mask = nn.Parameter(
            torch.randn(F, device=self.device) * 0.1
        )
        
        def init_edge_mask(E):
            """Initialize edge mask with Xavier/Glorot initialization."""
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            return nn.Parameter(torch.randn(E, device=self.device) * std)
        
        # Three separate masks for AST, CFG, PDG edges
        self.edge_mask_ast = init_edge_mask(data.edge_index_ast.size(1))
        self.edge_mask_cfg = init_edge_mask(data.edge_index_cfg.size(1))
        self.edge_mask_pdg = init_edge_mask(data.edge_index_pdg.size(1))

    def __clear_masks__(self):
        """Reset all masks to None after explanation extraction."""
        self.node_feat_mask = None
        self.edge_mask_ast = None
        self.edge_mask_cfg = None
        self.edge_mask_pdg = None

    def __loss__(self, pred, target):
        """
        Compute loss: prediction_loss + sparsity_loss
        
        Goal: Keep model's prediction while minimizing number of important edges
        """
        pred = pred.view(-1)[0]
        
        # Prediction loss: push model output toward target label
        if target == 1:
            pred_loss = -torch.log(pred + 1e-8)
        else:
            pred_loss = -torch.log(1 - pred + 1e-8)
        
        # Sparsity loss: minimize mask values (fewer important edges)
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
        
        # Combined loss
        loss = self.alpha * pred_loss + (1 - self.alpha) * edge_dist_loss
        return loss

    def gnn_explainer_alg(self, data, target_label):
        """
        Main optimization loop: learn which edges are important.
        
        For each epoch:
        1. Apply masks to node features and edges
        2. Forward pass through model
        3. Compute loss
        4. Backprop and update masks
        """
        optimizer = torch.optim.Adam(
            [self.node_feat_mask, self.edge_mask_ast, self.edge_mask_cfg, self.edge_mask_pdg],
            lr=self.lr
        )
        
        for epoch in range(1, self.epochs + 1):
            # Apply node feature mask (element-wise multiplication with sigmoid)
            h = data.x * self.node_feat_mask.sigmoid().view(1, -1)
            data_masked = data.clone()
            data_masked.x = h
            
            # Forward pass
            pred = self.model(data_masked)
            
            # Compute loss
            loss = self.__loss__(pred, target_label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 2.0)
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f'  CFExplainer Epoch {epoch}/{self.epochs} Loss: {loss.item():.4f}')
        
        # Return sigmoid-normalized masks (0-1 scale for importance scores)
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
        
        # If no target label provided, use model's prediction
        if target_label is None:
            with torch.no_grad():
                pred = self.model(data)
                target_label = int((pred > 0.5).long().item())
        
        # Initialize masks
        self.__clear_masks__()
        self.__set_masks__(data)
        
        # Optimize masks
        edge_masks = self.gnn_explainer_alg(data, target_label)
        
        # Clean up
        self.__clear_masks__()
        
        return {
            "edge_mask_ast": edge_masks[0],
            "edge_mask_cfg": edge_masks[1],
            "edge_mask_pdg": edge_masks[2],
            "target_label": target_label
        }

    def __repr__(self):
        return f'{self.__class__.__name__}(TripleView)'   
