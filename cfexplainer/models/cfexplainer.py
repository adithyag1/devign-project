from math import sqrt
import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.nn import MessagePassing

from dig.xgraph.method.base_explainer import ExplainerBase
from dig.version import debug

from typing import Union


class CFExplainer(ExplainerBase):

    def __init__(self,
                 model: torch.nn.Module,
                 epochs: int = 100,
                 lr: float = 0.01,
                 alpha: float = 0.9,
                 explain_graph: bool = True,
                 L1_dist: bool = False):

        super().__init__(model, epochs, lr, explain_graph)

        self.alpha = alpha
        self.L1_dist = L1_dist

    # ---------------------------------------------------
    # MASK INITIALIZATION (3 EDGE MASKS)
    # ---------------------------------------------------
    def __set_masks__(self, data):

        x = data.x
        edge_index_ast = data.edge_index_ast
        edge_index_cfg = data.edge_index_cfg
        edge_index_pdg = data.edge_index_pdg

        (N, F) = x.size()

        self.node_feat_mask = torch.nn.Parameter(
            torch.randn(F, device=self.device) * 0.1
        )

        def init_edge_mask(E):
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            return torch.nn.Parameter(torch.randn(E, device=self.device) * std)

        self.edge_mask_ast = init_edge_mask(edge_index_ast.size(1))
        self.edge_mask_cfg = init_edge_mask(edge_index_cfg.size(1))
        self.edge_mask_pdg = init_edge_mask(edge_index_pdg.size(1))

        # attach masks ONLY to correct layers
        self.model.ast_gnn.explain = True
        self.model.cfg_gnn.explain = True
        self.model.pdg_gnn.explain = True

        self.model.ast_gnn._edge_mask = self.edge_mask_ast
        self.model.cfg_gnn._edge_mask = self.edge_mask_cfg
        self.model.pdg_gnn._edge_mask = self.edge_mask_pdg

        self.model.ast_gnn._apply_sigmoid = True
        self.model.cfg_gnn._apply_sigmoid = True
        self.model.pdg_gnn._apply_sigmoid = True

    # ---------------------------------------------------
    def __clear_masks__(self):

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._apply_sigmoid = True

        self.node_feat_mask = None
        self.edge_mask_ast = None
        self.edge_mask_cfg = None
        self.edge_mask_pdg = None

    # ---------------------------------------------------
    # LOSS (BINARY)
    # ---------------------------------------------------
    def __loss__(self, pred: Tensor, target: int):

        pred = pred.view(-1)[0]

        if target == 1:
            pred_loss = -torch.log(pred + 1e-8)
        else:
            pred_loss = -torch.log(1 - pred + 1e-8)

        # sparsity loss
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

    # ---------------------------------------------------
    # MAIN OPTIMIZATION
    # ---------------------------------------------------
    def gnn_explainer_alg(self, data, target_label):

        self.to(data.x.device)

        optimizer = torch.optim.Adam(
            [self.node_feat_mask,
             self.edge_mask_ast,
             self.edge_mask_cfg,
             self.edge_mask_pdg],
            lr=self.lr
        )

        for epoch in range(1, self.epochs + 1):

            # feature masking
            h = data.x * self.node_feat_mask.sigmoid().view(1, -1)

            data.x = h

            pred = self.model(data)

            loss = self.__loss__(pred, target_label)

            if epoch % 20 == 0 and debug:
                print(f'Epoch {epoch} Loss: {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 2.0)
            optimizer.step()

        return (
            self.edge_mask_ast.sigmoid().detach(),
            self.edge_mask_cfg.sigmoid().detach(),
            self.edge_mask_pdg.sigmoid().detach()
        )

    # ---------------------------------------------------
    # FORWARD
    # ---------------------------------------------------
    def forward(self, data, target_label: int = None):

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