import os
import gc
import json
import random
import argparse
import warnings

import numpy as np
from tqdm import tqdm
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch_scatter

from transformers import AdamW

from src.process.model import TripleViewNet

from helpers import utils
from graph_dataset import VulGraphDataset, collate
from cfexplainer.models.cfexplainer import CFExplainer

warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# UTILS
# =========================================================
def calculate_metrics(y_true, y_pred):
    return {
        'binary_precision': round(precision_score(y_true, y_pred), 4),
        'binary_recall': round(recall_score(y_true, y_pred), 4),
        'binary_f1': round(f1_score(y_true, y_pred), 4),
    }


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)


# =========================================================
# TRAIN
# =========================================================
def train(args, train_loader, model):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.num_train_epochs):
        bar = tqdm(train_loader)
        for batch in bar:
            batch = batch.to(args.device)

            labels = torch_scatter.segment_csr(batch._VULN, batch.ptr).float()
            labels[labels != 0] = 1.0

            logits = model(batch)  # [B]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.set_description(f"loss {loss.item():.4f}")


# =========================================================
# EVAL
# =========================================================
def evaluate(args, loader, model):
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)

            labels = torch_scatter.segment_csr(batch._VULN, batch.ptr).float()
            labels[labels != 0] = 1.0

            logits = model(batch)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).long()

            preds.append(pred.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    labels_all = np.concatenate(labels_all)

    return calculate_metrics(labels_all, preds)


# =========================================================
# HELPER FOR EVAL
# =========================================================
def forward_with_new_edges(model, data, new_edge_index):
    data_clone = data.clone()

    E = new_edge_index.shape[1]
    e1 = E // 3
    e2 = 2 * E // 3

    data_clone.edge_index_ast = new_edge_index[:, :e1]
    data_clone.edge_index_cfg = new_edge_index[:, e1:e2]
    data_clone.edge_index_pdg = new_edge_index[:, e2:]

    return model(data_clone)


# =========================================================
# CFEXPLAINER RUN
# =========================================================
def cfexplainer_run(args, model, dataset):

    explainer = CFExplainer(model=model, epochs=200)
    explainer.device = args.device

    graph_exp_list = []

    for data in dataset:
        data = data.to(args.device)

        with torch.no_grad():
            prob = torch.sigmoid(model(data))

        pred = int((prob > 0.5).long().item())

        explanation = explainer(data, target_label=pred)

        data.edge_mask_ast = explanation["edge_mask_ast"].cpu()
        data.edge_mask_cfg = explanation["edge_mask_cfg"].cpu()
        data.edge_mask_pdg = explanation["edge_mask_pdg"].cpu()

        data.edge_index = torch.cat([
            data.edge_index_ast,
            data.edge_index_cfg,
            data.edge_index_pdg
        ], dim=1).cpu()

        data.edge_weight = torch.cat([
            data.edge_mask_ast,
            data.edge_mask_cfg,
            data.edge_mask_pdg
        ], dim=0).cpu()

        data.pred = pred

        graph_exp_list.append(data.cpu())

    return graph_exp_list


# =========================================================
# EVAL EXPLANATIONS
# =========================================================
def eval_exp(path, model, args):
    graphs = torch.load(path)

    pn = []
    for g in graphs:
        g = g.to(args.device)

        edge_index = g.edge_index
        edge_weight = g.edge_weight

        k = min(args.KM, len(edge_weight))
        _, idx = torch.topk(edge_weight, k=k)

        fac_edge = edge_index[:, idx]
        mask = torch.ones_like(edge_weight).bool()
        mask[idx] = False
        cf_edge = edge_index[:, mask]

        fac_pred = model(g.clone())
        cf_pred = forward_with_new_edges(model, g, cf_edge)

        fac_pred = (torch.sigmoid(fac_pred) > 0.5).long()
        cf_pred = (torch.sigmoid(cf_pred) > 0.5).long()

        pn.append(int(cf_pred != fac_pred))

    print("PN:", sum(pn) / len(pn))


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_explain', action='store_true')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--KM', type=int, default=8)

    args = parser.parse_args()

    args.device = torch.device(
        f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    )

    set_seed(42)

    # ✅ INIT YOUR MODEL
    model = TripleViewNet(feature_dim=768, device=args.device).to(args.device)

    train_ds = VulGraphDataset(
        root=str(utils.processed_dir() / "vul_graph_dataset"),
        partition='train'
    )
    test_ds = VulGraphDataset(
        root=str(utils.processed_dir() / "vul_graph_dataset"),
        partition='test'
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)

    if args.do_train:
        train(args, train_loader, model)

    if args.do_test:
        print(evaluate(args, test_loader, model))

    if args.do_explain:
        graphs = cfexplainer_run(args, model, test_ds)
        torch.save(graphs, "cfexp.pt")
        eval_exp("cfexp.pt", model, args)


if __name__ == "__main__":
    main()