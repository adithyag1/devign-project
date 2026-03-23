"""
cfexplainer_wrapper.py

Wraps CFExplainer to extract explanatory subgraphs with actual code snippets
and line numbers, providing human-readable vulnerability analysis reports.
"""

import json
import os
import torch
import numpy as np

from cfexplainer.models.cfexplainer import CFExplainer


class ExplanationExtractor:
    """
    Integrates CFExplainer with code extraction to produce human-readable
    vulnerability explanations including actual source code snippets.
    """

    def __init__(self, model, device, epochs=200, top_k=8):
        """
        Args:
            model:   Trained TripleViewNet instance.
            device:  torch.device to run on.
            epochs:  CFExplainer optimization epochs (higher = more precise).
            top_k:   Number of most important edges to extract per view.
        """
        self.model = model
        self.device = device
        self.top_k = top_k

        self.explainer = CFExplainer(model=model, epochs=epochs)
        self.explainer.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_explanation_with_code(self, pyg_data, nodes_dict, pred_label):
        """
        Run CFExplainer and return a fully annotated explanation dict.

        Args:
            pyg_data:    torch_geometric.data.Data with edge_index_ast/cfg/pdg.
            nodes_dict:  Ordered dict {node_id: Node} from func_obj.get_nodes().
            pred_label:  Integer prediction label (0 = clean, 1 = vulnerable).

        Returns:
            dict with keys:
              - edge_masks:          raw sigmoid masks per view
              - top_edges:           top-K edge indices per view
              - vulnerable_nodes:    deduplicated list of {code, line, type}
              - vulnerability_paths: edge paths per view with from/to code
              - summary:             human-readable one-line summary
        """
        # Build an index list so we can map int positions → node objects
        node_list = list(nodes_dict.values())

        # Run CFExplainer
        explanation = self.explainer(pyg_data, target_label=pred_label)

        edge_masks = {
            "ast": explanation["edge_mask_ast"],
            "cfg": explanation["edge_mask_cfg"],
            "pdg": explanation["edge_mask_pdg"],
        }

        edge_indices = {
            "ast": pyg_data.edge_index_ast,
            "cfg": pyg_data.edge_index_cfg,
            "pdg": pyg_data.edge_index_pdg,
        }

        top_edges = {}
        for view in ("ast", "cfg", "pdg"):
            top_edges[view] = self._top_k_edges(
                edge_masks[view], edge_indices[view], self.top_k
            )

        # Collect unique vulnerable node indices across all views
        vulnerable_node_indices = set()
        for view in ("ast", "cfg", "pdg"):
            if top_edges[view].shape[1] > 0:
                for idx in top_edges[view].flatten():
                    vulnerable_node_indices.add(int(idx))

        vulnerable_nodes = self._nodes_to_info(
            sorted(vulnerable_node_indices), node_list
        )

        # Build edge paths (from_node → to_node with importance score)
        vulnerability_paths = {}
        for view in ("ast", "cfg", "pdg"):
            vulnerability_paths[view] = self._build_paths(
                top_edges[view],
                edge_masks[view],
                edge_indices[view],
                node_list,
                self.top_k,
            )

        summary = self._build_summary(vulnerable_nodes, vulnerability_paths, pred_label)

        return {
            "edge_masks": {v: m.cpu().tolist() for v, m in edge_masks.items()},
            "top_edges": {v: e.tolist() for v, e in top_edges.items()},
            "vulnerable_nodes": vulnerable_nodes,
            "vulnerability_paths": vulnerability_paths,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _top_k_edges(mask, edge_index, k):
        """Return (2, min(k, E)) array of the highest-scored edge indices."""
        if mask.numel() == 0 or edge_index.size(1) == 0:
            return np.zeros((2, 0), dtype=np.int64)
        k = min(k, mask.numel())
        _, idx = torch.topk(mask, k=k)
        return edge_index[:, idx].cpu().numpy()

    @staticmethod
    def _node_info(node):
        """Extract code snippet, line number and type label from a Node."""
        code = node.get_code() or ""
        line = node.get_line_number()
        label = node.label if node.label else "Unknown"
        return {"code": code.strip(), "line": line, "type": label}

    def _nodes_to_info(self, indices, node_list):
        """Convert a list of integer node positions to info dicts (deduplicated)."""
        seen_codes = set()
        result = []
        for i in indices:
            if i >= len(node_list):
                continue
            info = self._node_info(node_list[i])
            key = (info["code"], info["line"])
            if key not in seen_codes:
                seen_codes.add(key)
                result.append(info)
        return result

    def _build_paths(self, top_edge_arr, mask, edge_index, node_list, k):
        """
        Build a list of edge path dicts sorted by descending importance.

        Each dict: {from_code, from_line, to_code, to_line, importance}
        """
        if mask.numel() == 0 or edge_index.size(1) == 0:
            return []

        k = min(k, mask.numel())
        scores, idx = torch.topk(mask, k=k)
        idx_np = idx.cpu().numpy()
        scores_np = scores.cpu().numpy()
        ei_np = edge_index.cpu().numpy()

        paths = []
        for rank, (edge_pos, score) in enumerate(zip(idx_np, scores_np)):
            src = int(ei_np[0, edge_pos])
            dst = int(ei_np[1, edge_pos])
            src_info = self._node_info(node_list[src]) if src < len(node_list) else {"code": "", "line": None, "type": ""}
            dst_info = self._node_info(node_list[dst]) if dst < len(node_list) else {"code": "", "line": None, "type": ""}
            paths.append({
                "from_code": src_info["code"],
                "from_line": src_info["line"],
                "to_code": dst_info["code"],
                "to_line": dst_info["line"],
                "importance": float(score),
            })
        return paths

    @staticmethod
    def _build_summary(vulnerable_nodes, vulnerability_paths, pred_label):
        total_edges = sum(len(v) for v in vulnerability_paths.values())
        view_counts = ", ".join(
            f"{v.upper()}:{len(p)}" for v, p in vulnerability_paths.items()
        )
        key_ops = [n["code"] for n in vulnerable_nodes[:3] if n["code"]]
        ops_str = ", ".join(key_ops) if key_ops else "N/A"
        status = "VULNERABLE" if pred_label == 1 else "CLEAN"
        return (
            f"Status: {status}. Found {total_edges} critical edges across 3 views "
            f"({view_counts}). Key operations: {ops_str}."
        )


# ------------------------------------------------------------------
# Report generation helpers
# ------------------------------------------------------------------

def print_explanation_report(func_name, source_file, probability, explanation):
    """Print a human-readable vulnerability report to stdout."""
    pred_label = explanation.get("target_label", int(probability > 0.5))
    vulnerable_nodes = explanation["vulnerable_nodes"]
    vulnerability_paths = explanation["vulnerability_paths"]
    summary = explanation["summary"]

    width = 80
    print("\n" + "=" * width)
    print(f"{'VULNERABILITY REPORT WITH EXPLANATION':^{width}}")
    print("=" * width)
    print(f"  Target Function : {func_name}")
    print(f"  Source File     : {os.path.basename(source_file)}")
    print(f"  Prediction      : {'[!] VULNERABLE' if probability > 0.5 else '[+] CLEAN'}")
    print(f"  Confidence      : {probability:.2%}")
    print("-" * width)

    if vulnerable_nodes:
        print(f"\n[*] Critical Nodes Identified ({len(vulnerable_nodes)} total):")
        for i, node in enumerate(vulnerable_nodes, 1):
            line_str = f"line {node['line']}" if node["line"] is not None else "no line info"
            print(f"    {i}. [{node['type']}] at {line_str}")
            if node["code"]:
                print(f"       Code: {node['code']}")

    print(f"\n[*] Vulnerability Propagation Paths (3 views):")
    for view, paths in vulnerability_paths.items():
        print(f"\n    {view.upper()} Graph ({len(paths)} critical edges):")
        if not paths:
            print("      (no edges found)")
            continue
        for i, edge in enumerate(paths, 1):
            from_line = f"line {edge['from_line']}" if edge["from_line"] is not None else "?"
            to_line   = f"line {edge['to_line']}"   if edge["to_line"]   is not None else "?"
            print(f"      {i}. {edge['from_code'] or '<no code>'} ({from_line})")
            print(f"         ↓ [{edge['importance']:.4f}]")
            print(f"         {edge['to_code'] or '<no code>'} ({to_line})")

    print(f"\n[*] Summary: {summary}")
    print("=" * width + "\n")


def save_explanation_report(output_path, func_name, source_file, probability, explanation):
    """Save full explanation as a JSON report."""
    report = {
        "function_name": func_name,
        "source_file": os.path.basename(source_file),
        "prediction": "vulnerable" if probability > 0.5 else "clean",
        "confidence": round(probability, 6),
        "vulnerable_nodes": explanation["vulnerable_nodes"],
        "vulnerability_paths": {
            view: paths for view, paths in explanation["vulnerability_paths"].items()
        },
        "edge_masks": explanation["edge_masks"],
        "summary": explanation["summary"],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[*] Explanation report saved to: {output_path}")
