"""
cfexplainer_wrapper.py

Wraps CFExplainer to extract explanatory subgraphs with actual code snippets
and line numbers, providing human-readable vulnerability analysis reports.
"""

import json
import os
import torch
import numpy as np

from cfexplainer_standalone import StandaloneCFExplainer as CFExplainer

# Node label mapping (from src/utils/objects/cpg/node.py)
NODE_LABELS = {
    0: "Block",
    1: "Call",
    2: "Comment",
    3: "ControlStructure",
    4: "File",
    5: "Identifier",
    6: "FieldIdentifier",
    7: "Literal",
    8: "Local",
    9: "Member",
    10: "MetaData",
    11: "Method",
    12: "MethodInst",
    13: "MethodParameterIn",
    14: "MethodParameterOut",
    15: "MethodReturn",
    16: "Namespace",
    17: "NamespaceBlock",
    18: "Return",
    19: "Type",
    20: "TypeDecl",
    21: "Unknown",
}


class ExplanationExtractor:
    """
    Integrates CFExplainer with code extraction to produce human-readable
    vulnerability explanations including actual source code snippets.
    """

    def __init__(self, model, device, epochs=200, top_k=5):
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
    def _get_label_name(type_id):
        """Convert numeric type ID to label name."""
        if isinstance(type_id, str):
            try:
                type_id = int(type_id)
            except (ValueError, TypeError):
                return type_id
        
        return NODE_LABELS.get(type_id, f"Unknown({type_id})")

    @staticmethod
    def _node_info(node):
        """
        Extract code snippet, line number and type label from a Node.
        
        Priority for code extraction:
        1. Properties.code() – already handles TYPE_FULL_NAME + CODE combination
        2. TYPE_FULL_NAME only – when no CODE property exists (code() returns None)
        3. METHOD_FULL_NAME property (for operators and methods)
        4. Node label (Block, Call, etc.)
        5. "<no-info>" fallback
        """
        # Primary: node.get_code() delegates to Properties.code(), which handles
        # the TYPE_FULL_NAME + CODE combination and returns None when there is
        # no CODE property (or when CODE is empty / the "<empty>" placeholder).
        code = node.get_code()

        if code is None:
            # No CODE property – fall back to TYPE_FULL_NAME
            node_type = node.properties.get_type()
            if node_type and node_type.strip() and node_type != "ANY":
                code = node_type
        
        if not code or code.strip() == "":
            # Try to get METHOD_FULL_NAME (for operators)
            method_name = node.properties.pairs.get("METHOD_FULL_NAME", "")
            if method_name and method_name.strip():
                # Extract just the operator/method name
                code = method_name.split(".")[-1]
        
        if not code or code.strip() == "":
            # Use node label as fallback
            label_name = ExplanationExtractor._get_label_name(node.label)
            code = f"{label_name}"
        
        # Final cleanup
        code = str(code).strip() if code else "<no-info>"
        if not code:
            code = "<no-info>"
        
        line = node.get_line_number()
        label = ExplanationExtractor._get_label_name(node.label)
        
        return {"code": code, "line": line, "type": label}

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
            src_info = self._node_info(node_list[src]) if src < len(node_list) else {"code": "<no-info>", "line": None, "type": "Unknown"}
            dst_info = self._node_info(node_list[dst]) if dst < len(node_list) else {"code": "<no-info>", "line": None, "type": "Unknown"}
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
        key_ops = [n["code"] for n in vulnerable_nodes[:3] if n["code"] and n["code"] != "<no-info>"]
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
    """Print a concise, elegant vulnerability report to stdout."""
    pred_label = explanation.get("target_label", int(probability > 0.5))
    vulnerable_nodes = explanation["vulnerable_nodes"]
    vulnerability_paths = explanation["vulnerability_paths"]
    summary = explanation["summary"]

    width = 80
    
    # Filter out noise: only keep nodes with actual code
    important_nodes = [
        n for n in vulnerable_nodes 
        if n["code"] and n["code"] != "<no-info>" and n["line"]
    ]

    print("\n" + "=" * width)
    print(f"{'VULNERABILITY REPORT':^{width}}")
    print("=" * width)
    print(f"Function     : {func_name}")
    print(f"File         : {os.path.basename(source_file)}")
    status_text = "[!] VULNERABLE" if probability > 0.5 else "[+] CLEAN"
    print(f"Prediction   : {status_text} ({probability:.1%})")
    print("=" * width)

    # Show critical code lines
    if important_nodes:
        print(f"\n[*] CRITICAL CODE LINES ({len(important_nodes)} identified):\n")
        for i, node in enumerate(important_nodes[:5], 1):  # Top 5 only
            line_str = f"Line {node['line']}" if node["line"] is not None else "Unknown"
            print(f"    {i}. {line_str}: {node['code']}")

    # Show data flow paths (top 1 per view)
    print(f"\n[*] VULNERABILITY DATA FLOW:\n")
    for view in ("ast", "cfg", "pdg"):
        paths = vulnerability_paths.get(view, [])
        if paths:
            top_path = paths[0]  # Only show the most important path
            from_line = f"Line {top_path['from_line']}" if top_path['from_line'] else "?"
            to_line = f"Line {top_path['to_line']}" if top_path['to_line'] else "?"
            importance = top_path['importance']
            
            print(f"    {view.upper()} View:")
            print(f"      {from_line}: {top_path['from_code']}")
            print(f"               ↓ (score: {importance:.3f})")
            print(f"      {to_line}: {top_path['to_code']}\n")

    print(f"[*] Summary: {summary}")
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