import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from . import utils
import scipy.sparse as sparse
from graphviz import Digraph

import torch
from torch_geometric.data import Data


def nodelabel2line(label: str):
    try:
        return str(int(label))
    except:
        return label.split(":")[0].split("_")[-1]


def randcolor():
    def r():
        return random.randint(0, 255)
    return "#%02X%02X%02X" % (r(), r(), r())


def get_digraph(nodes, edges, edge_label=True):
    dot = Digraph(comment="Combined Graph (AST + CFG + PDG)")

    nodes = [n + [nodelabel2line(n[1])] for n in nodes]
    colormap = {"": "white"}
    for n in nodes:
        if n[2] not in colormap:
            colormap[n[2]] = randcolor()

    for n in nodes:
        style = {"style": "filled", "fillcolor": colormap[n[2]]}
        dot.node(str(n[0]), str(n[1]), **style)

    for e in edges:
        style = {"color": "black"}

        if e[2] == "AST":
            style["color"] = "black"
        elif e[2] == "CFG":
            style["color"] = "red"
        elif e[2] == "CDG":
            style["color"] = "blue"
        elif e[2] == "REACHING_DEF":
            style["color"] = "orange"
        else:
            style["color"] = "gray"

        style["style"] = "solid"
        style["penwidth"] = "1"

        if edge_label:
            dot.edge(str(e[0]), str(e[1]), e[2], **style)
        else:
            dot.edge(str(e[0]), str(e[1]), **style)

    return dot


def run_joern(filepath: str, verbose: int):
    script_file = utils.external_dir() / "get_func_graph.scala"
    filename = utils.processed_dir() / filepath
    params = f"filename={filename}"
    command = f"joern --script {script_file} --params='{params}'"
    command = str(utils.external_dir() / "joern-cli" / command)

    if verbose > 2:
        utils.debug(command)

    utils.subprocess_cmd(command, verbose=verbose)

    try:
        shutil.rmtree(utils.storage_dir().parent / "workspace" / filename.name)
    except Exception:
        pass


def get_node_edges(filepath: str, verbose=0):
    outdir = Path(filepath).parent
    outfile = outdir / Path(filepath).name

    with open(str(outfile) + ".edges.json", "r") as f:
        edges = json.load(f)
        edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"])
        edges = edges.fillna("")

    with open(str(outfile) + ".nodes.json", "r") as f:
        nodes = json.load(f)
        nodes = pd.DataFrame.from_records(nodes)
        if "controlStructureType" not in nodes.columns:
            nodes["controlStructureType"] = ""
        nodes = nodes.fillna("")
        nodes = nodes[
            ["id", "_label", "name", "code", "lineNumber", "controlStructureType"]
        ]

    with open(filepath, "r") as f:
        code = f.readlines()

    lmap = assign_line_num_to_local(nodes, edges, code)

    nodes.lineNumber = nodes.apply(
        lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    )

    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)

    nodes["node_label"] = (
        nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    nodes = nodes[nodes._label != "COMMENT"]
    nodes = nodes[nodes._label != "FILE"]

    edges = edges[edges.etype.isin(["AST", "CFG", "REACHING_DEF", "CDG"])]

    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )

    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )

    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    return nodes, edges


def full_run_joern(filepath: str, verbose=0):
    try:
        run_joern(filepath, verbose)
        nodes, edges = get_node_edges(filepath)
        return {"nodes": nodes, "edges": edges}
    except:
        return None


def build_triple_graph(filepath):
    nodes, edges = get_node_edges(filepath)

    nodesline = nodes[nodes.lineNumber != ""].copy()
    nodesline.lineNumber = nodesline.lineNumber.astype(int)

    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )

    nodesline.id = nodesline.lineNumber
    nodesline = nodesline.reset_index(drop=True).reset_index()

    node_map = pd.Series(nodesline.index.values, index=nodesline.id).to_dict()

    def build_edge_index(edge_df):
        if len(edge_df) == 0:
            return np.zeros((2, 0), dtype=np.int64)

        edge_df = edge_df.copy()
        edge_df.innode = edge_df.line_in
        edge_df.outnode = edge_df.line_out

        edge_df = edge_df[
            edge_df.innode.apply(lambda x: isinstance(x, float)) &
            edge_df.outnode.apply(lambda x: isinstance(x, float))
        ]

        edge_df.innode = edge_df.innode.map(node_map)
        edge_df.outnode = edge_df.outnode.map(node_map)
        edge_df = edge_df.dropna()

        return np.array([
            edge_df.outnode.tolist(),
            edge_df.innode.tolist()
        ])

    ast_edges = build_edge_index(edges[edges.etype == "AST"])
    cfg_edges = build_edge_index(edges[edges.etype == "CFG"])
    pdg_edges = build_edge_index(edges[edges.etype.isin(["REACHING_DEF", "CDG"])])

    return nodesline, ast_edges, cfg_edges, pdg_edges


def neighbour_nodes(nodes, edges, nodeids: list, hop: int = 1):
    nodes_new = (
        nodes.reset_index(drop=True).reset_index().rename(columns={"index": "adj"})
    )
    id2adj = pd.Series(nodes_new.adj.values, index=nodes_new.id).to_dict()
    adj2id = {v: k for k, v in id2adj.items()}

    arr = []
    for e in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj)):
        arr.append([e[0], e[1]])
        arr.append([e[1], e[0]])

    arr = np.array(arr)
    shape = tuple(arr.max(axis=0)[:2] + 1)
    coo = sparse.coo_matrix((np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=shape)

    csr = coo.tocsr()
    csr **= hop

    neighbours = defaultdict(list)
    for nodeid in nodeids:
        neighbours[nodeid] += [
            adj2id[i]
            for i in csr[id2adj[nodeid]].toarray()[0].nonzero()[0]
        ]

    return neighbours


def plot_graph_node_edge_df(nodes, edges, edge_label=True):
    dot = get_digraph(
        nodes[["id", "node_label"]].to_numpy().tolist(),
        edges[["outnode", "innode", "etype"]].to_numpy().tolist(),
        edge_label=edge_label,
    )
    dot.render("/tmp/tmp.gv", view=True)