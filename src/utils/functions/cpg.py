from collections import OrderedDict
from ..objects.cpg.function import Function
from ..objects.cpg.node import Node

def order_nodes(nodes, max_nodes):
    # Sort for consistency, but DO NOT CUT during debugging
    nodes_by_column = sorted(nodes.items(), key=lambda n: int(n[1].get_column_number() or 0))
    nodes_by_line = sorted(nodes_by_column, key=lambda n: int(n[1].get_line_number() or 0))

    for i, node in enumerate(nodes_by_line):
        node[1].order = i

    # Increase max_nodes to a very high number (e.g., 5000) 
    # so you don't lose structural nodes.
    if len(nodes) > max_nodes:
        print(f"WARNING: CPG cut from {len(nodes)} to {max_nodes}. This will break edges.")
        return OrderedDict(nodes_by_line[:max_nodes])

    return OrderedDict(nodes_by_line)

def filter_nodes(nodes):
    return {n_id: node for n_id, node in nodes.items() if node.has_code() and
            node.has_line_number() and
            node.label not in ["Comment", "Unknown"]}
    #return nodes
    #return {k: v for k, v in nodes.items() if v.get_code() is not None}

def parse_to_nodes(cpg, max_nodes=2000): # Raised limit
    nodes = {}
    for function in cpg["functions"]:
        # 1. Add the root function node first
        root_node = Node(function, 0)
        nodes[str(root_node.id)] = root_node
        
        # 2. Add the AST nodes
        func = Function(function)
        all_nodes = func.get_nodes()
        for nid, nobj in all_nodes.items():
            nodes[str(nid)] = nobj

        # 3. Add CFG nodes that are not already present from the AST
        # Indentation depth 1 matches the depth used for AST child nodes
        # (Function.indentation=1 is what AST.__init__ receives).
        for node_data in function.get("CFG", []):
            nid = str(node_data["id"]).split(".")[-1]
            if nid not in nodes:
                nodes[nid] = Node(node_data, 1)

        # 4. Add PDG nodes that are not already present from the AST or CFG
        for node_data in function.get("PDG", []):
            nid = str(node_data["id"]).split(".")[-1]
            if nid not in nodes:
                nodes[nid] = Node(node_data, 1)

    return order_nodes(nodes, max_nodes)