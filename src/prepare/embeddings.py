import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import torch


class GraphsEmbedding:
    def __init__(self, edge_type):
        self.edge_type = edge_type.upper()

    def __call__(self, nodes_dict):
        edge_index = [[], []]
        
        mapping = {str(nid): i for i, nid in enumerate(nodes_dict.keys())}
        
        for node_id, node_obj in nodes_dict.items():
            # Use the raw_node we saved in Node.__init__
            node_raw_data = node_obj.raw_node 
            
            for edge in node_raw_data.get('edges', []):
                edge_id_full = edge['id'].lower()
                
                # Filter
                is_match = False
                if self.edge_type == "AST" and "ast" in edge_id_full:
                    is_match = True
                elif self.edge_type == "CFG" and "cfg" in edge_id_full:
                    is_match = True
                elif self.edge_type == "PDG":
                    if "reachingdef" in edge_id_full or "cdg" in edge_id_full:
                        is_match = True
                
                if is_match:
                    # These must be "6", "5", etc. to match the mapping keys
                    s_id = str(edge['out'])
                    t_id = str(edge['in'])
                    
                    if s_id in mapping and t_id in mapping:
                        edge_index[0].append(mapping[s_id])
                        edge_index[1].append(mapping[t_id])
                    else:
                        # DEBUG: See what we are missing
                        # print(f"DEBUG: s_id={s_id} (type {type(s_id)}), keys={list(mapping.keys())[:5]}")
                        pass
        return edge_index
                    

class NodesEmbedding:
    def __init__(self, nodes_dim, tokenizer, model, device="cpu"):
        self.nodes_dim = nodes_dim
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.kv_size = 768 

    def embed_nodes(self, nodes):
        embeddings = []
        # We use a simple list here to iterate, but let's keep it clean
        node_items = list(nodes.items())
        
        for n_id, node in node_items:
            node_code = node.get_code()
            # If code is empty, BERT will fail, so we provide a placeholder
            if not node_code or node_code.strip() == "":
                node_code = "empty"
                
            inputs = self.tokenizer(node_code, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use CLS token (index 0) instead of mean pooling for a slight speedup 
            # and better semantic representation for single nodes
            source_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            embedding = np.concatenate((np.array([node.type]), source_embedding), axis=0)
            embeddings.append(embedding)
        
        return torch.from_numpy(np.array(embeddings)).float()

    def __call__(self, nodes):
        target = torch.zeros(self.nodes_dim, self.kv_size + 1).float()
        embedded_nodes = self.embed_nodes(nodes)
        num_nodes = min(embedded_nodes.size(0), self.nodes_dim)
        target[:num_nodes, :] = embedded_nodes[:num_nodes, :]
        return target


def nodes_to_input(nodes, target, nodes_dim, nodes_embed_instance, graphs_embed_instance):
    """
    nodes: The dict {id: Node} from parse_to_nodes
    """
    # Create specific engines for each view
    ast_embed = GraphsEmbedding("Ast")
    cfg_embed = GraphsEmbedding("Cfg")
    pdg_embed = GraphsEmbedding("Pdg")
    
    # Generate the three distinct indices
    edge_ast = ast_embed(nodes)
    edge_cfg = cfg_embed(nodes)
    edge_pdg = pdg_embed(nodes)
    
    label = torch.tensor([target]).float()

    def to_tensor(edge_list):
        if len(edge_list[0]) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long)

    return Data(
        x=nodes_embed_instance(nodes), 
        edge_index_ast=to_tensor(edge_ast),
        edge_index_cfg=to_tensor(edge_cfg),
        edge_index_pdg=to_tensor(edge_pdg),
        y=label
    )