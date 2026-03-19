import torch
import os
import shutil
import json
import re
from src.prepare import cpg_generator, embeddings
from src.process.model import TripleViewNet
from src.utils.objects.cpg.function import Function
from transformers import AutoTokenizer, AutoModel

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# --- CONFIGURATION ---
JOERN_PATH = "joern/joern-cli/"
MODEL_PATH = "data/model/checkpoint.pt"
TEMP_DIR = "data/demo/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_single_file(c_file_path):
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
    
    # 1. JOERN PARSE (C -> BIN)
    print(f"[*] Parsing {c_file_path}...")
    bin_file = cpg_generator.joern_parse(JOERN_PATH, c_file_path, TEMP_DIR, "demo_func")
    
    # 2. JOERN CREATE (BIN -> JSON)
    print("[*] Exporting Graph JSON...")
    json_files = cpg_generator.joern_create(JOERN_PATH, TEMP_DIR, TEMP_DIR, [bin_file])
    
    # 3. PROCESS JSON TO NODES
    print("[*] Processing Graph Structure...")
    with open(os.path.join(TEMP_DIR, json_files[0])) as jf:
        cpg_json = json.loads(re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', jf.read()))

    # 3. PROCESS JSON TO NODES (Robust Version)
    valid_funcs = []
    for f in cpg_json["functions"]:
        nodes = f.get("AST", []) # Use AST to check for actual code content
        
        # LOGICAL CHECKS:
        # 1. Must have more than 3 nodes (entry, exit, and at least some logic)
        # 2. Must NOT be a virtual operator (contains <operator>)
        # 3. Must NOT be the global file wrapper
        name = f["function"]
        is_virtual = "<operator>" in name or name.endswith("<global>") or name == "ANY"
        has_content = len(nodes) > 5 

        if not is_virtual and has_content:
            valid_funcs.append(f)

    # Sort by node count - test the largest function
    valid_funcs.sort(key=lambda x: len(x.get("AST", [])), reverse=True)

    if not valid_funcs:
        print("[-] Error: No valid user-defined functions found.")
        return

    target_func = valid_funcs[0]

    if not target_func:
        print("[-] Warning: No explicit function found. Falling back to first available.")
        target_func = cpg_json["functions"][0]

    print(f"[*] Analyzing function: {target_func['function']}")
    func_obj = Function(target_func)
    nodes = func_obj.get_nodes()

    # 4. GENERATE EMBEDDINGS (CodeBERT)
    print("[*] Generating CodeBERT Embeddings...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(DEVICE)
    
    # Initialize your specific embedding classes
    nodes_embedder = embeddings.NodesEmbedding(nodes_dim=200, tokenizer=tokenizer, model=codebert, device=DEVICE)
    
    # Convert to PyG Data object
    # We use target=0 as a placeholder since we are predicting
    pyg_data = embeddings.nodes_to_input(nodes, 0, 200, nodes_embedder, None)

# 5. INFERENCE
    print("[*] Running Model Inference...")
    model = TripleViewNet(feature_dim=769, device=DEVICE)
    
    # Using weights_only=True to silence the FutureWarning
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False) 
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        # Ensure the data is on the correct device
        pyg_data = pyg_data.to(DEVICE)
        
        # Manually create the batch attribute for GlobalAttention/Pooling layers
        # Since it's a single graph, every node belongs to batch 0
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long).to(DEVICE)
        
        logits = model(pyg_data)
        probability = torch.sigmoid(logits).item()

    # Clear terminal line for clean output
    print("\n" + "="*40)
    print(f"{'VULNERABILITY REPORT':^40}")
    print("="*40)
    print(f"Target Function : {target_func['function']}")
    print(f"Source File     : {os.path.basename(c_file_path)}")
    print(f"Prediction      : {'[!] VULNERABLE' if probability > 0.5 else '[+] CLEAN'}")
    print(f"Confidence      : {probability:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict_single_file(sys.argv[1])
    else:
        print("Usage: python predict.py path/to/code.c")