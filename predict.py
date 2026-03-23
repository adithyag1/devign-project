import torch
import os
import json
import re
from src.prepare import cpg_generator, embeddings
from src.process.model import TripleViewNet
from src.utils.objects.cpg.function import Function
from transformers import AutoTokenizer, AutoModel
from cfexplainer_wrapper import (
    ExplanationExtractor,
    print_explanation_report,
    save_explanation_report,
)

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# --- CONFIGURATION ---
JOERN_PATH = "joern/joern-cli/"
MODEL_PATH = "data/model/checkpoint.pt"
TEMP_DIR = "data/demo/"
REPORT_PATH = "data/demo/explanation_report.json"
CFEXPLAINER_EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_single_file(c_file_path):
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

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

    valid_funcs = []
    for f in cpg_json["functions"]:
        nodes = f.get("AST", [])
        name = f["function"]
        is_virtual = "<operator>" in name or name.endswith("<global>") or name == "ANY"
        has_content = len(nodes) > 5

        if not is_virtual and has_content:
            valid_funcs.append(f)

    # Sort by node count - analyse the largest user-defined function
    valid_funcs.sort(key=lambda x: len(x.get("AST", [])), reverse=True)

    if not valid_funcs:
        print("[-] Error: No valid user-defined functions found.")
        return

    target_func = valid_funcs[0]
    print(f"[*] Analyzing function: {target_func['function']}")

    func_obj = Function(target_func)
    nodes = func_obj.get_nodes()

    # 4. GENERATE EMBEDDINGS (CodeBERT) + build PyG Data with triple edge views
    print("[*] Generating CodeBERT Embeddings...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(DEVICE)

    nodes_embedder = embeddings.NodesEmbedding(
        nodes_dim=200, tokenizer=tokenizer, model=codebert, device=DEVICE
    )

    # nodes_to_input returns Data(x, edge_index_ast, edge_index_cfg, edge_index_pdg, y)
    pyg_data = embeddings.nodes_to_input(nodes, 0, 200, nodes_embedder)

    # 5. LOAD MODEL AND RUN INFERENCE
    print("[*] Running Model Inference...")
    model = TripleViewNet(feature_dim=769, device=DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        pyg_data = pyg_data.to(DEVICE)
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long).to(DEVICE)

        logits = model(pyg_data)
        probability = torch.sigmoid(logits).item()

    pred_label = 1 if probability > 0.5 else 0

    # 6. CFExplainer – extract explanatory subgraph with code snippets
    print("[*] Extracting Explanation with CFExplainer...")
    extractor = ExplanationExtractor(model=model, device=DEVICE, epochs=CFEXPLAINER_EPOCHS, top_k=8)
    explanation = extractor.get_explanation_with_code(pyg_data, nodes, pred_label)
    explanation["target_label"] = pred_label

    # 7. OUTPUT – human-readable report
    print_explanation_report(
        func_name=target_func["function"],
        source_file=c_file_path,
        probability=probability,
        explanation=explanation,
    )

    # 8. SAVE JSON report
    save_explanation_report(
        output_path=REPORT_PATH,
        func_name=target_func["function"],
        source_file=c_file_path,
        probability=probability,
        explanation=explanation,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict_single_file(sys.argv[1])
    else:
        print("Usage: python predict.py path/to/code.c")