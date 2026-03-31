# -*- coding: utf-8 -*-
"""
    This module is intended to join all the pipeline in separated tasks
    to be executed individually or in a flow by using command-line options

    Example:
    Dataset embedding and processing:
        $ python taskflows.py -e -pS
"""

import argparse
import gc
import torch
from tqdm import tqdm
import shutil
from transformers import AutoTokenizer, AutoModel

import configs
import src.data as data
import src.prepare as prepare
import src.process as process
import src.utils.functions.cpg as cpg
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()

def select(dataset):
    result = dataset.loc[dataset.func.str.len() < 1200]
    vulnerable = result[result['target'] == 1]
    non_vulnerable = result[result['target'] == 0]
    min_count = min(len(vulnerable), len(non_vulnerable))
    vulnerable_sampled = vulnerable.sample(n=min_count, random_state=42)
    non_vulnerable_sampled = non_vulnerable.sample(n=min_count, random_state=42)
    
    print(f"Total available - Vuln: {len(vulnerable)}, Non-Vuln: {len(non_vulnerable)}")
    print(f"Selected {len(vulnerable_sampled)} vulnerable and {len(non_vulnerable_sampled)} non-vulnerable functions for a balanced dataset.")
    
    return pd.concat([vulnerable_sampled, non_vulnerable_sampled]).sample(frac=1, random_state=42)

def create_task():
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)
    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered)
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = prepare.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)
    # Create CPG with graphs json files
    json_files = prepare.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = prepare.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()


def embed_task():
    context = configs.Embed()
    dataset_files = data.get_directory_files(PATHS.cpg)
    
    # 1. Load CodeBERT once globally for the task
    print("Loading CodeBERT to memory...")
    import os
    os.environ['HF_HUB_TIMEOUT'] = '600'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", trust_remote_code=True)
    model = AutoModel.from_pretrained("microsoft/codebert-base", trust_remote_code=True).to(DEVICE)
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleared after loading model")
    
    # 2. Pre-instantiate the embedding engine (node-level)
    node_embed_instance = prepare.NodesEmbedding(tokenizer, model, DEVICE)
    
    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        
        # 3. Parse CPG to nodes (Structural parsing)
        print(f"Parsing CPG nodes for {file_name}...")
        cpg_dataset["nodes"] = cpg_dataset.apply(
            lambda row: cpg.parse_to_nodes(row.cpg, 100000), axis=1 # Large value to not truncate
        )
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        
        # 4. Embed using a loop with a progress bar
        print(f"Embedding {len(cpg_dataset)} functions with CodeBERT...")
        inputs = []
        total_ast_edges = 0
        total_cfg_edges = 0
        total_pdg_edges = 0
        
        # tqdm gives you a visual progress bar and ETA
        for index, row in tqdm(cpg_dataset.iterrows(), total=len(cpg_dataset), desc="Processing"):
            graph_data = prepare.nodes_to_input(
                row.nodes,
                row.target,
                node_embed_instance,
            )
            inputs.append(graph_data)
            total_ast_edges += graph_data.edge_index_ast.shape[1]
            total_cfg_edges += graph_data.edge_index_cfg.shape[1]
            total_pdg_edges += graph_data.edge_index_pdg.shape[1]
        
        num_functions = max(len(inputs), 1)
        print(f"[View Statistics] avg edges per function — "
              f"AST: {total_ast_edges/num_functions:.1f}, "
              f"CFG: {total_cfg_edges/num_functions:.1f}, "
              f"PDG: {total_pdg_edges/num_functions:.1f}")
        
        cpg_dataset["input"] = inputs
        
        # 5. Cleanup and Save
        data.drop(cpg_dataset, ["nodes"])
        print(f"Saving input dataset {file_name}...")
        data.write(cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}")
        
        del cpg_dataset
        gc.collect()

def split_task():
    """Create (or refresh) and save the global train/val/test split indices once."""
    print("Creating global train/val/test split...")
    data.global_train_val_test_split(PATHS.input, "data/")
    print("Split indices saved to data/split_indices.pkl.")


def process_task(use_early_stopping=False, evaluate_only=False):
    context = configs.Process()
    devign_configs = configs.Devign()
    embed_context = configs.Embed()
    feature_dim = embed_context.feature_dim

    model_obj = process.TripleViewNet(feature_dim=feature_dim, device=DEVICE)
    model_path = PATHS.model + FILES.model
    model = process.Devign(
        path=model_path, device=DEVICE, model=model_obj,
        learning_rate=context.learning_rate,
        weight_decay=context.weight_decay,
        weight_0=context.weight_0,
        weight_1=context.weight_1
    )
    model.accumulation_steps = context.accumulation_steps or 1

    # ── Global stratified split ──────────────────────────────────────────────
    # Loads ALL input files once and performs a SINGLE stratified 80/10/10
    # split across the entire dataset.  Split indices are cached in
    # data/split_indices.pkl so that training and evaluation always use the
    # exact same disjoint subsets.
    print("Loading global train/val/test split...")
    train_df, val_df, test_df = data.global_train_val_test_split(PATHS.input, "data/")

    # Compute class weights from the training set only
    weight_0, weight_1 = data.compute_class_weights(train_df)
    model.update_weights(weight_0, weight_1)

    # --- Model diagnostics ---
    total_params = sum(p.numel() for p in model_obj.parameters())
    trainable_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
    print(f"\n{'='*25} MODEL DIAGNOSTICS {'='*25}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    train_pos = int((train_df['target'] == 1).sum())
    train_neg = int((train_df['target'] == 0).sum())
    print(f"Train split: {len(train_df)} samples "
          f"(Pos={train_pos} [{100*train_pos/len(train_df):.1f}%], "
          f"Neg={train_neg} [{100*train_neg/len(train_df):.1f}%])")
    print(f"Val   split: {len(val_df)} samples")
    print(f"Test  split: {len(test_df)} samples")
    print(f"{'='*60}\n")

    # Build DataLoaders once – PyTorch resets the iterator each epoch automatically
    train_loader = data.InputDataset(train_df).get_loader(context.batch_size, shuffle=context.shuffle)
    val_loader   = data.InputDataset(val_df).get_loader(context.batch_size, shuffle=False)
    test_loader  = data.InputDataset(test_df).get_loader(context.batch_size, shuffle=False)

    if not evaluate_only:
        print(f"Starting Training for {context.epochs} epochs "
              f"(train={len(train_df)}, val={len(val_df)})...")
        trainer = process.Train(model, epochs=1, verbose=False)

        warmup_epochs = context.warmup_epochs or 0
        base_lr = context.learning_rate

        early_stopping = None
        if use_early_stopping:
            early_stopping = process.EarlyStopping(model, patience=context.patience, verbose=True)

        for epoch in range(context.epochs):
            print(f"\n{'='*20} Epoch {epoch+1}/{context.epochs} {'='*20}")

            # Linear LR warm-up
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"  LR warmup: {warmup_lr:.2e}")

            train_step = process.LoaderStep("Train", train_loader, DEVICE)
            val_step   = process.LoaderStep("Validation", val_loader, DEVICE)

            trainer(train_step, val_step, early_stopping=None, current_epoch=epoch + 1)

            # Retrieve per-epoch global metrics directly from LoaderStep
            train_loss = train_step.stats.loss()
            train_acc  = train_step.stats.acc()
            val_loss   = val_step.stats.loss()
            val_acc    = val_step.stats.acc()

            print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"  Val   → Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}")

            # Step the LR scheduler after warm-up
            if hasattr(model, 'scheduler') and epoch >= warmup_epochs:
                model.scheduler.step(val_loss)

            # Early stopping on global validation loss / manual checkpoint
            if use_early_stopping:
                if early_stopping(val_loss):
                    print(f"Early stopping triggered at Epoch {epoch + 1}.")
                    break
            else:
                model.save()

    else:
        model.load()

    # ── Final evaluation on the held-out test set ────────────────────────────
    print("\n" + "=" * 25 + " FINAL EVALUATION " + "=" * 25)
    print(f"Test Samples: {len(test_df)}")
    final_test_step = process.LoaderStep("Final Test", test_loader, DEVICE)
    process.predict(model, final_test_step)
        
def main():
    """
    main function that executes tasks based on command-line options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--create', action='store_true', help='Create dataset')
    parser.add_argument('-e', '--embed', action='store_true', help='Embed dataset')
    parser.add_argument('-S', '--split', action='store_true',
                        help='Create/refresh global train/val/test split indices')
    parser.add_argument('-p', '--process', action='store_true', help='Standard training')
    parser.add_argument('-s', '--stopping', action='store_true', help='Training with early stopping')
    parser.add_argument('-v', '--eval', action='store_true', help='Evaluation only mode')

    args = parser.parse_args()

    if args.create:
        create_task()
    if args.embed:
        embed_task()
    if args.split:
        split_task()

    # Standard training (No early stopping)
    if args.process:
        process_task(use_early_stopping=False, evaluate_only=False)

    # Training WITH early stopping
    elif args.stopping:
        process_task(use_early_stopping=True, evaluate_only=False)

    # Evaluation Only
    elif args.eval:
        process_task(evaluate_only=True)


if __name__ == "__main__":
    main()
