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
import random
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
    result = dataset.loc[dataset['project'] == "FFmpeg"]
    result = result.loc[result.func.str.len() < 1200]
    
    # Use .sample instead of .head to get a diverse variety of code
    vulnerable = result[result['target'] == 1].sample(n=min(len(result[result['target'] == 1]), 500), random_state=42)
    non_vulnerable = result[result['target'] == 0].sample(n=min(len(result[result['target'] == 0]), 500), random_state=42)
    
    print(f"Selected {len(vulnerable)} vulnerable and {len(non_vulnerable)} non-vulnerable functions.")
    return pd.concat([vulnerable, non_vulnerable]).sample(frac=1)

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
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(DEVICE)
    model.eval() 
    
    # 2. Pre-instantiate the embedding engines
    nodes_embed_instance = prepare.NodesEmbedding(context.nodes_dim, tokenizer, model, DEVICE)
    graphs_embed_instance = prepare.GraphsEmbedding(context.edge_type)

    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        
        # 3. Parse CPG to nodes (Structural parsing)
        print(f"Parsing CPG nodes for {file_name}...")
        cpg_dataset["nodes"] = cpg_dataset.apply(
            lambda row: cpg.parse_to_nodes(row.cpg, context.nodes_dim), axis=1
        )
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        
        # 4. Embed using a loop with a progress bar
        print(f"Embedding {len(cpg_dataset)} functions with CodeBERT...")
        inputs = []
        
        # tqdm gives you a visual progress bar and ETA
        for index, row in tqdm(cpg_dataset.iterrows(), total=len(cpg_dataset), desc="Processing"):
            graph_data = prepare.nodes_to_input(
                row.nodes, 
                row.target, 
                context.nodes_dim, 
                nodes_embed_instance, 
                graphs_embed_instance
            )
            inputs.append(graph_data)
        
        cpg_dataset["input"] = inputs
        
        # 5. Cleanup and Save
        data.drop(cpg_dataset, ["nodes"])
        print(f"Saving input dataset {file_name}...")
        data.write(cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}")
        
        del cpg_dataset
        gc.collect()

def process_task(use_early_stopping=False, evaluate_only=False):
    context = configs.Process()
    devign_configs = configs.Devign()
    feature_dim = 769 
    
    model_obj = process.TripleViewNet(feature_dim=feature_dim, device=DEVICE)
    model_path = PATHS.model + FILES.model
    model = process.Devign(
        path=model_path, device=DEVICE, model=model_obj,
        learning_rate=devign_configs.learning_rate,
        weight_decay=devign_configs.weight_decay,
        loss_lambda=devign_configs.loss_lambda,
        weight_0=devign_configs.weight_0,
        weight_1=devign_configs.weight_1
    )
    
    input_files = data.get_directory_files(PATHS.input)

    if not evaluate_only:
        print(f"Starting Sequential Training for {context.epochs} epochs...")
        trainer = process.Train(model, epochs=1, verbose=False) 
        
        early_stopping = None
        if use_early_stopping:
            # We set verbose=True here so the Bjarten class prints when it saves
            early_stopping = process.EarlyStopping(model, patience=context.patience, verbose=True)
        
        for epoch in range(context.epochs):
            print(f"\n{'='*20} Epoch {epoch+1}/{context.epochs} {'='*20}")
            random.shuffle(input_files)
            
            # New: Track losses for the whole epoch
            epoch_val_losses = []

            for f in input_files:
                chunk_df = data.load(PATHS.input, f).reset_index(drop=True)
                splits = data.train_val_test_split(chunk_df, shuffle=context.shuffle)
                
                # train=0, test=1, val=2
                train_loader, _, val_loader = [
                    x.get_loader(context.batch_size, shuffle=context.shuffle) for x in splits
                ]
                
                train_step = process.LoaderStep("Train", train_loader, DEVICE)
                val_step = process.LoaderStep("Validation", val_loader, DEVICE)

                # IMPORTANT: Pass early_stopping=None here so it doesn't stop inside a chunk
                trainer(train_step, val_step, early_stopping=None, current_epoch=epoch+1)
                
                # Get the val loss from this chunk
                val_stats = trainer.history.current()[0]
                epoch_val_losses.append(val_stats.loss())
                
                print(f" Chunk {f}: {trainer.history}")
                del chunk_df, train_loader, val_loader
                gc.collect()

            # 1. Calculate average epoch loss
            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)

            # 2. Step the scheduler if it exists
            if hasattr(model, 'scheduler'):
                model.scheduler.step(avg_val_loss)

            # 3. Handle Early Stopping vs Manual Save
            if use_early_stopping:
                # The early_stopping call usually saves the best model automatically
                if early_stopping(avg_val_loss):
                    print(f"Early stopping limit reached at Epoch {epoch+1}. Ending training.")
                    break
            else:
                # If not using early stopping, save the latest progress manually
                model.save()

    else:
        model.load()

# --- BRANCH 3: FINAL EVALUATION ---
    print("\n" + "="*25 + " FINAL EVALUATION (Aggregated) " + "="*25)    
    
    all_test_graphs = []
    eval_files = data.get_directory_files(PATHS.input)

    for f in eval_files:
        chunk_df = data.load(PATHS.input, f).reset_index(drop=True)
        
        # 1. Split the DataFrame (returns InputDataset objects)
        # Order: train, test, val
        splits = data.train_val_test_split(chunk_df, shuffle=False)
        test_dataset_obj = splits[1] 
        
        # 2. REACH INTO THE DATAFRAME INSIDE THE OBJECT
        # Most 'InputDataset' implementations store the DF in self.df or self.dataset
        # Since the error says it found a DataFrame, we need to get the 'input' column
        if hasattr(test_dataset_obj, 'df'):
            target_df = test_dataset_obj.df
        elif hasattr(test_dataset_obj, 'dataset') and isinstance(test_dataset_obj.dataset, pd.DataFrame):
            target_df = test_dataset_obj.dataset
        else:
            # If the object itself is not the DF, it might be the graphs 
            # but clearly, something is passing a DF to the loader.
            target_df = chunk_df # Fallback to the chunk itself if split fails

        # 3. EXTRACT THE GRAPH OBJECTS FROM THE 'input' COLUMN
        if 'input' in target_df.columns:
            # This extracts only the PyG Data objects (the graphs)
            graphs = target_df['input'].tolist()
            all_test_graphs.extend(graphs)

    print(f"Total Test Samples Collected: {len(all_test_graphs)}")

    # 4. Use PyG DataLoader on the FLAT LIST of Graphs
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    if len(all_test_graphs) > 0:
        final_loader = PyGDataLoader(all_test_graphs, batch_size=context.batch_size, shuffle=False)
        final_test_step = process.LoaderStep("Final Test", final_loader, DEVICE)

        # 5. Predict
        process.predict(model, final_test_step)
    else:
        print("Error: No graphs found in the 'input' column of the test files.")
        
def main():
    """
    main function that executes tasks based on command-line options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--create', action='store_true', help='Create dataset')
    parser.add_argument('-e', '--embed', action='store_true', help='Embed dataset')
    parser.add_argument('-p', '--process', action='store_true', help='Standard training')
    parser.add_argument('-s', '--stopping', action='store_true', help='Training with early stopping')
    parser.add_argument('-v', '--eval', action='store_true', help='Evaluation only mode')

    args = parser.parse_args()

    if args.create:
        create_task()
    if args.embed:
        embed_task()
        
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
