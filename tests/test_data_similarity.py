import torch
from ..src.data import datamanager as data

train_df, _, _ = data.global_train_val_test_split("data/input/", "data/")

print("Train split composition:")
print(f"  Total: {len(train_df)}")
print(f"  Class 0: {(train_df['target'] == 0).sum()} ({100*(train_df['target'] == 0).sum()/len(train_df):.1f}%)")
print(f"  Class 1: {(train_df['target'] == 1).sum()} ({100*(train_df['target'] == 1).sum()/len(train_df):.1f}%)")

# Check if they're actually DIFFERENT or just randomly shuffled
pos_graphs = train_df[train_df['target'] == 1]['input'].head(5)
neg_graphs = train_df[train_df['target'] == 0]['input'].head(5)

print("\nSample graph sizes:")
for i, (name, graphs) in enumerate([("Positive", pos_graphs), ("Negative", neg_graphs)]):
    print(f"\n{name}:")
    for idx, g in enumerate(graphs):
        print(f"  Graph {idx}: nodes={g.x.shape[0]}, ast_edges={g.edge_index_ast.shape[1]}, cfg_edges={g.edge_index_cfg.shape[1]}, pdg_edges={g.edge_index_pdg.shape[1]}")

# Most important: check if embedding MEAN is different
print("\n\nEmbedding statistics (mean across all samples):")
pos_embeddings = []
neg_embeddings = []

for idx in range(min(100, len(train_df))):
    emb = train_df['input'].iloc[idx].x.cpu().numpy().mean(axis=0)
    if train_df['target'].iloc[idx] == 1:
        pos_embeddings.append(emb)
    else:
        neg_embeddings.append(emb)

import numpy as np
pos_embeddings = np.array(pos_embeddings)
neg_embeddings = np.array(neg_embeddings)

print(f"Positive - mean embedding: {pos_embeddings.mean():.6f} ± {pos_embeddings.std():.6f}")
print(f"Negative - mean embedding: {neg_embeddings.mean():.6f} ± {neg_embeddings.std():.6f}")

# Check if there's actual pattern difference
from sklearn.metrics.pairwise import cosine_similarity
pos_mean = pos_embeddings.mean(axis=0)
neg_mean = neg_embeddings.mean(axis=0)
similarity = cosine_similarity([pos_mean], [neg_mean])[0][0]
print(f"\nCosine similarity between class centroids: {similarity:.4f}")

if similarity > 0.95:
    print("⚠️ CRITICAL: Classes are nearly identical! Data may be mislabeled or identical.")
elif similarity < 0.5:
    print("✅ Classes are sufficiently different")
else:
    print("⚠️ Classes are similar but distinguishable")
