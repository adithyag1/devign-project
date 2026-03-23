import glob
import pickle

import pandas as pd
import numpy as np
import os
import src.utils.functions.parse as parse

from os import listdir
from os.path import isfile, join
from src.utils.objects.input_dataset import InputDataset
from sklearn.model_selection import train_test_split


def read(path, json_file):
    """
    :param path: str
    :param json_file: str
    :return DataFrame
    """
    return pd.read_json(path + json_file)


def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]


def load(path, pickle_file, ratio=1):
    dataset = pd.read_pickle(path + pickle_file)
   # dataset.info(memory_usage='deep')
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)

    return dataset


def write(data_frame: pd.DataFrame, path, file_name):
    data_frame.to_pickle(path + file_name)


def apply_filter(data_frame: pd.DataFrame, filter_func):
    return filter_func(data_frame)


def rename(data_frame: pd.DataFrame, old, new):
    return data_frame.rename(columns={old: new})


def tokenize(data_frame: pd.DataFrame):
    data_frame.func = data_frame.func.apply(parse.tokenizer)
    # Change column name
    data_frame = rename(data_frame, 'func', 'tokens')
    # Keep just the tokens
    return data_frame[["tokens"]]


def to_files(data_frame: pd.DataFrame, out_path):
    # path = f"{self.out_path}/{self.dataset_name}/"
    os.makedirs(out_path, exist_ok = True)

    for idx, row in data_frame.iterrows():
        file_name = f"{idx}.c"
        with open(out_path + file_name, 'w') as f:
            f.write(row.func)


def create_with_index(data, columns):
    data_frame = pd.DataFrame(data, columns=columns)
    data_frame.index = list(data_frame["Index"])

    return data_frame


def inner_join_by_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


def train_val_test_split(data_frame: pd.DataFrame, shuffle=True):
    #print("Splitting Dataset")

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]

    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    test_false, val_false = train_test_split(test_false, test_size=0.5, shuffle=shuffle)
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)
    test_true, val_true = train_test_split(test_true, test_size=0.5, shuffle=shuffle)

    train = pd.concat([train_false,train_true])
    val = pd.concat([val_false,val_true])
    test = pd.concat([test_false,test_true])

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return InputDataset(train), InputDataset(test), InputDataset(val)


def _compute_split_from_targets(targets, test_size=0.2, val_size=0.1, random_state=42):
    """Compute split using only target array (no full dataframe needed)."""
    indices = np.arange(len(targets))
    
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=targets, random_state=random_state
    )
    
    val_fraction = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_fraction,
        stratify=targets[train_val_idx], random_state=random_state
    )
    
    return train_idx, val_idx, test_idx

def global_train_val_test_split(input_dir, split_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load ALL input files globally, perform ONE stratified split, and cache the
    indices to *split_path/split_indices.pkl* for full reproducibility.

    Returns
    -------
    (train_df, val_df, test_df) – disjoint DataFrames with 'input' and 'target' columns.
    """
    os.makedirs(split_path, exist_ok=True)
    split_file = os.path.join(split_path, "split_indices.pkl")

    # Load and concatenate all input files (sorted for determinism)
    all_files = sorted(get_directory_files(input_dir))
    if not all_files:
        raise ValueError(f"No input files found in {input_dir}")

    # loads files incrementally to save RAM
    n_total = 0
    all_targets = []
    all_indices = []
    current_idx = 0
    
    for f in all_files:
        df = load(input_dir, f)
        n_total += len(df)
        all_targets.extend(df['target'].values)
        all_indices.extend([current_idx + i for i in range(len(df))])
        current_idx += len(df)
        del df  # Free memory immediately
    
    full_targets = np.array(all_targets)

    # Load cached split indices or recompute
    if os.path.exists(split_file):
        with open(split_file, 'rb') as fh:
            split_data = pickle.load(fh)

        if split_data.get('total_samples') == n_total:
            print(f"Loaded cached split indices ({n_total} samples).")
            train_idx = split_data['train']
            val_idx = split_data['val']
            test_idx = split_data['test']
        else:
            print(
                f"Dataset size changed "
                f"({split_data.get('total_samples')} → {n_total}), recomputing split..."
            )
            train_idx, val_idx, test_idx = _compute_split_from_targets(
                full_targets, test_size, val_size, random_state
            )
            with open(split_file, 'wb') as fh:
                pickle.dump(
                    {'total_samples': n_total, 'train': train_idx,
                     'val': val_idx, 'test': test_idx},
                    fh
                )
    else:
        print(f"Computing new global split for {n_total} samples...")
        train_idx, val_idx, test_idx = _compute_split_from_targets(
            full_targets, test_size, val_size, random_state
        )
        with open(split_file, 'wb') as fh:
            pickle.dump(
                {'total_samples': n_total, 'train': train_idx,
                 'val': val_idx, 'test': test_idx},
                fh
            )

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)

    for label, df in [("Train", train_df), ("Val  ", val_df), ("Test ", test_df)]:
        pos = int((df['target'] == 1).sum())
        print(f"  {label}: {len(df)} (pos={pos}, neg={len(df) - pos})")

    return train_df, val_df, test_df


def compute_class_weights(data_frame: pd.DataFrame):
    """Return (weight_0, weight_1) balanced class weights for a training DataFrame."""
    class_counts = data_frame['target'].value_counts()
    total = len(data_frame)
    weight_0 = total / (2.0 * max(int(class_counts.get(0, 1)), 1))
    weight_1 = total / (2.0 * max(int(class_counts.get(1, 1)), 1))
    return weight_0, weight_1


def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]


def loads(data_sets_dir, ratio=1):
    data_sets_files = sorted([f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f))])

    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)

    dataset = load(data_sets_dir, data_sets_files[0])
    data_sets_files.remove(data_sets_files[0])

    for ds_file in data_sets_files:
        dataset = pd.concat([dataset,load(data_sets_dir, ds_file)])

    return dataset


def clean(data_frame: pd.DataFrame):
    return data_frame.drop_duplicates(subset="func", keep=False)


def drop(data_frame: pd.DataFrame, keys):
    for key in keys:
        del data_frame[key]


def slice_frame(data_frame: pd.DataFrame, size: int):
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)
