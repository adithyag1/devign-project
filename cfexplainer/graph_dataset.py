import sys, json, os
import os.path as osp
from typing import Callable, Optional, Union, List, Tuple
import pickle as pkl
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data, Batch
from tqdm.std import trange
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizer,
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
    DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
    T5Config, T5ForConditionalGeneration, T5Tokenizer
)

from helpers import utils
from helpers import joern
from data_pre import bigvul


class VulGraphDataset(Dataset):
    def __init__(self, root: Optional[str] = "storage/processed/vul_graph_dataset",
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, log: bool = True,
                 encoder=None, tokenizer=None, partition=None,
                 vulonly=False, sample=-1, splits="default"):

        os.makedirs(root, exist_ok=True)

        self.encoder = encoder
        self.word_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.detach().cpu().numpy() if self.encoder is not None else None
        self.tokenizer = tokenizer
        self.partition = partition
        self.vulonly = vulonly
        self.sample = sample
        self.splits = splits

        super().__init__(root, transform, pre_transform, pre_filter, log)

        self.data_list = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'{self.partition}_processed')

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    def process(self):
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(utils.processed_dir() / "bigvul/before/*nodes*"))
        ]

        self.df = bigvul(splits=self.splits)
        self.df = self.df[self.df.label == self.partition]
        self.df = self.df[self.df.id.isin(self.finished)]

        vul = self.df[self.df.vul == 1]
        nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
        self.df = pd.concat([vul, nonvul])

        if self.sample > 0:
            self.df = self.df.sample(self.sample, random_state=0)

        if self.vulonly:
            self.df = self.df[self.df.vul == 1]

        self.df["valid"] = utils.dfmp(
            self.df, VulGraphDataset.check_validity, "id", desc="Validate Samples: "
        )
        self.df = self.df[self.df.valid]

        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        data_list = []
        for idx in trange(self.df.shape[0]):
            _id = self.idx2id[idx]

            result = self.feature_extraction(VulGraphDataset.itempath(_id))
            if result is None:
                continue

            n, ast_e, cfg_e, pdg_e = result

            x = np.array(list(n.subseq_feat.values))

            data = Data(
                x=torch.FloatTensor(x),
                edge_index_ast=torch.LongTensor(ast_e),
                edge_index_cfg=torch.LongTensor(cfg_e),
                edge_index_pdg=torch.LongTensor(pdg_e)
            )

            n["vuln"] = n.id.map(self.get_vuln_indices(_id)).fillna(0)

            data.__setitem__("_VULN", torch.Tensor(n["vuln"].astype(int).to_numpy()))
            data.__setitem__("_LINE", torch.Tensor(n["id"].astype(int).to_numpy()))
            data.__setitem__("_SAMPLE", torch.Tensor([_id] * len(n)))

            data_list.append(data)

        print('Saving...')
        torch.save(data_list, self.processed_paths[0])

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    def itempath(_id):
        return utils.processed_dir() / f"bigvul/before/{_id}.c"

    def check_validity(_id):
        valid = 0
        try:
            with open(str(VulGraphDataset.itempath(_id)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n:
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:
                            valid = 1
                            break
                if valid == 0:
                    return False

            with open(str(VulGraphDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception:
            return False

    def get_vuln_indices(self, _id):
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])

    def feature_extraction(self, filepath):
        cache_name = "_".join(str(filepath).split("/")[-3:])
        cachefp = utils.get_dir(utils.cache_dir() / "vul_graph_feat") / Path(cache_name).stem

        if os.path.exists(cachefp):
            with open(cachefp, "rb") as f:
                return pkl.load(f)

        try:
            nodes, edges = joern.get_node_edges(filepath)
        except:
            return None

        edges_ast = edges[edges.etype == "AST"].copy()
        edges_cfg = edges[edges.etype == "CFG"].copy()
        edges_pdg = edges[edges.etype.isin(["REACHING_DEF", "CDG"])].copy()

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

        ast_edges = build_edge_index(edges_ast)
        cfg_edges = build_edge_index(edges_cfg)
        pdg_edges = build_edge_index(edges_pdg)

        # Feature extraction
        subseq = (
            nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
            .groupby("lineNumber")
            .head(1)
        )

        subseq = subseq[["lineNumber", "code", "local_type"]].copy()
        subseq.code = subseq.local_type + " " + subseq.code
        subseq = subseq.drop(columns="local_type")
        subseq = subseq[~subseq.eq("").any(axis='columns')]
        subseq = subseq[subseq.code != " "]
        subseq = subseq[subseq.code.notnull()]
        subseq.lineNumber = subseq.lineNumber.astype(int)
        subseq = subseq.sort_values("lineNumber")

        subseq.code = subseq.code.apply(lambda s: ' '.join(s.split()))
        subseq.code = subseq.code.apply(
            lambda s: [self.tokenizer.cls_token] + self.tokenizer.tokenize(s) + [self.tokenizer.sep_token]
        )

        subseq["code_feat"] = subseq.code.apply(
            lambda token_ids: self.tokenizer.convert_tokens_to_ids(token_ids)
        )

        subseq.code_feat = subseq.code_feat.apply(
            lambda token_ids: np.mean(self.word_embeddings[token_ids], axis=0)
        )

        subseq_feat = subseq.set_index("lineNumber").to_dict()["code_feat"]

        pdg_nodes = nodesline.copy()[["id"]].sort_values("id")
        pdg_nodes["subseq_feat"] = pdg_nodes.id.map(subseq_feat).fillna(0)

        with open(cachefp, "wb") as f:
            pkl.dump([pdg_nodes, ast_edges, cfg_edges, pdg_edges], f)

        return pdg_nodes, ast_edges, cfg_edges, pdg_edges


def collate(data_list):
    return Batch.from_data_list(data_list)


if __name__ == '__main__':
    MODEL_CLASSES = {
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }

    model_type = "roberta"
    model_name_or_path = "microsoft/graphcodebert-base"
    tokenizer_name = "microsoft/graphcodebert-base"

    partition = sys.argv[1]

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    language_model = model_class.from_pretrained(model_name_or_path, config=config)

    dataset = VulGraphDataset(
        root=str(utils.processed_dir() / "vul_graph_dataset"),
        encoder=language_model,
        tokenizer=tokenizer,
        partition=partition
    )

    print(dataset)
    print(dataset.data_list[0])
    print(dataset.data_list[0].edge_index_ast)
    print(dataset.data_list[0].edge_index_cfg)
    print(dataset.data_list[0].edge_index_pdg)