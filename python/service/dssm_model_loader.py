"""
DSSM 模型加载器：Query 向量服务
"""
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

QUERY_COLS = ["query_f1", "query_f2", "query_f3"]
EMBED_DIM = 64
TOWER_LAYERS = [128, 64]


class DSSM(nn.Module):
    """与 train_dssm 一致：Query 塔 + Doc 塔，此处仅用 Query 塔。"""

    def __init__(self, dim_in: int, embed_dim: int, layers: list[int]):
        super().__init__()
        self.embed_dim = embed_dim

        def mlp(dims):
            mods = []
            for i in range(len(dims) - 1):
                mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(0.2)]
            return nn.Sequential(*mods)

        dims_q = [dim_in] + layers + [embed_dim]
        dims_d = [dim_in] + layers + [embed_dim]
        self.query_tower = mlp(dims_q)
        self.doc_tower = mlp(dims_d)

    def query_embedding(self, q):
        qe = self.query_tower(q)
        return torch.nn.functional.normalize(qe, p=2, dim=1)


class DSSMModelLoader:
    def __init__(self, model_path: str, meta_path: str):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.query_cols = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"DSSM 模型或元数据不存在: {self.model_path} / {self.meta_path}")
        with open(self.meta_path) as f:
            meta = json.load(f)
        self.query_cols = meta.get("query_cols", QUERY_COLS)
        ckpt = torch.load(self.model_path, map_location=self.device)
        cfg = ckpt["config"]
        self.model = DSSM(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        logger.info("DSSM 模型加载成功: %s", self.model_path)

    def get_query_embedding(self, query_features: dict) -> list[float]:
        if self.model is None:
            raise RuntimeError("DSSM 未加载")
        vec = [float(query_features.get(c, 0.0)) for c in self.query_cols]
        x = torch.FloatTensor([vec]).to(self.device)
        with torch.no_grad():
            emb = self.model.query_embedding(x)
        return emb[0].cpu().numpy().tolist()
