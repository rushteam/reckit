"""
双塔模型加载器：User Embedding / Item Embedding 推理

供 Golang TwoTowerRecall 调用：传入用户特征字典，返回 User Embedding（[]float64）。
也可用于离线批量计算 Item Embedding 写入向量库。

字段与 train_two_tower.py 保持一致：
- user_cols / item_cols / embed_dim：从 two_tower_meta.json 读取（与训练脚本写入一致）
- user_dim / item_dim / user_layers / item_layers：从 checkpoint["config"] 读取（训练时保存）
"""
from __future__ import annotations

import json
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 与 train_two_tower.py 中 USER_COLS / ITEM_COLS / EMBED_DIM 一致，仅作 meta 缺失时的默认值
USER_COLS_DEFAULT = ["user_age", "user_gender"]
ITEM_COLS_DEFAULT = ["item_ctr", "item_cvr", "item_price"]
EMBED_DIM_DEFAULT = 64
TOWER_LAYERS_DEFAULT = [128, 64]


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Module:
    mods = []
    for i in range(len(dims) - 1):
        mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*mods)


class TwoTower(nn.Module):
    """与 train_two_tower 一致：User Tower + Item Tower，L2 归一化。"""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        embed_dim: int,
        user_layers: list[int],
        item_layers: list[int],
    ):
        super().__init__()
        self.embed_dim = embed_dim
        dims_u = [user_dim] + user_layers + [embed_dim]
        dims_i = [item_dim] + item_layers + [embed_dim]
        self.user_tower = _mlp(dims_u, dropout=0.0)
        self.item_tower = _mlp(dims_i, dropout=0.0)

    def user_embedding(self, user_f: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_tower(user_f), p=2, dim=1)

    def item_embedding(self, item_f: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_tower(item_f), p=2, dim=1)


class TwoTowerModelLoader:
    """双塔模型加载器：支持 User Embedding、Item Embedding、以及 /predict 协议（返回 embedding 列表）。"""

    def __init__(self, model_path: str, meta_path: str):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.user_cols = None
        self.item_cols = None
        self.embed_dim = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        if not os.path.exists(self.model_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"双塔模型或元数据不存在: {self.model_path} / {self.meta_path}")
        with open(self.meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.user_cols = meta.get("user_cols", USER_COLS_DEFAULT)
        self.item_cols = meta.get("item_cols", ITEM_COLS_DEFAULT)
        self.embed_dim = meta.get("embed_dim", EMBED_DIM_DEFAULT)
        ckpt = torch.load(self.model_path, map_location=self.device)
        cfg = ckpt["config"]
        self.model = TwoTower(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        logger.info("双塔模型加载成功: %s", self.model_path)

    def _features_to_tensor(self, features: dict[str, float], cols: list[str]) -> torch.Tensor:
        vec = [float(features.get(c, 0.0)) for c in cols]
        return torch.FloatTensor([vec]).to(self.device)

    def get_user_embedding(self, user_features: dict[str, float]) -> list[float]:
        """单条用户特征 → User Embedding（供 Golang TwoTowerRecall 调用）。"""
        if self.model is None:
            raise RuntimeError("双塔模型未加载")
        x = self._features_to_tensor(user_features, self.user_cols)
        with torch.no_grad():
            emb = self.model.user_embedding(x)
        return emb[0].cpu().numpy().tolist()

    def get_item_embedding(self, item_features: dict[str, float]) -> list[float]:
        """单条物品特征 → Item Embedding（用于离线写入向量库）。"""
        if self.model is None:
            raise RuntimeError("双塔模型未加载")
        x = self._features_to_tensor(item_features, self.item_cols)
        with torch.no_grad():
            emb = self.model.item_embedding(x)
        return emb[0].cpu().numpy().tolist()

    def predict_user_embeddings(self, features_list: list[dict[str, float]]) -> list[list[float]]:
        """
        批量用户特征 → User Embedding 列表。
        与 MLService.Predict 协议对齐：Golang 传 Features，期望 Predictions 为 embedding 向量；
        单条时返回 [embedding]，Golang 取 resp.Predictions 即该 embedding。
        """
        if self.model is None or not features_list:
            return []
        rows = []
        for f in features_list:
            rows.append([float(f.get(c, 0.0)) for c in self.user_cols])
        x = torch.FloatTensor(rows).to(self.device)
        with torch.no_grad():
            emb = self.model.user_embedding(x)
        return emb.cpu().numpy().tolist()
