#!/usr/bin/env python3
"""
双塔模型（User Tower + Item Tower）训练脚本

User Tower：用户特征 → User Embedding
Item Tower：物品特征 → Item Embedding
训练目标：正样本 (user, item) 相似度高，负样本低（in-batch 负采样 + 交叉熵）

用法:
    python train/train_two_tower.py [--data data/two_tower_data.csv] [--epochs 20]
    python train/train_two_tower.py --data-source file --data-path data/train.csv

数据格式（CSV）：需包含 user_*、item_* 列及 label（0/1）
示例列：user_age, user_gender, item_ctr, item_cvr, item_price, label
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 特征列：与 features.py 及 EnrichNode 对齐（user_* / item_*）
USER_COLS = ["user_age", "user_gender"]
ITEM_COLS = ["item_ctr", "item_cvr", "item_price"]
LABEL_COL = "label"

MODEL_DIR = os.path.join(project_root, "model")
DATA_DIR = os.path.join(project_root, "data")
TWO_TOWER_MODEL_PATH = os.path.join(MODEL_DIR, "two_tower_model.pt")
TWO_TOWER_META_PATH = os.path.join(MODEL_DIR, "two_tower_meta.json")

EMBED_DIM = 64
USER_TOWER_LAYERS = [128, 64]
ITEM_TOWER_LAYERS = [128, 64]


def _mlp(dims: list[int], dropout: float = 0.2) -> nn.Module:
    mods = []
    for i in range(len(dims) - 1):
        mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*mods)


class TwoTower(nn.Module):
    """User Tower + Item Tower，L2 归一化后内积，用于召回/排序。"""

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
        self.user_tower = _mlp(dims_u)
        self.item_tower = _mlp(dims_i)

    def forward(self, user_f: torch.Tensor, item_f: torch.Tensor) -> torch.Tensor:
        u = F.normalize(self.user_tower(user_f), p=2, dim=1)
        i = F.normalize(self.item_tower(item_f), p=2, dim=1)
        logits = torch.mm(u, i.t())
        return logits

    def user_embedding(self, user_f: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_tower(user_f), p=2, dim=1)

    def item_embedding(self, item_f: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_tower(item_f), p=2, dim=1)


class TwoTowerDataset(Dataset):
    def __init__(self, user_feats: np.ndarray, item_feats: np.ndarray, labels: np.ndarray):
        self.user_feats = torch.FloatTensor(user_feats)
        self.item_feats = torch.FloatTensor(item_feats)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_feats[idx], self.item_feats[idx], self.labels[idx]


def generate_two_tower_data(path: str, n: int = 5000) -> pd.DataFrame:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.random.seed(42)
    user_feats = np.random.randn(n, len(USER_COLS)).astype(np.float32) * 0.5
    item_feats = np.random.randn(n, len(ITEM_COLS)).astype(np.float32) * 0.5
    user_feats = np.clip(user_feats, 0, 1)
    item_feats = np.clip(item_feats, 0, 1)
    sim = np.sum(user_feats * item_feats, axis=1) / (
        1e-6 + np.linalg.norm(user_feats, axis=1) * np.linalg.norm(item_feats, axis=1)
    )
    label = (sim + np.random.randn(n) * 0.3 > 0).astype(np.float32)
    cols = USER_COLS + ITEM_COLS + [LABEL_COL]
    data = np.hstack([user_feats, item_feats, label.reshape(-1, 1)])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(path, index=False)
    print(f"生成双塔示例数据: {path}, 行数: {len(df)}")
    return df


def train_model(
    data_path: str | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> None:
    path = data_path or os.path.join(DATA_DIR, "two_tower_data.csv")
    if not os.path.exists(path):
        generate_two_tower_data(path)

    df = pd.read_csv(path)
    for c in USER_COLS + ITEM_COLS + [LABEL_COL]:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")

    user_feats = df[USER_COLS].values.astype(np.float32)
    item_feats = df[ITEM_COLS].values.astype(np.float32)
    labels = (df[LABEL_COL].values.astype(np.float32) > 0.5).astype(np.int64)

    ds = TwoTowerDataset(user_feats, item_feats, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTower(
        user_dim=len(USER_COLS),
        item_dim=len(ITEM_COLS),
        embed_dim=EMBED_DIM,
        user_layers=USER_TOWER_LAYERS,
        item_layers=ITEM_TOWER_LAYERS,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total = 0.0
        for u_b, i_b, _ in loader:
            u_b, i_b = u_b.to(device), i_b.to(device)
            opt.zero_grad()
            logits = model(u_b, i_b)
            target = torch.arange(logits.size(0), device=device)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            opt.step()
            total += loss.item()
        total /= len(loader)
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1}/{epochs} loss={total:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "user_dim": len(USER_COLS),
                "item_dim": len(ITEM_COLS),
                "embed_dim": EMBED_DIM,
                "user_layers": USER_TOWER_LAYERS,
                "item_layers": ITEM_TOWER_LAYERS,
            },
        },
        TWO_TOWER_MODEL_PATH,
    )
    meta = {
        "user_cols": USER_COLS,
        "item_cols": ITEM_COLS,
        "embed_dim": EMBED_DIM,
    }
    with open(TWO_TOWER_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"双塔模型已保存: {TWO_TOWER_MODEL_PATH}")
    print(f"元数据: {TWO_TOWER_META_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="双塔（User Tower + Item Tower）训练")
    ap.add_argument("--data", default=None, help="训练数据 CSV 路径")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_model(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
