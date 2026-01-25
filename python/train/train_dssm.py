#!/usr/bin/env python3
"""
DSSM (Deep Structured Semantic Model) 训练脚本

Query 塔 + Doc 塔，语义匹配。用于搜索推荐、Query-Doc 召回。

用法:
    python train/train_dssm.py [--data data/dssm_data.csv] [--epochs 20]

数据 CSV: query_f1, query_f2, ..., doc_f1, doc_f2, ..., label (0/1)
示例生成：query 3 维，doc 3 维，label 由相似度 + 噪声生成。
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
MODEL_DIR = os.path.join(project_root, "model")
DATA_DIR = os.path.join(project_root, "data")

DSSM_MODEL_PATH = os.path.join(MODEL_DIR, "dssm_model.pt")
DSSM_META_PATH = os.path.join(MODEL_DIR, "dssm_meta.json")

QUERY_COLS = ["query_f1", "query_f2", "query_f3"]
DOC_COLS = ["doc_f1", "doc_f2", "doc_f3"]
LABEL_COL = "label"
EMBED_DIM = 64
TOWER_LAYERS = [128, 64]


def generate_dssm_data(path: str, n: int = 5000) -> pd.DataFrame:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.random.seed(42)
    q = np.random.randn(n, 3).astype(np.float32) * 0.5
    d = np.random.randn(n, 3).astype(np.float32) * 0.5
    sim = np.sum(q * d, axis=1) / (1e-6 + np.linalg.norm(q, axis=1) * np.linalg.norm(d, axis=1))
    label = (sim + np.random.randn(n) * 0.3 > 0).astype(np.float32)
    df = pd.DataFrame(
        np.hstack([q, d, label.reshape(-1, 1)]),
        columns=QUERY_COLS + DOC_COLS + [LABEL_COL],
    )
    df.to_csv(path, index=False)
    print(f"生成 DSSM 示例数据: {path}, 行数: {len(df)}")
    return df


class DSSMDataset(Dataset):
    def __init__(self, query: np.ndarray, doc: np.ndarray, label: np.ndarray):
        self.q = torch.FloatTensor(query)
        self.d = torch.FloatTensor(doc)
        self.y = torch.FloatTensor(label).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.q[i], self.d[i], self.y[i]


class DSSM(nn.Module):
    """Query 塔 + Doc 塔，余弦相似度，输出 sigmoid 匹配分。"""

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

    def forward(self, q, d):
        qe = self.query_tower(q)
        de = self.doc_tower(d)
        qe = nn.functional.normalize(qe, p=2, dim=1)
        de = nn.functional.normalize(de, p=2, dim=1)
        logits = (qe * de).sum(dim=1, keepdim=True)
        return torch.sigmoid(logits)

    def query_embedding(self, q):
        qe = self.query_tower(q)
        return nn.functional.normalize(qe, p=2, dim=1)


def train_model(data_path: str | None = None, epochs: int = 20, batch_size: int = 64) -> None:
    path = data_path or os.path.join(DATA_DIR, "dssm_data.csv")
    if not os.path.exists(path):
        generate_dssm_data(path)

    df = pd.read_csv(path)
    q = df[QUERY_COLS].values.astype(np.float32)
    d = df[DOC_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    ds = DSSMDataset(q, d, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSSM(dim_in=len(QUERY_COLS), embed_dim=EMBED_DIM, layers=TOWER_LAYERS).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    for ep in range(epochs):
        model.train()
        total = 0.0
        for qb, db, yb in loader:
            qb, db, yb = qb.to(device), db.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(qb, db)
            loss = bce(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        total /= len(loader)
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1}/{epochs} loss={total:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"dim_in": len(QUERY_COLS), "embed_dim": EMBED_DIM, "layers": TOWER_LAYERS},
        "query_cols": QUERY_COLS,
        "doc_cols": DOC_COLS,
    }, DSSM_MODEL_PATH)

    meta = {"query_cols": QUERY_COLS, "doc_cols": DOC_COLS, "embed_dim": EMBED_DIM}
    with open(DSSM_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"DSSM 模型已保存: {DSSM_MODEL_PATH}")
    print(f"元数据: {DSSM_META_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DSSM 训练")
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    train_model(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size)
