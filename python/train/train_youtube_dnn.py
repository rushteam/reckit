#!/usr/bin/env python3
"""
YouTube DNN 训练脚本

将推荐建模为多分类：用户特征 + 用户历史行为 -> 用户向量 Vu；
Softmax  over 物品，物品嵌入矩阵即 Item Embeddings。
服务时：用 Vu 做 ANN 检索 Item Embeddings。

用法:
    python train/train_youtube_dnn.py [--data data/youtube_dnn_data.csv] [--epochs 20]

数据格式 CSV: user_id, user_age, user_gender, history, target_item_id
- history: 逗号分隔的物品 ID，如 "item_1,item_2,item_3"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

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

YOUTUBE_DNN_MODEL_PATH = os.path.join(MODEL_DIR, "youtube_dnn_model.pt")
YOUTUBE_DNN_META_PATH = os.path.join(MODEL_DIR, "youtube_dnn_meta.json")

USER_FEATURE_COLS = ["user_age", "user_gender"]
MAX_HISTORY_LEN = 50
EMBED_DIM = 64
HIDDEN = [128, 64]


def generate_youtube_dnn_data(path: str, num_users: int = 500, num_items: int = 200, num_rows: int = 5000) -> pd.DataFrame:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.random.seed(42)
    item_ids = [f"item_{i+1}" for i in range(num_items)]
    rows = []
    for _ in range(num_rows):
        uid = f"u{np.random.randint(1, num_users + 1)}"
        age = float(np.random.randint(18, 56))
        gender = float(np.random.randint(0, 3))
        seq_len = np.random.randint(1, min(MAX_HISTORY_LEN, num_items))
        hist = list(np.random.choice(item_ids, size=seq_len, replace=True))
        target = np.random.choice(item_ids)
        rows.append({
            "user_id": uid,
            "user_age": age,
            "user_gender": gender,
            "history": ",".join(hist),
            "target_item_id": target,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"生成 YouTube DNN 示例数据: {path}, 行数: {len(df)}")
    return df


class YouTubeDNNDataset(Dataset):
    def __init__(self, user_feats, history_idx, target_idx, seq_len):
        self.user_feats = torch.FloatTensor(user_feats)
        self.history_idx = torch.LongTensor(history_idx)
        self.target_idx = torch.LongTensor(target_idx).squeeze()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.target_idx)

    def __getitem__(self, i):
        return self.user_feats[i], self.history_idx[i], self.target_idx[i]


class YouTubeDNN(nn.Module):
    """User 特征 + History 平均嵌入 -> DNN -> 用户向量；输出层为物品数 Softmax。"""

    def __init__(self, num_items: int, embed_dim: int, num_user_feats: int, hidden: list[int], max_history: int):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_history = max_history
        self.item_embed = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.user_proj = nn.Linear(num_user_feats, embed_dim)
        dims = [embed_dim * 2] + hidden + [embed_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(0.2)]
        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(embed_dim, num_items)

    def forward(self, user_feat, hist_idx, return_embedding: bool = False):
        # user_feat: [B, U], hist_idx: [B, L]
        B = user_feat.size(0)
        u = self.user_proj(user_feat)
        hist_emb = self.item_embed(hist_idx)
        mask = (hist_idx > 0).float().unsqueeze(-1).clamp(min=1e-9)
        hist_sum = (hist_emb * mask).sum(1)
        hist_cnt = mask.sum(1).clamp(min=1)
        h_avg = hist_sum / hist_cnt
        x = torch.cat([u, h_avg], dim=1)
        user_emb = self.mlp(x)
        if return_embedding:
            return user_emb
        logits = self.fc_out(user_emb)
        return logits

    def get_user_embedding(self, user_feat, hist_idx):
        return self.forward(user_feat, hist_idx, return_embedding=True)


def build_item_vocab(df: pd.DataFrame) -> dict[str, int]:
    vocab = {"__pad__": 0}
    for _, r in df.iterrows():
        for x in r["history"].split(","):
            x = x.strip()
            if x and x not in vocab:
                vocab[x] = len(vocab)
        tid = str(r["target_item_id"]).strip()
        if tid and tid not in vocab:
            vocab[tid] = len(vocab)
    return vocab


def encode_history(s: str, vocab: dict, max_len: int) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    idx = [vocab.get(x, 0) for x in parts[-max_len:]]
    pad = max_len - len(idx)
    return [0] * pad + idx if pad > 0 else idx


def train_model(data_path: str | None = None, epochs: int = 20, batch_size: int = 64) -> None:
    path = data_path or os.path.join(DATA_DIR, "youtube_dnn_data.csv")
    if not os.path.exists(path):
        generate_youtube_dnn_data(path)

    df = pd.read_csv(path)
    df["history"] = df["history"].astype(str)
    df["target_item_id"] = df["target_item_id"].astype(str)

    vocab = build_item_vocab(df)
    num_items = len(vocab) - 1
    assert num_items > 0

    user_feats = df[USER_FEATURE_COLS].values.astype(np.float32)
    history_idx = np.array([encode_history(s, vocab, MAX_HISTORY_LEN) for s in df["history"]])
    target_raw = np.array([vocab.get(t, 0) for t in df["target_item_id"]])
    valid = target_raw >= 1
    user_feats = user_feats[valid]
    history_idx = history_idx[valid]
    target_idx = (target_raw[valid] - 1).astype(np.int64)

    ds = YouTubeDNNDataset(user_feats, history_idx, target_idx, MAX_HISTORY_LEN)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YouTubeDNN(
        num_items=num_items,
        embed_dim=EMBED_DIM,
        num_user_feats=len(USER_FEATURE_COLS),
        hidden=HIDDEN,
        max_history=MAX_HISTORY_LEN,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for uf, hist, tgt in loader:
            uf, hist, tgt = uf.to(device), hist.to(device), tgt.to(device)
            opt.zero_grad()
            logits = model(uf, hist)
            loss = ce(logits, tgt)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        total_loss /= len(loader)
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1}/{epochs} loss={total_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    item_emb = model.item_embed.weight.detach().cpu().numpy()
    id_to_idx = {k: v for k, v in vocab.items() if k != "__pad__"}
    item_vectors = {}
    for item_id, idx in id_to_idx.items():
        if idx > 0:
            item_vectors[item_id] = item_emb[idx].tolist()

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "num_items": num_items,
            "embed_dim": EMBED_DIM,
            "num_user_feats": len(USER_FEATURE_COLS),
            "hidden": HIDDEN,
            "max_history": MAX_HISTORY_LEN,
        },
        "user_feature_cols": USER_FEATURE_COLS,
        "item_vocab": vocab,
    }, YOUTUBE_DNN_MODEL_PATH)

    meta = {
        "user_feature_cols": USER_FEATURE_COLS,
        "embed_dim": EMBED_DIM,
        "max_history": MAX_HISTORY_LEN,
        "item_vocab": vocab,
        "item_vectors": item_vectors,
    }
    with open(YOUTUBE_DNN_META_PATH, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"YouTube DNN 模型已保存: {YOUTUBE_DNN_MODEL_PATH}")
    print(f"元数据（含 item 向量）: {YOUTUBE_DNN_META_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YouTube DNN 训练")
    ap.add_argument("--data", default=None, help="CSV 路径")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    train_model(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size)
