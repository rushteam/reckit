"""
YouTube DNN 模型加载器：用户特征 + 历史 -> 用户向量
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

USER_FEATURE_COLS = ["user_age", "user_gender"]
MAX_HISTORY_LEN = 50
EMBED_DIM = 64
HIDDEN = [128, 64]


class YouTubeDNN(nn.Module):
    """与 train_youtube_dnn 一致；推理时仅用 mlp 输出作为 user embedding。"""

    def __init__(self, num_items: int, embed_dim: int, num_user_feats: int, hidden: list, max_history: int):
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

    def forward(self, user_feat, hist_idx):
        u = self.user_proj(user_feat)
        hist_emb = self.item_embed(hist_idx)
        mask = (hist_idx > 0).float().unsqueeze(-1).clamp(min=1e-9)
        hist_sum = (hist_emb * mask).sum(1)
        hist_cnt = mask.sum(1).clamp(min=1)
        h_avg = hist_sum / hist_cnt
        x = torch.cat([u, h_avg], dim=1)
        return self.mlp(x)

    def get_user_embedding(self, user_feat, hist_idx):
        return self.forward(user_feat, hist_idx)


class YouTubeDNNModelLoader:
    def __init__(self, model_path: str, meta_path: str):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.vocab = None
        self.user_feature_cols = None
        self.max_history = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"模型或元数据不存在: {self.model_path} / {self.meta_path}")
        with open(self.meta_path) as f:
            meta = json.load(f)
        self.vocab = meta.get("item_vocab", {})
        self.user_feature_cols = meta.get("user_feature_cols", USER_FEATURE_COLS)
        self.max_history = meta.get("max_history", MAX_HISTORY_LEN)
        ckpt = torch.load(self.model_path, map_location=self.device)
        cfg = ckpt["config"]
        self.model = YouTubeDNN(
            num_items=cfg["num_items"],
            embed_dim=cfg["embed_dim"],
            num_user_feats=cfg["num_user_feats"],
            hidden=cfg["hidden"],
            max_history=cfg["max_history"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        logger.info("YouTube DNN 模型加载成功: %s", self.model_path)

    def _encode_history(self, ids: list[str]) -> list[int]:
        idx = [self.vocab.get(x, 0) for x in ids[-self.max_history:]]
        pad = self.max_history - len(idx)
        return ([0] * pad + idx) if pad > 0 else idx

    def get_user_embedding(self, user_features: dict, history_item_ids: list[str]) -> list[float]:
        if self.model is None:
            raise RuntimeError("YouTube DNN 未加载")
        uf = np.array([[float(user_features.get(c, 0)) for c in self.user_feature_cols]], dtype=np.float32)
        hist = [self._encode_history(history_item_ids)]
        uf_t = torch.FloatTensor(uf).to(self.device)
        hist_t = torch.LongTensor(hist).to(self.device)
        with torch.no_grad():
            emb = self.model.get_user_embedding(uf_t, hist_t)
        return emb[0].cpu().numpy().tolist()
