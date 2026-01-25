"""
MMoE 模型加载器：多任务预测 (ctr, watch_time, gmv)
"""
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, project_root)


class MMoE(nn.Module):
    """与 train_mmoe 一致的 MMoE 结构"""

    def __init__(
        self,
        num_features: int,
        expert_units: list[int] = [64, 32],
        num_experts: int = 4,
        task_units: list[int] = [32, 16],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_experts = num_experts
        layers = [num_features] + expert_units
        self.shared = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.shared.append(nn.Linear(layers[i], layers[i + 1]))
            self.shared.append(nn.ReLU())
            self.shared.append(nn.Dropout(dropout))
        shared_out = layers[-1]

        def make_expert(inp, out):
            return nn.Sequential(nn.Linear(inp, out), nn.ReLU())

        self.experts = nn.ModuleList([
            make_expert(shared_out, expert_units[-1]) for _ in range(num_experts)
        ])
        expert_out = expert_units[-1]

        self.gate_ctr = nn.Sequential(nn.Linear(shared_out, num_experts), nn.Softmax(dim=-1))
        self.gate_watch = nn.Sequential(nn.Linear(shared_out, num_experts), nn.Softmax(dim=-1))
        self.gate_gmv = nn.Sequential(nn.Linear(shared_out, num_experts), nn.Softmax(dim=-1))

        task_in = expert_out * num_experts
        task_layers = [task_in] + task_units + [1]

        def make_tower(ls, sigmoid):
            mods = []
            for i in range(len(ls) - 1):
                mods.append(nn.Linear(ls[i], ls[i + 1]))
                if i < len(ls) - 2:
                    mods.append(nn.ReLU())
                    mods.append(nn.Dropout(0.2))
            if sigmoid:
                mods.append(nn.Sigmoid())
            return nn.Sequential(*mods)

        self.tower_ctr = make_tower(task_layers, sigmoid=True)
        self.tower_watch = make_tower(task_layers, sigmoid=False)
        self.tower_gmv = make_tower(task_layers, sigmoid=False)

    def forward(self, x):
        h = x
        for layer in self.shared:
            h = layer(h)
        expert_outs = [e(h) for e in self.experts]
        expert_stack = torch.stack(expert_outs, dim=1)
        g_ctr = self.gate_ctr(h).unsqueeze(-1)
        g_watch = self.gate_watch(h).unsqueeze(-1)
        g_gmv = self.gate_gmv(h).unsqueeze(-1)
        ctr_in = (expert_stack * g_ctr).flatten(1)
        watch_in = (expert_stack * g_watch).flatten(1)
        gmv_in = (expert_stack * g_gmv).flatten(1)
        ctr = self.tower_ctr(ctr_in)
        watch = torch.relu(self.tower_watch(watch_in))
        gmv = torch.relu(self.tower_gmv(gmv_in))
        return ctr, watch, gmv


class MMoEModelLoader:
    def __init__(self, model_path: str, feature_meta_path: str, model_version: str | None = None):
        self.model_path = model_path
        self.feature_meta_path = feature_meta_path
        self.model_version = model_version
        self.model = None
        self.feature_columns = None
        self.feature_count = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.feature_meta_path):
            raise FileNotFoundError(f"模型或元数据不存在: {self.model_path} / {self.feature_meta_path}")
        with open(self.feature_meta_path) as f:
            meta = json.load(f)
        self.feature_columns = meta.get("feature_columns", [])
        self.feature_count = len(self.feature_columns)
        self.model_version = meta.get("model_version", self.model_version)
        ckpt = torch.load(self.model_path, map_location=self.device)
        cfg = ckpt["model_config"]
        self.model = MMoE(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.scaler_mean = np.array(ckpt.get("scaler_mean", []))
        self.scaler_scale = np.array(ckpt.get("scaler_scale", []))
        logger.info("MMoE 模型加载成功: %s", self.model_path)

    def _validate_normalize(self, features: dict) -> dict:
        out = {}
        for col in self.feature_columns:
            v = features.get(col)
            try:
                out[col] = float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                out[col] = 0.0
            if not np.isfinite(out[col]):
                out[col] = 0.0
        if len(self.scaler_mean) == len(self.feature_columns) and len(self.scaler_scale) == len(self.feature_columns):
            for i, col in enumerate(self.feature_columns):
                m, s = self.scaler_mean[i], self.scaler_scale[i]
                if s > 0:
                    out[col] = (out[col] - m) / s
        return out

    def predict_multi_task(self, features_list: list[dict]) -> list[dict]:
        if self.model is None:
            raise RuntimeError("MMoE 未加载")
        if not features_list:
            return []
        vectors = []
        for f in features_list:
            v = self._validate_normalize(f)
            vectors.append([v[c] for c in self.feature_columns])
        X = torch.FloatTensor(vectors).to(self.device)
        with torch.no_grad():
            ctr, watch, gmv = self.model(X)
        result = []
        for i in range(len(features_list)):
            result.append({
                "ctr": float(ctr[i].item()),
                "watch_time": float(watch[i].item()),
                "gmv": float(gmv[i].item()),
            })
        return result
