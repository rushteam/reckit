#!/usr/bin/env python3
"""
MMoE (Multi-gate Mixture-of-Experts) 多目标训练脚本

多任务：CTR（点击）、watch_time（时长）、gmv（GMV）。
用于重排阶段多目标优化，解决「看而不点」「点而不转」等目标冲突。

用法:
    python train/train_mmoe.py [--version VERSION] [--epochs EPOCHS] [--batch-size BATCH_SIZE]

数据: 与 DeepFM 相同特征列，另需 label, watch_time, gmv 三列。
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import importlib.util
features_path = os.path.join(os.path.dirname(__file__), "features.py")
spec = importlib.util.spec_from_file_location("features", features_path)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)
FEATURE_COLUMNS = features.FEATURE_COLUMNS
FEATURE_META_PATH = features.FEATURE_META_PATH
LABEL_COLUMN = features.LABEL_COLUMN
MODEL_DIR = features.MODEL_DIR

_data_loader_path = os.path.join(os.path.dirname(__file__), "data_loader.py")
_data_loader_spec = importlib.util.spec_from_file_location("data_loader", _data_loader_path)
_data_loader_mod = importlib.util.module_from_spec(_data_loader_spec)
_data_loader_spec.loader.exec_module(_data_loader_mod)
load_training_df = _data_loader_mod.load_training_df

MMOE_MODEL_PATH = os.path.join(MODEL_DIR, "mmoe_model.pt")
MMOE_FEATURE_META_PATH = os.path.join(MODEL_DIR, "mmoe_feature_meta.json")

# 多任务标签列
TASK_COLUMNS = ["label", "watch_time", "gmv"]


class MMoEDataset(Dataset):
    def __init__(self, X, y_ctr, y_watch, y_gmv):
        self.X = torch.FloatTensor(X)
        self.y_ctr = torch.FloatTensor(y_ctr).unsqueeze(1)
        self.y_watch = torch.FloatTensor(y_watch).unsqueeze(1)
        self.y_gmv = torch.FloatTensor(y_gmv).unsqueeze(1)

    def __len__(self):
        return len(self.y_ctr)

    def __getitem__(self, idx):
        return self.X[idx], self.y_ctr[idx], self.y_watch[idx], self.y_gmv[idx]


class MMoE(nn.Module):
    """
    MMoE: 共享底层 + 多专家 + 每任务门控 + 任务塔。
    输出三路：ctr (sigmoid), watch_time (非负), gmv (非负)。
    """

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

        # 共享底层
        layers = [num_features] + expert_units
        self.shared = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.shared.append(nn.Linear(layers[i], layers[i + 1]))
            self.shared.append(nn.ReLU())
            self.shared.append(nn.Dropout(dropout))
        shared_out = layers[-1]

        # 专家网络
        self.experts = nn.ModuleList([
            self._make_expert(shared_out, expert_units[-1]) for _ in range(num_experts)
        ])
        expert_out = expert_units[-1]

        # 每任务门控 + 任务塔
        self.gate_ctr = nn.Sequential(nn.Linear(shared_out, num_experts), nn.Softmax(dim=-1))
        self.gate_watch = nn.Sequential(nn.Linear(shared_out, num_experts), nn.Softmax(dim=-1))
        self.gate_gmv = nn.Sequential(nn.Linear(shared_out, num_experts), nn.Softmax(dim=-1))

        task_in = expert_out * num_experts
        task_layers = [task_in] + task_units + [1]
        self.tower_ctr = self._make_tower(task_layers, sigmoid=True)
        self.tower_watch = self._make_tower(task_layers, sigmoid=False)
        self.tower_gmv = self._make_tower(task_layers, sigmoid=False)

    def _make_expert(self, inp: int, out: int):
        return nn.Sequential(
            nn.Linear(inp, out),
            nn.ReLU(),
        )

    def _make_tower(self, layers: list[int], sigmoid: bool):
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.ReLU())
                mods.append(nn.Dropout(0.2))
        if sigmoid:
            mods.append(nn.Sigmoid())
        return nn.Sequential(*mods)

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


def generate_mmoe_sample_data(output_path: str, n_samples: int = 1000):
    np.random.seed(42)
    item_ctr = np.random.uniform(0.01, 0.5, n_samples)
    item_cvr = np.random.uniform(0.001, 0.1, n_samples)
    item_price = np.random.uniform(10, 200, n_samples)
    user_age = np.random.randint(18, 60, n_samples).astype(np.float64)
    user_gender = np.random.randint(0, 3, n_samples).astype(np.float64)
    cross_age_ctr = user_age * item_ctr
    cross_gender_price = user_gender * item_price

    label_raw = (
        0.5 * item_ctr * 10 + 0.3 * item_cvr * 20
        + 0.1 * user_age / 100 + 0.05 * user_gender / 2.0
        + np.random.normal(0, 0.1, n_samples)
    )
    label = (label_raw > 0.5).astype(np.float64)
    watch_time = np.clip(30.0 * label + np.random.exponential(5.0, n_samples), 0, 600.0)
    gmv = np.clip(
        (label * item_price * (0.5 + 0.5 * np.random.rand(n_samples))) + np.random.uniform(0, 10, n_samples),
        0, 1e6,
    )

    df = pd.DataFrame({
        "item_ctr": item_ctr,
        "item_cvr": item_cvr,
        "item_price": item_price,
        "user_age": user_age,
        "user_gender": user_gender,
        "cross_age_x_ctr": cross_age_ctr,
        "cross_gender_x_price": cross_gender_price,
        "label": label,
        "watch_time": watch_time,
        "gmv": gmv,
    })
    df.to_csv(output_path, index=False)
    print(f"生成 MMoE 示例数据: {output_path}, 样本数: {n_samples}")
    return df


def train_model(
    data_source: str = "file",
    data_path: str | None = None,
    model_version: str | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    **kwargs,
):
    if data_source.strip().lower() == "file":
        if not data_path:
            data_path = os.path.join(project_root, "data", "mmoe_train_data.csv")
        if not os.path.exists(data_path):
            os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
            df = generate_mmoe_sample_data(data_path)
        else:
            df = load_training_df("file", path=data_path)
    else:
        df = load_training_df(
            data_source,
            path=data_path,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

    required = FEATURE_COLUMNS + TASK_COLUMNS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {missing}")

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y_ctr = df["label"].values.astype(np.float32)
    y_watch = df["watch_time"].values.astype(np.float32)
    y_gmv = df["gmv"].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    stratify = (y_ctr > 0.5).astype(int) if len(np.unique(y_ctr)) > 1 else None
    X_train, X_val, yc_t, yc_v, yw_t, yw_v, yg_t, yg_v = train_test_split(
        X, y_ctr, y_watch, y_gmv, test_size=0.2, random_state=42, stratify=stratify
    )

    train_ds = MMoEDataset(X_train, yc_t, yw_t, yg_t)
    val_ds = MMoEDataset(X_val, yc_v, yw_v, yg_v)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMoE(
        num_features=len(FEATURE_COLUMNS),
        expert_units=[64, 32],
        num_experts=4,
        task_units=[32, 16],
        dropout=0.3,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, bc, bw, bg in train_loader:
            bx, bc, bw, bg = bx.to(device), bc.to(device), bw.to(device), bg.to(device)
            opt.zero_grad()
            ctr, watch, gmv = model(bx)
            l_ctr = bce(ctr, bc)
            l_watch = mse(watch, bw)
            l_gmv = mse(gmv, bg)
            loss = l_ctr + 0.01 * l_watch + 1e-6 * l_gmv
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, bc, bw, bg in val_loader:
                bx, bc, bw, bg = bx.to(device), bc.to(device), bw.to(device), bg.to(device)
                ctr, watch, gmv = model(bx)
                l_ctr = bce(ctr, bc)
                l_watch = mse(watch, bw)
                l_gmv = mse(gmv, bg)
                val_loss += (l_ctr + 0.01 * l_watch + 1e-6 * l_gmv).item()
        val_loss /= len(val_loader)
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    version = model_version or datetime.utcnow().strftime("%Y%m%d%H%M")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_features": len(FEATURE_COLUMNS),
            "expert_units": [64, 32],
            "num_experts": 4,
            "task_units": [32, 16],
            "dropout": 0.3,
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "model_version": version,
    }, MMOE_MODEL_PATH)

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "model_version": version,
        "tasks": TASK_COLUMNS,
    }
    with open(MMOE_FEATURE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"MMoE 模型已保存: {MMOE_MODEL_PATH}")
    print(f"特征元数据: {MMOE_FEATURE_META_PATH}")
    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMoE 多目标训练")
    parser.add_argument("--data-source", default="file", choices=("file", "oss", "mysql", "doris"))
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train_model(
        data_source=args.data_source,
        data_path=args.data_path,
        model_version=args.version,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
