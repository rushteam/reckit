#!/usr/bin/env python3
"""
Node2Vec 训练脚本（社交/关注图 -> 节点嵌入）

用于关注页召回、相似用户推荐等。输入边表 (user_id, follow_user_id)，输出节点嵌入。

用法:
    python train/train_node2vec.py [--edges data/graph_edges.csv] [--dim 64] [--epochs 10]

数据 CSV: 两列 from_id, to_id（如 user_id, follow_user_id）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
MODEL_DIR = os.path.join(project_root, "model")
DATA_DIR = os.path.join(project_root, "data")

NODE2VEC_META_PATH = os.path.join(MODEL_DIR, "node2vec_meta.json")
DEFAULT_DIM = 64
DEFAULT_WALKS = 50
DEFAULT_WALK_LEN = 20
DEFAULT_EPOCHS = 10


def generate_graph_edges(path: str, num_nodes: int = 300, num_edges: int = 2000) -> pd.DataFrame:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.random.seed(42)
    nodes = [f"u{i+1}" for i in range(num_nodes)]
    rows = []
    for _ in range(num_edges):
        u, v = np.random.choice(nodes, 2, replace=False)
        rows.append({"from_id": u, "to_id": v})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"生成图边表示例: {path}, 边数: {len(df)}")
    return df


def load_edges(path: str, from_col: str = "from_id", to_col: str = "to_id") -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    df[from_col] = df[from_col].astype(str)
    df[to_col] = df[to_col].astype(str)
    return list(zip(df[from_col], df[to_col]))


def run_node2vec_gensim(edges: list[tuple[str, str]], dim: int, num_walks: int, walk_length: int, epochs: int):
    """简单随机游走 + Word2Vec，不依赖 node2vec 包。"""
    import networkx as nx
    from gensim.models import Word2Vec

    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)

    nodes = list(G.nodes())
    np.random.seed(42)

    def random_walks():
        for _ in range(num_walks):
            for start in nodes:
                walk = [start]
                for _ in range(walk_length - 1):
                    nbrs = list(G.neighbors(walk[-1]))
                    if not nbrs:
                        break
                    walk.append(np.random.choice(nbrs))
                yield [str(x) for x in walk]

    sentences = list(random_walks())
    model = Word2Vec(sentences, vector_size=dim, window=5, min_count=1, sg=1, epochs=epochs, seed=42)
    embeddings = {}
    for n in nodes:
        embeddings[n] = model.wv[n].tolist()
    return embeddings


def train_model(
    edges_path: str | None = None,
    dim: int = DEFAULT_DIM,
    num_walks: int = DEFAULT_WALKS,
    walk_length: int = DEFAULT_WALK_LEN,
    epochs: int = DEFAULT_EPOCHS,
) -> None:
    path = edges_path or os.path.join(DATA_DIR, "graph_edges.csv")
    if not os.path.exists(path):
        generate_graph_edges(path)

    edges = load_edges(path)
    if not edges:
        raise ValueError("边表为空")

    try:
        emb = run_node2vec_gensim(edges, dim, num_walks, walk_length, epochs)
    except ImportError as e:
        raise RuntimeError("需要 networkx 和 gensim: pip install networkx gensim") from e

    os.makedirs(MODEL_DIR, exist_ok=True)
    meta = {
        "embeddings": emb,
        "dim": dim,
        "num_nodes": len(emb),
    }
    with open(NODE2VEC_META_PATH, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Node2Vec 嵌入已保存: {NODE2VEC_META_PATH} (dim={dim}, nodes={len(emb)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Node2Vec 训练")
    ap.add_argument("--edges", default=None, help="边表 CSV: from_id, to_id")
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--num-walks", type=int, default=DEFAULT_WALKS)
    ap.add_argument("--walk-length", type=int, default=DEFAULT_WALK_LEN)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = ap.parse_args()
    train_model(
        edges_path=args.edges,
        dim=args.dim,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        epochs=args.epochs,
    )
