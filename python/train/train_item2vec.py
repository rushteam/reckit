#!/usr/bin/env python3
"""
Item2Vec / Word2Vec 训练脚本

用法:
    # Item2Vec：从用户行为序列训练物品向量
    python train/train_item2vec.py --mode item2vec --data data/behavior.csv \\
        --user-col user_id --sequence-col sequence \\
        --dim 128 --epochs 10 --out model/item2vec_vectors.json

    # Word2Vec：从文本语料训练词向量
    python train/train_item2vec.py --mode word2vec --data data/corpus.txt \\
        --dim 128 --epochs 10 --out model/word2vec_vectors.json

    # 无数据时生成示例并训练（Item2Vec）
    python train/train_item2vec.py --mode item2vec --data data/behavior.csv \\
        --out model/item2vec_vectors.json

输出:
    JSON 文件，格式 {"id": [f1, f2, ...], ...}，供 Golang LoadWord2VecFromMap 加载。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import numpy as np
    import gensim
    from gensim.models import Word2Vec
except ImportError as e:
    print("请安装依赖: pip install gensim numpy")
    raise SystemExit(1) from e

DEFAULT_DIM = 128
DEFAULT_EPOCHS = 10
DEFAULT_WINDOW = 5
DEFAULT_MIN_COUNT = 1
DEFAULT_WORKERS = 4


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------- 示例数据生成 ----------


def generate_behavior_csv(path: str, num_users: int = 500, num_items: int = 200) -> str:
    """生成示例用户行为序列 CSV。列: user_id, sequence（逗号分隔的物品 ID）。"""
    _ensure_dir(path)
    np.random.seed(42)
    rows = []
    for i in range(num_users):
        uid = f"u{i+1}"
        n = np.random.randint(5, 50)
        seq = np.random.choice([f"item_{j+1}" for j in range(num_items)], size=n, replace=True)
        rows.append((uid, ",".join(seq)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("user_id,sequence\n")
        for uid, seq in rows:
            f.write(f'"{uid}","{seq}"\n')
    print(f"已生成示例行为数据: {path} ({num_users} 用户)")
    return path


def generate_corpus_txt(path: str, num_lines: int = 2000) -> str:
    """生成示例文本语料。每行一句，空格分词。"""
    _ensure_dir(path)
    words = (
        "electronics smartphone tech mobile device laptop computer "
        "game sports music movie book fashion food travel"
    ).split()
    np.random.seed(42)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(num_lines):
            n = np.random.randint(3, 15)
            f.write(" ".join(np.random.choice(words, n)) + "\n")
    print(f"已生成示例语料: {path} ({num_lines} 行)")
    return path


# ---------- Item2Vec ----------


def load_behavior_sequences(
    data_path: str,
    user_col: str = "user_id",
    sequence_col: str = "sequence",
    sep: str = ",",
) -> list[list[str]]:
    """
    从 CSV 加载用户行为序列。
    CSV 需包含 user_col、sequence_col。sequence_col 为逗号分隔的物品 ID。
    """
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("Item2Vec 需 pandas: pip install pandas") from None

    df = pd.read_csv(data_path)
    if user_col not in df.columns or sequence_col not in df.columns:
        raise ValueError(f"CSV 需包含列: {user_col}, {sequence_col}")

    sentences = []
    for _, row in df.iterrows():
        raw = row[sequence_col]
        if isinstance(raw, str):
            seq = [x.strip() for x in raw.split(sep) if x.strip()]
        else:
            seq = []
        if seq:
            sentences.append(seq)
    return sentences


def train_item2vec(
    sentences: list[list[str]],
    *,
    dim: int = DEFAULT_DIM,
    epochs: int = DEFAULT_EPOCHS,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    workers: int = DEFAULT_WORKERS,
    seed: int = 42,
) -> Word2Vec:
    """训练 Item2Vec：物品 ID 为词，用户序列为句子。"""
    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed,
        sg=1,  # Skip-gram
    )
    return model


# ---------- Word2Vec（文本）----------


def load_text_corpus(data_path: str) -> list[list[str]]:
    """从纯文本文件加载语料。每行一句，按空格分词。"""
    sentences = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if words:
                sentences.append(words)
    return sentences


def train_word2vec(
    sentences: list[list[str]],
    *,
    dim: int = DEFAULT_DIM,
    epochs: int = DEFAULT_EPOCHS,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    workers: int = DEFAULT_WORKERS,
    seed: int = 42,
) -> Word2Vec:
    """训练 Word2Vec：文本词为词，每行一句。"""
    return train_item2vec(
        sentences,
        dim=dim,
        epochs=epochs,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )


# ---------- 导出 JSON ----------


def export_vectors_json(model: Word2Vec, out_path: str) -> None:
    """
    导出 {word: [f1, f2, ...]} 的 JSON，供 Golang LoadWord2VecFromMap 使用。
    """
    _ensure_dir(out_path)
    vocab = model.wv
    obj = {}
    for w in vocab.index_to_key:
        obj[w] = [float(x) for x in vocab[w].tolist()]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"已导出向量 JSON: {out_path} ({len(obj)} 个词/物品)")


# ---------- CLI ----------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Item2Vec / Word2Vec 训练，输出 JSON 供 Reckit Golang 加载",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", choices=("item2vec", "word2vec"), required=True, help="item2vec 或 word2vec")
    ap.add_argument("--data", default="", help="训练数据路径。Item2Vec: CSV；Word2Vec: 纯文本 .txt")
    ap.add_argument("--out", default="model/word2vec_vectors.json", help="输出 JSON 路径")
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM, help="向量维度")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="上下文窗口")
    ap.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT, help="最小词频")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并行 workers")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    # Item2Vec 专用
    ap.add_argument("--user-col", default="user_id", help="用户 ID 列名")
    ap.add_argument("--sequence-col", default="sequence", help="序列列名（逗号分隔物品 ID）")
    ap.add_argument("--no-gen", action="store_true", help="禁用自动生成示例数据")

    args = ap.parse_args()
    data_path = args.data.strip()
    out_path = args.out.strip()

    if args.mode == "item2vec":
        if not data_path:
            data_path = os.path.join(project_root, "data", "behavior.csv")
        if not os.path.exists(data_path) and not args.no_gen:
            generate_behavior_csv(data_path)
        if not os.path.exists(data_path):
            raise SystemExit(f"数据不存在: {data_path}，且未生成示例。可去掉 --no-gen 或提供 --data。")
        sentences = load_behavior_sequences(
            data_path,
            user_col=args.user_col,
            sequence_col=args.sequence_col,
        )
        print(f"Item2Vec: 加载 {len(sentences)} 条用户序列")
        model = train_item2vec(
            sentences,
            dim=args.dim,
            epochs=args.epochs,
            window=args.window,
            min_count=args.min_count,
            workers=args.workers,
            seed=args.seed,
        )
    else:
        if not data_path:
            data_path = os.path.join(project_root, "data", "corpus.txt")
        if not os.path.exists(data_path) and not args.no_gen:
            generate_corpus_txt(data_path)
        if not os.path.exists(data_path):
            raise SystemExit(f"数据不存在: {data_path}，且未生成示例。可去掉 --no-gen 或提供 --data。")
        sentences = load_text_corpus(data_path)
        print(f"Word2Vec: 加载 {len(sentences)} 句")
        model = train_word2vec(
            sentences,
            dim=args.dim,
            epochs=args.epochs,
            window=args.window,
            min_count=args.min_count,
            workers=args.workers,
            seed=args.seed,
        )

    export_vectors_json(model, out_path)
    print("完成。")


if __name__ == "__main__":
    main()
