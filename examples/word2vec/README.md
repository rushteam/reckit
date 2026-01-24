# Word2Vec / Item2Vec 示例

本示例演示 **Word2Vec**（文本）与 **Item2Vec**（序列）两种召回模式。

## 运行

```bash
# 从项目根目录
go run ./examples/word2vec
```

模型优先从以下 JSON 加载（与 `python/train/train_item2vec.py` 输出格式一致）：

1. `examples/word2vec/item2vec_vectors.json`
2. `python/model/item2vec_vectors.json`

若不存在则使用内联示例向量。

## Python 训练

```bash
cd python
pip install -r requirements.txt

# Item2Vec：用户行为序列 -> 物品向量
python train/train_item2vec.py --mode item2vec --data data/behavior.csv \
  --out model/item2vec_vectors.json

# Word2Vec：文本语料 -> 词向量
python train/train_item2vec.py --mode word2vec --data data/corpus.txt \
  --out model/word2vec_vectors.json
```

无 `--data` 时脚本会生成示例数据并训练。详见 `docs/WORD2VEC_ITEM2VEC.md`。

## 模式说明

| 模式 | 使用场景 | 用户向量 | 物品向量 |
|------|----------|----------|----------|
| `text` | 标题/标签相似 | 最近点击物品的文本编码 | 物品文本编码 |
| `sequence` (Item2Vec) | 行为序列相似 | 用户行为序列（物品 ID 列表）编码 | 物品 ID 向量 |

## 参考

- [Word2Vec 与 Item2Vec 使用指南](../../docs/WORD2VEC_ITEM2VEC.md)
- [召回算法说明](../../docs/RECALL_ALGORITHMS.md)
