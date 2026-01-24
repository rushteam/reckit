# Word2Vec 与 Item2Vec 使用指南

本文档说明 Reckit 中 **Word2Vec**（文本词向量）与 **Item2Vec**（物品序列向量）的用法，包括 **Python 训练方法**、模型导出与 Golang 召回接入。

---

## Python 训练方法（摘要）

1. **安装依赖**：`pip install gensim numpy pandas`（见 `python/requirements.txt`）。
2. **准备数据**：
   - **Item2Vec**：CSV，列 `user_id`、`sequence`；`sequence` 为逗号分隔的物品 ID，如 `"item_1,item_2,item_3"`。
   - **Word2Vec**：纯文本，每行一句，空格分词。
3. **训练命令**：
   ```bash
   cd python
   # Item2Vec
   python train/train_item2vec.py --mode item2vec --data data/behavior.csv --out model/item2vec_vectors.json
   # Word2Vec
   python train/train_item2vec.py --mode word2vec --data data/corpus.txt --out model/word2vec_vectors.json
   ```
4. **输出**：JSON 文件 `{"id": [f1, f2, ...], ...}`，供 Golang `LoadWord2VecFromMap` 加载。

不加 `--no-gen` 且数据不存在时会自动生成示例数据并训练；加 `--no-gen` 则不生成，数据缺失时报错。详见下文。

---

## 一、概念与区别

| 维度 | Word2Vec | Item2Vec |
|------|----------|----------|
| **“词”** | 文本中的词（如 "手机"、"电脑"） | 物品 ID（如 "item_1"、"item_2"） |
| **“句子”** | 一段文本（按空格分词） | 用户行为序列（点击/浏览/购买顺序） |
| **训练数据** | 语料文本 | 用户行为日志（user_id → 物品ID序列） |
| **输出** | 词 → 向量 | 物品ID → 向量 |
| **典型用途** | 标题/描述相似、标签匹配 | I2I 相似、序列召回、下一跳预测 |

Reckit 的 `Word2VecModel` 与 `Word2VecRecall` **同时支持** 两种用法：

- **文本模式**：`Mode: "text"`，用 `EncodeText` / `EncodeWords`，词表为文本词。
- **Item2Vec 模式**：`Mode: "sequence"`，用 `EncodeSequence`，词表为物品 ID；序列来自 `GetUserSequence` 或 `User.RecentClicks`。

---

## 二、Python 训练方法

### 2.1 环境与依赖

```bash
cd python
pip install -r requirements.txt
# Item2Vec / Word2Vec 训练额外依赖
pip install gensim
```

### 2.2 训练脚本

在 `python/` 目录下运行 `train/train_item2vec.py`：

```bash
cd python

# Item2Vec：从用户行为序列训练物品向量
python train/train_item2vec.py --mode item2vec \
  --data data/behavior.csv \
  --user-col user_id --sequence-col sequence \
  --dim 128 --epochs 10 \
  --out model/item2vec_vectors.json

# Word2Vec（文本）：从文本语料训练词向量
python train/train_item2vec.py --mode word2vec \
  --data data/corpus.txt \
  --dim 128 --epochs 10 \
  --out model/word2vec_vectors.json
```

**无数据时**：不传 `--data` 或指向不存在路径，且未加 `--no-gen` 时，脚本会自动生成示例数据（Item2Vec 生成 `data/behavior.csv`，Word2Vec 生成 `data/corpus.txt`）并训练。

**输出**：JSON 文件，格式为 `"词或物品ID" -> [f1, f2, ...]`，可直接被 Golang 的 `LoadWord2VecFromMap` 或等价逻辑加载。

**常用参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | 必填 | `item2vec` 或 `word2vec` |
| `--data` | - | 训练数据路径；可省略以使用示例数据 |
| `--out` | `model/word2vec_vectors.json` | 输出 JSON 路径 |
| `--dim` | 128 | 向量维度 |
| `--epochs` | 10 | 训练轮数 |
| `--window` | 5 | 上下文窗口大小 |
| `--min-count` | 1 | 最小词/物品出现次数 |
| `--user-col` | `user_id` | Item2Vec CSV 用户列名 |
| `--sequence-col` | `sequence` | Item2Vec CSV 序列列名 |
| `--no-gen` | - | 不生成示例数据，数据不存在则报错 |

### 2.3 Item2Vec 训练数据格式

**文件**：`data/behavior.csv`（可自定义 `--data` 路径）

**格式**：CSV，两列 `user_id`、`sequence`。`sequence` 为**逗号分隔的物品 ID 列表**，表示该用户的行为序列。

**列说明**

| 列名 | 含义 |
|------|------|
| `user_id` | 用户 ID，如 `u1`、`u2` |
| `sequence` | 行为序列，逗号分隔物品 ID |

**生成规则**（无数据时自动生成）：
- 默认 500 个用户、200 个物品（`item_1` … `item_200`）
- 每用户随机 5～50 个行为，从 `item_1`…`item_200` **有放回**抽样组成序列
- `np.random.seed(42)` 可复现

**示例**

```text
user_id,sequence
"u1","item_42,item_17,item_89,item_42,item_3,..."
"u2","item_5,item_120,item_88,..."
"u3","item_1,item_1,item_200,item_50,..."
```

- `user_id`、`sequence` 的列名可通过 `--user-col`、`--sequence-col` 修改。
- 脚本按行读取，每行得到一个「句子」；所有句子组成语料，用 gensim `Word2Vec` 训练，物品 ID 即词。
- 若原始数据是「逐条行为日志」（如 `user_id, item_id, timestamp`），需先离线聚合为 `user_id -> 有序 item_id 序列`，再写成上述 CSV。

**完整说明**见 `python/data/README.md`。

### 2.4 Word2Vec（文本）训练数据格式

**文件**：`data/corpus.txt`（可自定义 `--data` 路径）

**格式**：纯文本，**每行一句**，按空格分词，UTF-8 编码。

**词表**（示例数据共 16 个词）：
`electronics smartphone tech mobile device laptop computer game sports music movie book fashion food travel`

**生成规则**（无数据时自动生成）：
- 每行随机选 3～15 个词（有放回），用空格拼接
- 默认 2000 行，`np.random.seed(42)` 可复现

**示例**

```text
electronics laptop game music fashion
smartphone tech device sports
book travel food electronics smartphone tech mobile
laptop computer game
...
```

脚本按行读取，按空格分词，每行得到一个「句子」，再训练 Word2Vec。

**完整说明**见 `python/data/README.md`。

### 2.5 训练流程小结

1. 准备数据：Item2Vec 用行为序列（或可聚合的日志）；Word2Vec 用文本语料。
2. 运行 `train_item2vec.py`，得到 `*_vectors.json`。
3. 在 Golang 中加载该 JSON，构建 `Word2VecModel`，接入 `Word2VecRecall`。

---

## 三、Golang 侧接入

### 3.1 加载模型

JSON 格式为 `map[string][]float64` 的序列化，例如：

```json
{
  "item_1": [0.01, -0.02, 0.03, ...],
  "item_2": [0.02, -0.01, 0.04, ...]
}
```

加载示例：

```go
import (
    "encoding/json"
    "os"
    "github.com/rushteam/reckit/model"
)

func loadWord2VecFromJSON(path string) (*model.Word2VecModel, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }
    var raw map[string]interface{}
    if err := json.Unmarshal(data, &raw); err != nil {
        return nil, err
    }
    return model.LoadWord2VecFromMap(raw)
}
```

### 3.2 文本模式（Word2Vec）

```go
w2v := loadWord2VecFromJSON("model/word2vec_vectors.json")

recall := &recall.Word2VecRecall{
    Model:     w2v,
    Store:     store,
    TopK:      20,
    Mode:      "text",
    TextField: "title", // 或 "description" / "tags"
}
```

- 用户侧：用最近点击物品的**文本**（如标题）拼成一段话，`EncodeText` 得到用户向量。
- 物品侧：用物品的 **title/description/tags** 编码为向量，再算相似度召回。

### 3.3 Item2Vec 模式（序列召回）

```go
w2v := loadWord2VecFromJSON("model/item2vec_vectors.json")

recall := &recall.Word2VecRecall{
    Model: w2v,
    Store: store,
    TopK:  20,
    Mode:  "sequence",
}
```

- 用户侧：用 `GetUserSequence` 或 `User.RecentClicks` 得到物品 ID 序列，`EncodeSequence` 得到用户向量。
- 物品侧：每个候选物品 ID 对应一个向量（即训练得到的物品向量），用 `GetWordVector(itemID)`，再算相似度召回。

注意：Item2Vec 时 `WordVectors` 的 key 必须为物品 ID，与行为序列、候选物品 ID 一致。

### 3.4 序列向量化与相似度

```go
// 用户行为序列 → 向量（Item2Vec）
seq := []string{"item_1", "item_2", "item_3"}
userVec := w2v.EncodeSequence(seq)

// 单个物品向量
itemVec := w2v.GetWordVector("item_1")

// 余弦相似度
sim := w2v.Similarity(userVec, itemVec)
```

---

## 四、存储接口 `Word2VecStore`

召回依赖存储实现以下接口：

```go
type Word2VecStore interface {
    GetItemText(ctx context.Context, itemID string) (string, error)
    GetItemTags(ctx context.Context, itemID string) ([]string, error)
    GetUserSequence(ctx context.Context, userID string, maxLen int) ([]string, error)
    GetAllItems(ctx context.Context) ([]string, error)
}
```

- **text 模式**：需要 `GetItemText` / `GetItemTags`、`GetAllItems`；`GetUserSequence` 可选。
- **Item2Vec 模式**：需要 `GetUserSequence`、`GetAllItems`；`GetItemText` / `GetItemTags` 在纯 Item2Vec 下可不使用，但接口仍要实现。

---

## 五、运行示例

```bash
# 1. 训练 Item2Vec 并导出 JSON（见 2.2）
python train/train_item2vec.py --mode item2vec --data data/behavior.csv ...

# 2. 运行 Golang 示例（含 Item2Vec 与加载 JSON）
go run ./examples/word2vec
```

示例中的 `MemoryWord2VecStore` 仅作演示；实际应对接 Redis/MySQL 等实现 `Word2VecStore`，并从 `train_item2vec.py` 生成的 JSON 加载模型。

---

## 六、参考

- 召回算法总览：`docs/RECALL_ALGORITHMS.md`
- 模型选型：`docs/MODEL_SELECTION.md`
- 示例代码：`examples/word2vec/main.go`
- 训练脚本：`python/train/train_item2vec.py`
