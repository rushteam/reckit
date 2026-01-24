# 示例数据格式说明

本目录存放训练脚本使用的示例数据。数据文件不存在时，各训练脚本会**自动生成**对应格式的示例数据。本文档说明各文件的格式与生成规则。

---

## 1. train_data.csv（XGBoost / DeepFM）

**路径**：`python/data/train_data.csv`  
**用途**：`train_xgb.py`、`train_deepfm.py`  
**格式**：CSV，特征列与 `train/features.py` 的 `FEATURE_COLUMNS` 一致（带前缀），外加 `label`。

### 列说明

| 列名 | 含义 | 生成规则 |
|------|------|----------|
| `item_ctr` | 物品点击率 | `uniform(0.01, 0.5)` |
| `item_cvr` | 物品转化率 | `uniform(0.001, 0.1)` |
| `item_price` | 物品价格 | `uniform(10, 200)` |
| `user_age` | 用户年龄 | `randint(18, 60)` → float |
| `user_gender` | 用户性别 | `randint(0, 3)` → float（0=未知 / 1=男 / 2=女） |
| `cross_age_x_ctr` | 年龄×CTR | `user_age * item_ctr` |
| `cross_gender_x_price` | 性别×价格 | `user_gender * item_price` |
| `label` | 二分类标签 | 见下方公式 |

### 标签公式

```text
label_raw = 0.5 * item_ctr * 10 + 0.3 * item_cvr * 20 + 0.1 * user_age/100 + 0.05 * user_gender/2 + N(0, 0.1)
label = 1  if label_raw > 0.5  else 0
```

### 示例行

```csv
item_ctr,item_cvr,item_price,user_age,user_gender,cross_age_x_ctr,cross_gender_x_price,label
0.15,0.08,99.0,25.0,1.0,3.75,99.0,1
0.12,0.05,150.0,30.0,2.0,3.6,300.0,0
0.08,0.03,200.0,45.0,1.0,3.6,200.0,0
0.20,0.10,50.0,22.0,0.0,4.4,0.0,1
...
```

默认生成 1000 条，`np.random.seed(42)` 可复现。详见 `python/TRAINING_DATA.md`。

---

## 2. behavior.csv（Item2Vec）

**路径**：`python/data/behavior.csv`  
**用途**：`train_item2vec.py --mode item2vec`  
**格式**：CSV，两列 `user_id`、`sequence`。`sequence` 为**逗号分隔的物品 ID 列表**，表示该用户的行为序列。

### 列说明

| 列名 | 含义 |
|------|------|
| `user_id` | 用户 ID，如 `u1`、`u2` |
| `sequence` | 行为序列，逗号分隔物品 ID，如 `"item_1,item_2,item_3"` |

### 生成规则

- 默认 500 个用户、200 个物品（`item_1` … `item_200`）
- 每用户随机 5～50 个行为，从 `item_1`…`item_200` **有放回**抽样，组成一条序列
- `np.random.seed(42)` 可复现

### 示例

```csv
user_id,sequence
"u1","item_42,item_17,item_89,item_42,item_3,..."
"u2","item_5,item_120,item_88,..."
"u3","item_1,item_1,item_200,item_50,..."
...
```

详见 `docs/WORD2VEC_ITEM2VEC.md`。

---

## 3. corpus.txt（Word2Vec 文本）

**路径**：`python/data/corpus.txt`  
**用途**：`train_item2vec.py --mode word2vec`  
**格式**：纯文本，**每行一句**，按空格分词，UTF-8 编码。

### 词表（16 个词）

```text
electronics smartphone tech mobile device laptop computer game sports music movie book fashion food travel
```

### 生成规则

- 每行随机选 3～15 个词（有放回），用空格拼接
- 默认 2000 行
- `np.random.seed(42)` 可复现

### 示例

```text
electronics laptop game music fashion
smartphone tech device sports
book travel food electronics smartphone tech mobile
laptop computer game
...
```

详见 `docs/WORD2VEC_ITEM2VEC.md`。

---

## 汇总

| 数据类型 | 文件 | 用途 | 脚本 |
|----------|------|------|------|
| 排序/CTR | `train_data.csv` | XGBoost、DeepFM | `train_xgb.py`、`train_deepfm.py` |
| 行为序列 | `behavior.csv` | Item2Vec | `train_item2vec.py --mode item2vec` |
| 文本语料 | `corpus.txt` | Word2Vec | `train_item2vec.py --mode word2vec` |

生成逻辑见各训练脚本中的 `generate_sample_data` / `generate_behavior_csv` / `generate_corpus_txt`。
