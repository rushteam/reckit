# Reckit 召回算法支持情况

## 召回算法分类与实现状态

本文档按照工业界常见的召回算法分类，全面梳理 Reckit 的召回算法实现情况。

### 1. 基于行为的召回

这是最基础、最常用的召回方式，利用用户和物品的交互矩阵。

| 算法 | Reckit Node | 支持状态 | 实现文件 | Label | 适用场景 |
|------|-------------|----------|----------|-------|----------|
| **Item-CF** | I2IRecall | ✅ 已实现 | `recall/collaborative_filtering.go` | `recall.i2i` | 电商、物品更新不频繁 |
| **User-CF** | U2IRecall | ✅ 已实现 | `recall/collaborative_filtering.go` | `recall.u2i` | 新闻、短视频、兴趣变化快 |
| **Swing** | - | ❌ 未实现 | - | - | 改进的 Item-CF，惩罚热门物品 |
| **矩阵分解 (MF/ALS)** | MFRecall | ✅ 已实现 | `recall/matrix_factorization.go` | `recall.mf` | 隐式反馈召回 |

**实现度**：3/4 (75%)

**说明**：
- ✅ **Item-CF**：工业级召回的"常青树"，电商、内容流、短视频都在用
- ✅ **User-CF**：适合兴趣变化快的场景，但计算复杂度高，通常用于离线分析
- ✅ **矩阵分解**：将用户-物品交互矩阵分解为隐向量，计算复杂度低
- ❌ **Swing**：阿里系常用的改进版 Item-CF，通过惩罚热门物品挖掘更有说服力的相似物品

### 2. 向量化/深度学习召回

利用深度神经网络学习 Embedding，实现语义级别的召回。

| 算法 | Reckit Node | 支持状态 | 实现文件 | Label | 适用场景 |
|------|-------------|----------|----------|-------|----------|
| **双塔模型 (Two-Tower)** | TwoTowerRecall | ✅ 已实现 | `recall/two_tower_recall.go` | `recall.two_tower` | 大规模初筛、跨域召回 |
| **DSSM** | DSSMRecall | ✅ 已实现 | `recall/dssm_recall.go` | `recall.dssm` | 搜索推荐、语义匹配 |
| **YouTube DNN** | YouTubeDNNRecall | ✅ 已实现 | `recall/youtube_dnn_recall.go` | `recall.youtube_dnn` | 视频/内容流召回 |
| **多兴趣模型 (MIND/SDM)** | - | ❌ 未实现 | - | - | 多维度兴趣召回 |

**实现度**：3/4 (75%)

**说明**：
- ✅ **双塔模型**：工业界召回标准，QPS 高、延迟低，Item 向量可预计算
- ✅ **DSSM**：深度语义匹配，最早用于搜索，后广泛应用于推荐
- ✅ **YouTube DNN**：经典的召回模型，用用户历史观看序列作为输入
- ❌ **多兴趣模型**：一个用户生成多个向量表示不同维度的兴趣，匹配更精准

### 3. 基于内容的召回

不依赖用户行为，完全依赖物品本身的属性进行召回。

| 算法 | Reckit Node | 支持状态 | 实现文件 | Label | 适用场景 |
|------|-------------|----------|----------|-------|----------|
| **向量化召回** | Word2VecRecall, BERTRecall | ✅ 已实现 | `recall/word2vec_recall.go`, `recall/bert_recall.go` | `recall.word2vec`, `recall.bert` | 文本语义召回 |
| **标签/类目召回** | ContentRecall | ✅ 已实现 | `recall/content.go` | `recall.content` | 基于物品特征召回 |
| **知识图谱召回** | - | ❌ 未实现 | - | - | 利用实体关系召回 |

**实现度**：2/3 (67%)

**说明**：
- ✅ **向量化召回**：提取物品的文本、图像向量，或利用 TF-IDF，计算相似度
- ✅ **标签/类目召回**：召回相同 Tag、相同 Category 的物品
- ❌ **知识图谱召回**：利用实体关系（如：演员A -> 出演 -> 电影B）来挖掘关联

### 4. 基于图的召回

将用户和物品构建成二部图或异构图，利用图结构信息进行召回。

| 算法 | Reckit Node | 支持状态 | 实现文件 | Label | 适用场景 |
|------|-------------|----------|----------|-------|----------|
| **GraphSAGE / PinSage / Node2vec** | GraphRecall | ✅ 已实现 | `recall/graph_recall.go` | `recall.graph` | 社交推荐、关注页 |

**实现度**：1/1 (100%)

**说明**：
- ✅ **GraphSAGE/Node2vec**：在图上进行随机游走或采样，学习节点的向量表示，能非常好地利用全局的结构信息，挖掘出多跳的关联（比如：用户A买了尿布 -> 关联啤酒 -> 关联花生）

### 5. 其他策略召回

兜底策略和特殊场景召回。

| 算法 | Reckit Node | 支持状态 | 实现文件 | Label | 适用场景 |
|------|-------------|----------|----------|-------|----------|
| **热门/Trending** | Hot | ✅ 已实现 | `recall/hot.go` | `recall.hot` | 兜底策略、冷启动 |
| **地理位置召回 (LBS)** | - | ❌ 未实现 | - | - | 基于距离的推荐 |
| **复购召回** | - | ❌ 未实现 | - | - | 快消品、外卖场景 |

**实现度**：1/3 (33%)

**说明**：
- ✅ **热门/Trending**：兜底策略，解决冷启动和流量突发
- ❌ **地理位置召回 (LBS)**：基于距离的推荐（如抖音、美团）
- ❌ **复购召回**：针对快消品、外卖等场景，直接召回用户历史购买过的物品

### 6. 其他已实现的召回

| 算法 | Reckit Node | 支持状态 | 实现文件 | Label | 说明 |
|------|-------------|----------|----------|-------|------|
| **Embedding 向量检索** | EmbRecall / ANN | ✅ 已实现 | `recall/ann.go` | `recall.emb` | 通用向量检索 |
| **用户历史召回** | UserHistory | ✅ 已实现 | `recall/user_history.go` | `recall.user_history` | 基于行为历史 |
| **RPC 召回** | RPCRecall | ✅ 已实现 | `recall/rpc_recall.go` | `recall.rpc` | 外部服务召回 |

---

## 总体实现情况

**已实现**：13 个召回算法
**未实现**：5 个召回算法

**总体实现度**：13/18 (72.2%)

### 实现情况统计

| 分类 | 已实现 | 未实现 | 实现度 |
|------|--------|--------|--------|
| 基于行为的召回 | 3 | 1 (Swing) | 75% |
| 向量化/深度学习召回 | 3 | 1 (MIND/SDM) | 75% |
| 基于内容的召回 | 2 | 1 (知识图谱) | 67% |
| 基于图的召回 | 1 | 0 | 100% |
| 其他策略召回 | 1 | 2 (LBS, 复购) | 33% |
| 其他已实现 | 3 | 0 | 100% |
| **总计** | **13** | **5** | **72.2%** |

---

## 算法支持概览（详细）

| 算法 | Reckit Node | 阶段 | 支持状态 | 实现文件 | Label |
|------|-------------|------|----------|----------|-------|
| User-CF | U2IRecall | Recall | ✅ 已支持 | `recall/collaborative_filtering.go` | `recall.u2i` |
| Item-CF | I2IRecall | Recall | ✅ 已支持 | `recall/collaborative_filtering.go` | `recall.i2i` |
| MF / ALS | MFRecall | Recall | ✅ 已支持 | `recall/matrix_factorization.go` | `recall.mf` |
| Embedding | EmbRecall | Recall | ✅ 已支持 | `recall/ann.go` | `recall.emb` |
| Content | ContentRecall | Recall | ✅ 已支持 | `recall/content.go` | `recall.content` |
| Word2Vec | Word2VecRecall | Recall | ✅ 已支持 | `recall/word2vec_recall.go` | `recall.word2vec` |
| BERT | BERTRecall | Recall | ✅ 已支持 | `recall/bert_recall.go` | `recall.bert` |
| Two-Tower | TwoTowerRecall | Recall | ✅ 已支持 | `recall/two_tower_recall.go` | `recall.two_tower` |
| DSSM | DSSMRecall | Recall | ✅ 已支持 | `recall/dssm_recall.go` | `recall.dssm` |
| YouTube DNN | YouTubeDNNRecall | Recall | ✅ 已支持 | `recall/youtube_dnn_recall.go` | `recall.youtube_dnn` |
| Graph/Node2vec | GraphRecall | Recall | ✅ 已支持 | `recall/graph_recall.go` | `recall.graph` |
| Hot | Hot | Recall | ✅ 已支持 | `recall/hot.go` | `recall.hot` |
| UserHistory | UserHistory | Recall | ✅ 已支持 | `recall/user_history.go` | `recall.user_history` |
| RPC 召回 | RPCRecall | Recall | ✅ 已支持 | `recall/rpc_recall.go` | `recall.rpc` |
| Swing | - | Recall | ❌ 未实现 | - | - |
| MIND/SDM | - | Recall | ❌ 未实现 | - | - |
| 知识图谱 | - | Recall | ❌ 未实现 | - | - |
| LBS 召回 | - | Recall | ❌ 未实现 | - | - |
| 复购召回 | - | Recall | ❌ 未实现 | - | - |

## 详细说明

### 1. User-CF → U2IRecall ✅

**实现类**: `UserBasedCF` / `U2IRecall`

**核心思想**: "兴趣相似的用户，喜欢相似的物品"

**使用示例**:
```go
config := &core.DefaultRecallConfig{}
u2i := &recall.U2IRecall{
    Store:                cfStore,
    TopKSimilarUsers:     10,
    TopKItems:            5,
    SimilarityCalculator: &recall.CosineSimilarity{},
    Config:                config,
}
```

**Label**: `recall.u2i`

### 2. Item-CF → I2IRecall ✅

**实现类**: `ItemBasedCF` / `I2IRecall`

**核心思想**: "被同一批用户喜欢的物品，相互相似"

**使用示例**:
```go
config := &core.DefaultRecallConfig{}
i2i := &recall.I2IRecall{
    Store:                cfStore,
    TopKSimilarItems:     10,
    TopKItems:            5,
    SimilarityCalculator: &recall.CosineSimilarity{},
    Config:                config,
}
```

**Label**: `recall.i2i`

### 3. MF / ALS → MFRecall ✅

**实现类**: `MFRecall`

**核心思想**: 将用户-物品交互矩阵分解为用户隐向量和物品隐向量，预测分数 = 用户隐向量 · 物品隐向量

**算法类型**:
- MF (Matrix Factorization): 基础矩阵分解
- ALS (Alternating Least Squares): 交替最小二乘法
- SVD (Singular Value Decomposition): 奇异值分解

**使用示例**:
```go
mfRecall := &recall.MFRecall{
    Store: mfStore,
    TopK:  20,
    UserVectorKey: "user_vector", // 从 Context 获取
}
```

**存储适配器**:
```go
mfStore := recall.NewStoreMFAdapter(memStore, "mf")
```

**Label**: `recall.mf`

### 4. Embedding → EmbRecall ✅

**实现类**: `ANN` / `EmbRecall`

**核心思想**: 基于 Embedding 向量的近似最近邻检索

**详细文档**: 请参考 [Embedding 能力抽象](./EMBEDDING_ABSTRACT.md)

**使用示例**:

```go
import (
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/vector"
)

// 1. 创建 Milvus 服务（扩展包）
import milvus "github.com/rushteam/reckit/ext/vector/milvus"
// 安装：go get github.com/rushteam/reckit/ext/vector/milvus

milvusService := milvus.NewMilvusService(
    "localhost:19530",
    milvus.WithMilvusAuth("root", "Milvus"),
    milvus.WithMilvusDatabase("recommend"),
    milvus.WithMilvusTimeout(30),
)
defer milvusService.Close()

// 2. 创建适配器
adapter := vector.NewVectorStoreAdapter(milvusService, "items")

// 3. 创建 Embedding 召回
userVector := []float64{0.1, 0.2, 0.3, ...} // 用户向量
embRecall := &recall.EmbRecall{
    Store:      adapter,
    TopK:       20,
    Metric:     "cosine",
    UserVector: userVector, // 直接提供用户向量
}

// 4. 执行召回
items, err := embRecall.Recall(ctx, rctx)
```

**从 Context 获取用户向量**:

```go
rctx := &core.RecommendContext{
    UserID: "user_123",
    Scene:  "feed",
    UserProfile: map[string]any{
        "user_vector": []float64{0.1, 0.2, 0.3, ...},
    },
}

embRecall := &recall.EmbRecall{
    Store:  adapter,
    TopK:   20,
    Metric: "cosine",
    // 不设置 UserVector，会从 rctx.UserProfile["user_vector"] 获取
}
```

**使用自定义向量提取器**:

```go
embRecall := &recall.EmbRecall{
    Store:  adapter,
    TopK:   20,
    Metric: "cosine",
    UserVectorExtractor: func(rctx *core.RecommendContext) []float64 {
        // 自定义提取逻辑
        if rctx.UserProfile == nil {
            return nil
        }
        if uv, ok := rctx.UserProfile["user_vector"]; ok {
            if vec, ok := uv.([]float64); ok {
                return vec
            }
        }
        return nil
    },
}
```

**集成到 Fanout（多路召回）**:

```go
fanout := &recall.Fanout{
    Sources: []recall.Source{
        &recall.Hot{IDs: []string{"1", "2", "3"}},
        embRecall, // Embedding 召回
        &recall.I2IRecall{...},
    },
    Dedup:         true,
    Timeout:       2 * time.Second,
    MaxConcurrent: 5,
    MergeStrategy: &recall.PriorityMergeStrategy{},
}
```

**存储适配器**:

```go
// 使用 Milvus 服务（扩展包）
import milvus "github.com/rushteam/reckit/ext/vector/milvus"
// 安装：go get github.com/rushteam/reckit/ext/vector/milvus

milvusService := milvus.NewMilvusService("localhost:19530")
adapter := vector.NewVectorStoreAdapter(milvusService, "items")

// 使用适配器
embRecall := &recall.EmbRecall{
    Store: adapter,
    // ...
}
```

**Label**: `recall.emb`

**完整示例**: 参考 `examples/milvus_ann/main.go`

### 5. Content → ContentRecall ✅

**实现类**: `ContentRecall`

**核心思想**: "用户喜欢具有某些特征的物品，推荐具有相似特征的其他物品"

**使用示例**:
```go
contentRecall := &recall.ContentRecall{
    Store:            contentStore,
    TopK:             20,
    SimilarityMetric: "cosine", // cosine / jaccard
    UserPreferencesKey: "user_preferences",
}
```

**存储适配器**:
```go
contentStore := recall.NewStoreContentAdapter(memStore, "content")
```

**Label**: `recall.content`

### 6. Word2Vec / Item2Vec → Word2VecRecall ✅

**实现类**: `Word2VecRecall`

**核心思想**: "将文本/序列转换为向量，通过向量相似度找到相似物品"

**两种模式**:

| 模式 | 说明 | 用户向量 | 物品向量 |
|------|------|----------|----------|
| **text** (Word2Vec) | 文本词为「词」 | 最近点击物品的文本编码 | 物品标题/标签等编码 |
| **sequence** (Item2Vec) | 物品 ID 为「词」 | 用户行为序列（物品 ID 列表）编码 | 物品 ID 向量 |

**使用场景**:
- 文本特征向量化：物品标题、描述、标签 → 向量（Word2Vec）
- 序列向量化：用户行为序列（点击的物品ID序列）→ 向量（**Item2Vec**）
- I2I 召回：基于物品文本相似度或物品序列相似度

**Python 训练**: 使用 `python/train/train_item2vec.py`。Item2Vec 需用户行为序列 CSV；Word2Vec 需文本语料。输出 JSON 供 Golang `LoadWord2VecFromMap` 加载。详见 **`docs/WORD2VEC_ITEM2VEC.md`**。

**使用示例**:

```go
import "github.com/rushteam/reckit/model"
import "github.com/rushteam/reckit/recall"

// 1. 创建 Word2Vec 模型（从预训练的词向量）
wordVectors := map[string][]float64{
    "electronics": []float64{0.1, 0.2, 0.3, 0.4},
    "smartphone":  []float64{0.2, 0.3, 0.4, 0.5},
    // ... 更多词向量
}
w2vModel := model.NewWord2VecModel(wordVectors, 128)

// 2. 创建 Word2Vec 召回（基于文本）
word2vecRecall := &recall.Word2VecRecall{
    Model:     w2vModel,
    Store:     word2vecStore,
    TopK:      20,
    Mode:      "text",      // text 或 sequence
    TextField: "title",     // title / description / tags
}

// 3. 创建 Word2Vec 召回（基于序列）
sequenceRecall := &recall.Word2VecRecall{
    Model: w2vModel,
    Store: word2vecStore,
    TopK:  20,
    Mode:  "sequence", // 基于用户行为序列
}
```

**存储接口** (`Word2VecStore`):

```go
type Word2VecStore interface {
    GetItemText(ctx context.Context, itemID string) (string, error)
    GetItemTags(ctx context.Context, itemID string) ([]string, error)
    GetUserSequence(ctx context.Context, userID string, maxLen int) ([]string, error)
    GetAllItems(ctx context.Context) ([]string, error)
}
```

**文本向量化**:

```go
// 文本 → 向量
text := "electronics smartphone tech"
vector := w2vModel.EncodeText(text)

// 词列表 → 向量
words := []string{"electronics", "smartphone", "tech"}
vector := w2vModel.EncodeWords(words)
```

**序列向量化**:

```go
// 用户行为序列 → 向量
sequence := []string{"item_1", "item_2", "item_3"}
userVector := w2vModel.EncodeSequence(sequence)
```

**Label**: `recall.word2vec`

**完整示例**: `examples/word2vec/main.go`（含 Word2Vec 文本模式与 Item2Vec 序列模式、从 JSON 加载模型）

**文档**: `docs/WORD2VEC_ITEM2VEC.md`（概念、Python 训练方法、数据格式、Golang 接入）

### 7. BERT → BERTRecall ✅

**实现类**: `BERTRecall`

**核心思想**: "使用 BERT 将文本编码为语义向量，通过向量相似度找到语义相似的物品"

**使用场景**:
- 文本语义召回：基于物品标题、描述的语义相似度
- 搜索推荐：用户查询与物品文本的语义匹配
- I2I 召回：基于物品文本语义相似度

**使用示例**:

```go
import "github.com/rushteam/reckit/model"
import "github.com/rushteam/reckit/recall"
import "github.com/rushteam/reckit/service"

// 1. 创建 BERT 服务客户端（使用 TorchServe 或 TensorFlow Serving）
torchServeClient := service.NewTorchServeClient(
    "http://localhost:8080", // TorchServe 端点
    "bert-base",              // 模型名称
    service.WithTorchServeTimeout(5*time.Second),
)

// 2. 创建 BERT 模型
bertModel := model.NewBERTModel(torchServeClient, 768).
    WithModelName("bert-base").
    WithMaxLength(512).
    WithPoolingStrategy("cls") // cls / mean / max

// 3. 创建 BERT 召回（基于文本）
bertRecall := &recall.BERTRecall{
    Model:     bertModel,
    Store:     bertStore,
    TopK:      20,
    Mode:      "text",      // text 或 query
    TextField: "title",     // title / description / tags
    BatchSize: 32,          // 批量编码大小（提高效率）
}

// 4. 创建 BERT 召回（基于查询）
queryRecall := &recall.BERTRecall{
    Model:     bertModel,
    Store:     bertStore,
    TopK:      20,
    Mode:      "query",     // 基于用户查询
    TextField: "title",
    BatchSize: 32,
}
```

**存储接口** (`BERTStore`):

```go
type BERTStore interface {
    GetItemText(ctx context.Context, itemID string) (string, error)
    GetItemTags(ctx context.Context, itemID string) ([]string, error)
    GetUserQuery(ctx context.Context, userID string) (string, error)
    GetAllItems(ctx context.Context) ([]string, error)
}
```

**文本编码**:

```go
// 单个文本编码
vector, _ := bertModel.EncodeText(ctx, "electronics smartphone tech")

// 批量编码（提高效率）
vectors, _ := bertModel.EncodeTexts(ctx, []string{"text1", "text2", "text3"})
```

**语义相似度**:

```go
// 计算两个向量的余弦相似度
similarity := bertModel.Similarity(vec1, vec2)
```

**Label**: `recall.bert`

**完整示例**: 参考 `examples/bert/main.go`

**注意事项**:
- BERT 模型需要通过外部服务（TorchServe/TensorFlow Serving）进行推理
- 支持批量编码以提高效率（BatchSize 参数）
- 适合文本丰富的场景，语义理解能力强

### 8. RPC 召回 → RPCRecall ✅

**实现类**: `RPCRecall`

**核心思想**: 通过 RPC/HTTP 调用外部召回服务，支持微服务架构和第三方召回 API

**使用场景**:
- 调用远程召回服务（Python/Java 等实现）
- 调用微服务架构中的召回服务
- 调用第三方召回 API
- 集成已有的召回系统

**使用示例**:

```go
// 基础用法（使用默认请求/响应格式）
rpcRecall := recall.NewRPCRecall(
    "http://localhost:8080/recall", // 召回服务端点
    2*time.Second,                   // 超时时间
).WithTopK(20)

// 在 Fanout 中使用
fanout := &recall.Fanout{
    Sources: []recall.Source{
        rpcRecall,
        &recall.Hot{IDs: []string{"1", "2", "3"}},
    },
    Dedup:         true,
    Timeout:       2 * time.Second,
    MaxConcurrent: 5,
    MergeStrategy: &recall.PriorityMergeStrategy{},
}
```

**默认请求格式（JSON）**:

```json
{
  "user_id": "user_123",
  "top_k": 20,
  "scene": "feed",
  "user_profile": {...},
  "realtime": {...}
}
```

**默认响应格式（JSON）**:

```json
{
  "items": [
    {"id": "item_1", "score": 0.95, "features": {...}, "meta": {...}},
    {"id": "item_2", "score": 0.87, "features": {...}, "meta": {...}}
  ]
}
```

或者简化格式：

```json
{
  "item_ids": ["item_1", "item_2", "item_3"],
  "scores": [0.95, 0.87, 0.82]
}
```

**Label**: `recall.rpc`

**工程特征**:
- 实时性：取决于远程服务响应时间
- 可扩展性：强（支持任意远程服务）
- 灵活性：高（支持自定义请求/响应格式）
- 适用场景：微服务架构、第三方集成

## 存储接口

### CFStore（协同过滤）
- `GetUserItems`: 获取用户交互物品
- `GetItemUsers`: 获取物品交互用户
- `GetAllUsers`: 获取所有用户
- `GetAllItems`: 获取所有物品

### MFStore（矩阵分解）
- `GetUserVector`: 获取用户隐向量
- `GetItemVector`: 获取物品隐向量
- `GetAllItemVectors`: 获取所有物品隐向量

### VectorStore（Embedding）
- `GetVector`: 获取单个物品向量
- `ListVectors`: 获取所有物品向量

### ContentStore（内容推荐）
- `GetItemFeatures`: 获取物品特征
- `GetUserPreferences`: 获取用户偏好
- `GetAllItems`: 获取所有物品

## 集成到 Pipeline

所有召回算法都可以集成到 Pipeline 中：

```go
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{
            Sources: []recall.Source{
                &recall.U2IRecall{...},      // User-CF
                &recall.I2IRecall{...},      // Item-CF
                &recall.MFRecall{...},       // MF/ALS
                &recall.EmbRecall{...},      // Embedding
                &recall.ContentRecall{...},   // Content
                recall.NewRPCRecall("http://localhost:8080/recall", 2*time.Second), // RPC 召回
            },
            MergeStrategy: &recall.PriorityMergeStrategy{},
        },
        // 其他节点...
    },
}
```

## 多路并发召回与融合去重

Reckit 通过 `Fanout` 节点实现多路召回源的并发执行和结果融合，这是工业级推荐系统的核心能力。

### 并发召回机制

#### 核心特性

- ✅ **多路并发执行**：使用 `golang.org/x/sync/errgroup` 实现多个召回源的并发执行
- ✅ **并发限流控制**：通过 `MaxConcurrent` 参数控制最大并发数，防止资源耗尽
- ✅ **独立超时控制**：每个召回源可设置独立的超时时间，超时不影响其他召回源
- ✅ **容错机制**：单个召回源失败不会中断整个召回流程
- ✅ **线程安全**：使用 `sync.Mutex` 保护共享数据结构

#### 配置参数

```go
fanout := &recall.Fanout{
    Sources: []recall.Source{
        &recall.U2IRecall{...},
        &recall.I2IRecall{...},
        &recall.MFRecall{...},
    },
    Dedup:         true,              // 是否启用去重
    Timeout:       2 * time.Second,   // 每个召回源的超时时间
    MaxConcurrent: 5,                 // 最大并发数（0 表示无限制）
    MergeStrategy: "priority",        // 合并策略
}
```

#### 并发执行流程

1. **启动阶段**：遍历所有召回源，为每个召回源创建独立的 goroutine
2. **限流控制**：如果设置了 `MaxConcurrent`，通过 semaphore（channel）控制并发数
3. **超时控制**：为每个召回源创建带超时的 context，超时后自动取消
4. **结果收集**：使用互斥锁保护共享结果集合，安全地收集各召回源的结果
5. **错误处理**：单个召回源失败或超时返回空结果，不影响其他召回源继续执行

### 融合去重策略

#### 三种合并策略

##### 1. `first`（默认策略）

- **去重规则**：按物品 ID 去重，保留第一个出现的物品
- **Label 处理**：重复物品的 labels 会合并到第一个物品上
- **适用场景**：简单的去重需求，不关心召回源优先级

```go
fanout := &recall.Fanout{
    Sources: []recall.Source{...},
    Dedup:         true,
    MergeStrategy: "first",  // 或默认不设置
}
```

##### 2. `union`（并集策略）

- **去重规则**：不去重，保留所有召回源的结果
- **Label 处理**：每个物品保留各自的 labels
- **适用场景**：需要保留所有来源信息，用于分析召回效果或调试

```go
fanout := &recall.Fanout{
    Sources: []recall.Source{...},
    Dedup:         false,  // 或设置为 true 但使用 union 策略
    MergeStrategy: "union",
}
```

##### 3. `priority`（优先级策略）

- **去重规则**：按优先级去重，优先级由 `Sources` 数组的索引决定（索引越小优先级越高）
- **Label 处理**：相同 ID 的物品出现时，保留优先级更高的物品，优先级低的物品的 labels 会合并到优先级高的物品上
- **适用场景**：需要控制召回源优先级，确保重要召回源的结果优先保留

```go
fanout := &recall.Fanout{
    Sources: []recall.Source{
        &recall.Hot{...},        // 优先级 0（最高）
        &recall.I2IRecall{...},  // 优先级 1
        &recall.U2IRecall{...},  // 优先级 2
    },
    Dedup:         true,
    MergeStrategy: "priority",
}
```

#### 去重实现细节

- **去重算法**：使用 `map[string]*core.Item` 实现 O(1) 时间复杂度的去重
- **Label 记录**：每个物品自动记录召回来源信息：
  - `recall_source`: 召回源名称（如 "recall.u2i"、"recall.i2i"）
  - `recall_priority`: 优先级（0-9，对应 Sources 数组索引）
- **Label 合并**：去重时自动合并重复物品的 labels，保留完整的召回轨迹

#### 使用示例

```go
// 多路并发召回示例
fanout := &recall.Fanout{
    Sources: []recall.Source{
        &recall.Hot{
            IDs: []string{"1", "2", "3", "4", "5"},
        },
        &recall.U2IRecall{
            Store:            cfStore,
            TopKSimilarUsers: 10,
            TopKItems:        20,
            SimilarityMetric: "cosine",
        },
        &recall.I2IRecall{
            Store:            cfStore,
            TopKSimilarItems: 10,
            TopKItems:        20,
            SimilarityMetric: "cosine",
        },
        &recall.MFRecall{
            Store: mfStore,
            TopK:  20,
        },
    },
    Dedup:         true,
    Timeout:       2 * time.Second,
    MaxConcurrent: 5,
    MergeStrategy: "priority",
}

// 在 Pipeline 中使用
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        fanout,
        // 其他节点...
    },
}

items, err := p.Run(ctx, rctx, nil)
```

### 性能与最佳实践

#### 性能优化建议

1. **并发数设置**：根据召回源的响应时间和系统资源合理设置 `MaxConcurrent`，避免过多并发导致资源竞争
2. **超时设置**：为每个召回源设置合理的超时时间，避免慢召回源影响整体响应时间
3. **优先级设计**：将响应快、质量高的召回源放在 `Sources` 数组前面，提高优先级策略的效果
4. **去重策略选择**：
   - 生产环境推荐使用 `priority` 策略，保证重要召回源的结果优先
   - 调试分析时使用 `union` 策略，查看所有召回源的结果
   - 简单场景使用 `first` 策略

#### 监控与调试

- 通过 `recall_source` label 追踪每个物品来自哪个召回源
- 通过 `recall_priority` label 了解物品的优先级信息
- 使用 `union` 策略可以同时看到所有召回源的结果，便于分析召回效果

## 未实现的召回算法

### 1. Swing 召回

**状态**：❌ 未实现

**描述**：Swing 是阿里系常用的改进版 Item-CF 算法。它通过惩罚"热门物品"带来的虚假关联，能挖掘出更有说服力的相似物品，是目前业界非常主流的 CF 变种。

**核心思想**：
- 传统 Item-CF：如果两个物品被很多用户同时喜欢，则认为它们相似
- Swing：对热门物品进行惩罚，避免热门物品带来的虚假关联

**适用场景**：
- 电商场景（物品更新不频繁）
- 需要挖掘长尾物品的相似关系
- 热门效应明显的场景

**实现计划**：
- 在 `ItemBasedCF` 基础上扩展
- 添加热门物品惩罚机制
- 支持 Swing 相似度计算

---

### 2. 多兴趣模型 (MIND/SDM)

**状态**：❌ 未实现

**描述**：考虑到用户不仅仅只有一个兴趣点（比如既喜欢鞋子又喜欢苹果），这些模型用一个用户生成多个向量来表示不同维度的兴趣，匹配更精准。

**核心思想**：
- 传统双塔：一个用户 → 一个用户向量
- 多兴趣模型：一个用户 → 多个兴趣向量（如：鞋子兴趣、苹果兴趣、科技兴趣）

**适用场景**：
- 用户兴趣多样化的场景
- 需要捕捉用户多个兴趣维度的场景
- 提升召回精准度

**实现计划**：
- 扩展 `TwoTowerRecall` 支持多兴趣向量
- 或创建新的 `MultiInterestRecall`
- 支持兴趣向量聚合和匹配

---

### 3. 知识图谱召回

**状态**：❌ 未实现

**描述**：利用实体关系（如：演员A -> 出演 -> 电影B）来挖掘关联，进行召回。

**核心思想**：
- 构建知识图谱（实体-关系-实体）
- 通过图遍历找到关联物品
- 利用实体关系进行召回

**适用场景**：
- 内容推荐（电影、音乐、书籍）
- 需要利用实体关系的场景
- 知识密集型推荐

**实现计划**：
- 定义知识图谱存储接口
- 实现图遍历算法
- 支持多跳关系召回

---

### 4. 地理位置召回 (LBS)

**状态**：❌ 未实现

**描述**：基于距离的推荐（如抖音、美团），根据用户地理位置召回附近的物品。

**核心思想**：
- 获取用户地理位置（经纬度）
- 计算物品与用户的距离
- 按距离排序召回

**适用场景**：
- 本地生活服务（美团、大众点评）
- 地理位置相关的推荐（抖音同城）
- O2O 场景

**实现计划**：
- 定义地理位置存储接口
- 实现距离计算（Haversine 公式）
- 支持地理围栏和距离排序

---

### 5. 复购召回

**状态**：❌ 未实现

**描述**：针对快消品、外卖等场景，直接召回用户历史购买过的物品。

**核心思想**：
- 获取用户历史购买记录
- 根据购买频率、时间等召回
- 适合高频复购场景

**适用场景**：
- 快消品推荐
- 外卖推荐
- 高频复购场景

**实现计划**：
- 扩展 `UserHistory` 支持购买历史
- 或创建新的 `RepurchaseRecall`
- 支持购买频率和时间权重

---

## 工程特征对比

| 算法 | 实时性 | 计算复杂度 | 可解释性 | 冷启动 | 工业使用 |
|------|--------|------------|----------|--------|----------|
| User-CF | 较差 | 高 | 强 | 差 | 离线分析 |
| Item-CF | 好 | 可控 | 强 | 中等 | 工业常青树 |
| Swing | 好 | 中等 | 强 | 中等 | 阿里系主流 |
| MF/ALS | 好 | 低 | 中等 | 中等 | 广泛使用 |
| Embedding | 好 | 中等 | 弱 | 好 | 主流方案 |
| Content | 好 | 中等 | 强 | 好 | 冷启动首选 |
| Two-Tower | 好 | 低 | 弱 | 好 | 工业标准 |
| YouTube DNN | 好 | 中等 | 弱 | 中等 | 视频流 |
| MIND/SDM | 好 | 中等 | 弱 | 好 | 多兴趣场景 |
| Graph/Node2vec | 中等 | 高 | 中等 | 中等 | 社交推荐 |
| LBS | 好 | 低 | 强 | 好 | 本地服务 |
| 复购 | 好 | 低 | 强 | 差 | 快消品 |
