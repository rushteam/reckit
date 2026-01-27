# 召回模块专有名词字典

本文档定义了 Reckit 召回模块中所有统一后的专有名词及其含义，方便开发者查阅和使用。

## 目录

- [核心概念](#核心概念)
- [字段命名规范](#字段命名规范)
- [Label 命名规范](#label-命名规范)
- [接口命名规范](#接口命名规范)
- [数据流命名规范](#数据流命名规范)

---

## 核心概念

### 召回源（Recall Source）

**定义**：实现 `recall.Source` 接口的召回策略单元，负责从候选池中召回物品。

**统一命名**：所有召回源都通过 `Name()` 方法返回标识符，格式为 `recall.{type}`。

**示例**：
- `recall.two_tower` - 双塔召回
- `recall.youtube_dnn` - YouTube DNN 召回
- `recall.dssm` - DSSM 召回
- `recall.ann` / `recall.emb` - Embedding 向量召回
- `recall.content` - 内容召回
- `recall.u2i` - 用户协同过滤召回
- `recall.i2i` - 物品协同过滤召回

---

## 字段命名规范

### 1. TopK

**定义**：返回的 TopK 个物品数量。

**类型**：`int`

**统一命名**：`TopK`（所有召回源统一）

**默认值**：各召回源不同（10/20/100）

**使用示例**：
```go
type TwoTowerRecall struct {
    TopK int // ✅ 统一命名
}
```

---

### 2. Metric

**定义**：向量相似度或距离度量方式。

**类型**：`string`

**统一命名**：`Metric`（所有召回源统一）

**可选值**：
- `"cosine"` - 余弦相似度
- `"euclidean"` - 欧氏距离
- `"inner_product"` - 内积（点积）
- `"jaccard"` - Jaccard 相似度（Content 召回）

**默认值**：各召回源不同
- 双塔召回：`"inner_product"`
- DSSM 召回：`"cosine"`
- Content 召回：`"cosine"`

**使用示例**：
```go
type TwoTowerRecall struct {
    Metric string // ✅ 统一命名（原 SimilarityMetric → Metric）
}
```

---

### 3. Collection

**定义**：向量数据库中存储物品向量的集合名称。

**类型**：`string`

**统一命名**：`Collection`（所有向量召回源统一）

**默认值**：各召回源不同
- 双塔召回：`"item_embeddings"`
- YouTube DNN 召回：`"youtube_dnn_items"`
- DSSM 召回：`"dssm_docs"`

**使用示例**：
```go
type TwoTowerRecall struct {
    Collection string // ✅ 统一命名
}
```

---

### 4. UserEmbedding

**定义**：用户向量（Embedding），用于向量检索召回。

**类型**：`[]float64`

**统一命名**：
- **字段名**：`UserEmbedding`（所有召回源统一）
- **变量名**：`userEmbedding`（Go 代码统一）

**历史命名**：
- ❌ `UserVector` → ✅ `UserEmbedding`
- ❌ `userVector` → ✅ `userEmbedding`

**使用示例**：
```go
type ANN struct {
    UserEmbedding []float64 // ✅ 统一字段名
}

// 代码中使用
userEmbedding, err := r.runUserTower(ctx, userFeatures) // ✅ 统一变量名
```

---

### 5. QueryEmbedding

**定义**：查询向量（Embedding），用于 Query-Doc 语义召回（DSSM）。

**类型**：`[]float64`

**统一命名**：
- **字段名**：`QueryEmbedding`（DSSM 专用，语义区分）
- **变量名**：`queryEmbedding`（Go 代码统一）

**使用示例**：
```go
type DSSMRecall struct {
    QueryEmbeddingURL string // ✅ 查询向量 URL
}

// 代码中使用
queryEmbedding, err := r.fetchQueryEmbedding(ctx, qf) // ✅ 统一变量名
```

---

### 6. Endpoint

**定义**：HTTP 服务的端点 URL。

**类型**：`string`

**统一命名**：`Endpoint`（所有 HTTP 召回源统一）

**历史命名**：
- ❌ `UserEmbeddingURL` → ✅ `Endpoint`（YouTube DNN）
- ❌ `QueryEmbeddingURL` → ✅ `Endpoint`（DSSM）

**使用范围**：
- `YouTubeDNNRecall` - YouTube DNN 服务端点
- `DSSMRecall` - DSSM 服务端点
- `GraphRecall` - 图召回服务端点
- `RPCRecall` - RPC 召回服务端点

**使用示例**：
```go
type YouTubeDNNRecall struct {
    Endpoint string // ✅ 统一命名（原 UserEmbeddingURL）
}

type DSSMRecall struct {
    Endpoint string // ✅ 统一命名（原 QueryEmbeddingURL）
}
```

---

### 7. Timeout

**定义**：HTTP 请求的超时时间。

**类型**：`time.Duration`

**统一命名**：`Timeout`（所有 HTTP 召回源统一）

**默认值**：`5 * time.Second`

**使用示例**：
```go
type YouTubeDNNRecall struct {
    Timeout time.Duration // ✅ 统一命名
}
```

---

### 8. Client

**定义**：HTTP 客户端，用于发送 HTTP 请求。

**类型**：`*http.Client`

**统一命名**：`Client`（所有 HTTP 召回源统一）

**使用示例**：
```go
type DSSMRecall struct {
    Client *http.Client // ✅ 统一命名
}
```

---

### 9. Key / KeyPrefix

**定义**：存储系统中的 key 或 key 前缀，用于标识数据位置。

**类型**：`string`

**统一命名**：
- **简单 key**：`Key`（ANN, Hot）
- **key 前缀**：`KeyPrefix`（UserHistory）

**使用示例**：
```go
type Hot struct {
    Key string // ✅ 简单 key
}

type UserHistory struct {
    KeyPrefix string // ✅ key 前缀
}
```

---

### 10. FeatureExtractor

**定义**：特征提取器，用于从 `RecommendContext` 提取特征。

**类型**：`feature.FeatureExtractor` 接口或函数类型

**统一命名**：根据提取内容语义区分
- `UserFeatureExtractor` - 用户特征提取器
- `QueryFeatureExtractor` - 查询特征提取器（DSSM）
- `UserPreferencesExtractor` - 用户偏好提取器（Content）
- `UserEmbeddingExtractor` - 用户向量提取器（ANN, MF）
- `HistoryExtractor` - 历史序列提取器（YouTube DNN）

**使用示例**：
```go
type TwoTowerRecall struct {
    UserFeatureExtractor feature.FeatureExtractor // ✅ 用户特征
}

type DSSMRecall struct {
    QueryFeatureExtractor feature.FeatureExtractor // ✅ 查询特征（语义区分）
}
```

---

### 11. ContextKey

**定义**：从 `RecommendContext.UserProfile` 或 `RecommendContext.Params` 获取数据的 key。

**类型**：`string`

**统一命名**：根据数据语义命名
- `UserPreferencesKey` - 用户偏好 key（Content）
- `UserEmbeddingKey` - 用户向量 key（MF）
- `UserHistoryKey` - 用户历史 key（ItemBasedCF）

**使用示例**：
```go
type ContentRecall struct {
    UserPreferencesKey string // ✅ 语义明确
}
```

---

## Label 命名规范

### recall_source

**定义**：召回源标识，用于区分不同的召回策略。

**类型**：`utils.Label`

**值格式**：`recall.{type}`

**必填**：✅ 是（所有召回源必须设置）

**示例值**：
- `"two_tower"` - 双塔召回
- `"youtube_dnn"` - YouTube DNN 召回
- `"dssm"` - DSSM 召回
- `"ann"` - ANN 召回
- `"content"` - 内容召回
- `"u2i"` - 用户协同过滤
- `"i2i"` - 物品协同过滤

**使用示例**：
```go
item.PutLabel("recall_source", utils.Label{Value: "two_tower", Source: "recall"})
```

---

### recall_type

**定义**：召回类型，用于标识召回的技术类型。

**类型**：`utils.Label`

**可选**：可选（仅在需要时设置）

**示例值**：
- `"vector_search"` - 向量检索召回
- `"node2vec"` - 图嵌入召回

**使用示例**：
```go
item.PutLabel("recall_type", utils.Label{Value: "vector_search", Source: "recall"})
```

---

### recall_mode

**定义**：召回模式，用于标识同一召回源的不同工作模式。

**类型**：`utils.Label`

**可选**：可选（仅在需要时设置）

**示例值**：
- `"text"` - 文本模式（BERT, Word2Vec）
- `"sequence"` - 序列模式（Word2Vec）
- `"query"` - 查询模式（BERT）

**使用示例**：
```go
item.PutLabel("recall_mode", utils.Label{Value: "text", Source: "recall"})
```

---

### recall_metric

**定义**：距离度量方式，用于标识向量检索使用的度量方法。

**类型**：`utils.Label`

**可选**：可选（仅在设置 Metric 时添加）

**统一命名**：`recall_metric`（所有召回源统一）

**历史命名**：
- ❌ `content_metric` → ✅ `recall_metric`
- ❌ `ann_metric` → ✅ `recall_metric`

**示例值**：
- `"cosine"` - 余弦相似度
- `"euclidean"` - 欧氏距离
- `"inner_product"` - 内积
- `"jaccard"` - Jaccard 相似度

**使用示例**：
```go
if r.Metric != "" {
    item.PutLabel("recall_metric", utils.Label{Value: r.Metric, Source: "recall"})
}
```

---

### recall_collection

**定义**：向量集合名称，用于标识物品向量存储的集合。

**类型**：`utils.Label`

**可选**：可选（仅在设置 Collection 时添加）

**使用示例**：
```go
if r.Collection != "" {
    item.PutLabel("recall_collection", utils.Label{Value: r.Collection, Source: "recall"})
}
```

---

## 接口命名规范

### VectorService

**定义**：向量检索服务接口，用于向量相似度搜索。

**类型**：`core.VectorService`

**统一命名**：`VectorService`（所有向量召回源统一）

**使用示例**：
```go
type TwoTowerRecall struct {
    VectorService core.VectorService // ✅ 统一命名
}
```

---

### FeatureService

**定义**：特征服务接口，用于获取用户特征。

**类型**：`feature.FeatureService`

**统一命名**：`FeatureService`（所有需要特征服务的召回源统一）

**使用示例**：
```go
type TwoTowerRecall struct {
    FeatureService feature.FeatureService // ✅ 统一命名
}
```

---

## 数据流命名规范

### HistoryItemIDs

**定义**：用户历史行为序列（物品 ID 列表）。

**类型**：`[]string`

**统一命名**：
- **Go 代码**：`HistoryItemIDs` 或 `historyItemIDs`
- **JSON 字段**：`history_item_ids`（与 Python 服务一致）

**数据来源**：
- `UserProfile.RecentClicks` - 最近点击物品
- `HistoryExtractor.Extract()` - 历史提取器
- `Store.GetUserHistory()` - 存储中的历史数据

**使用示例**：
```go
// JSON 请求
type youtubeUserEmbReq struct {
    HistoryItemIDs []string `json:"history_item_ids"` // ✅ 统一 JSON 字段
}

// Go 代码
historyItemIDs := rctx.User.RecentClicks // ✅ 从 UserProfile 获取
```

---

## 字段统一对照表

| 概念 | 统一命名 | 历史命名（已废弃） | 使用范围 |
|------|---------|------------------|---------|
| 返回数量 | `TopK` | - | 所有召回源 |
| 距离度量 | `Metric` | `SimilarityMetric` | 所有召回源 |
| 向量集合 | `Collection` | - | 向量召回源 |
| 用户向量字段 | `UserEmbedding` | `UserVector` | ANN, MF |
| 用户向量变量 | `userEmbedding` | `userVector` | 所有向量召回源 |
| 查询向量变量 | `queryEmbedding` | `emb` | DSSM |
| HTTP 端点 | `Endpoint` | `UserEmbeddingURL`, `QueryEmbeddingURL` | YouTubeDNN, DSSM, Graph, RPC |
| HTTP 超时 | `Timeout` | - | HTTP 召回源 |
| HTTP 客户端 | `Client` | - | HTTP 召回源 |
| 存储 key | `Key` / `KeyPrefix` | - | 存储召回源 |
| 向量服务 | `VectorService` | - | 向量召回源 |
| 特征服务 | `FeatureService` | - | 特征召回源 |

---

## Label 统一对照表

| Label 名称 | 统一命名 | 历史命名（已废弃） | 必填 | 说明 |
|-----------|---------|------------------|------|------|
| 召回源标识 | `recall_source` | - | ✅ | 区分召回源 |
| 召回类型 | `recall_type` | - | ❌ | 技术类型 |
| 召回模式 | `recall_mode` | - | ❌ | 工作模式 |
| 距离度量 | `recall_metric` | `content_metric`, `ann_metric` | ❌ | 度量方式 |
| 向量集合 | `recall_collection` | - | ❌ | 集合名称 |

---

## 使用建议

### 1. 字段命名

- ✅ **统一使用**：`TopK`, `Metric`, `Collection`, `UserEmbedding`
- ✅ **语义区分**：`UserFeatureExtractor` vs `QueryFeatureExtractor`
- ✅ **HTTP 统一**：`Endpoint`, `Timeout`, `Client`

### 2. Label 使用

- ✅ **必填**：`recall_source`（所有召回源必须设置）
- ✅ **可选**：`recall_metric`, `recall_type`, `recall_mode`（按需添加）
- ✅ **统一命名**：使用 `recall_metric` 而非 `{source}_metric`

### 3. 变量命名

- ✅ **统一风格**：`userEmbedding`, `queryEmbedding`（Go 代码）
- ✅ **JSON 字段**：`history_item_ids`, `user_embedding`（与 Python 服务一致）

---

## 相关文档

- [召回算法文档](./RECALL_ALGORITHMS.md)
- [架构设计文档](./ARCHITECTURE.md)
- [接口与实现文档](./INTERFACES_AND_IMPLEMENTATIONS.md)
