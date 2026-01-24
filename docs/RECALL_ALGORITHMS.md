# Reckit 召回算法支持情况

## 算法支持概览

| 算法 | Reckit Node | 阶段 | 支持状态 | 实现文件 | Label |
|------|-------------|------|----------|----------|-------|
| User-CF | U2IRecall | Recall | ✅ 已支持 | `recall/collaborative_filtering.go` | `recall.u2i` |
| Item-CF | I2IRecall | Recall | ✅ 已支持 | `recall/collaborative_filtering.go` | `recall.i2i` |
| MF / ALS | MFRecall | Recall | ✅ 已支持 | `recall/matrix_factorization.go` | `recall.mf` |
| Embedding | EmbRecall | Recall | ✅ 已支持 | `recall/ann.go` | `recall.emb` |
| Content | ContentRecall | Recall | ✅ 已支持 | `recall/content.go` | `recall.content` |
| Word2Vec | Word2VecRecall | Recall | ✅ 已支持 | `recall/word2vec_recall.go` | `recall.word2vec` |
| RPC 召回 | RPCRecall | Recall | ✅ 已支持 | `recall/rpc_recall.go` | `recall.rpc` |

**所有算法均已实现并支持！** ✅

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

// 1. 创建 Milvus 服务
milvusService := vector.NewMilvusService(
    "localhost:19530",
    vector.WithMilvusAuth("root", "Milvus"),
    vector.WithMilvusDatabase("recommend"),
    vector.WithMilvusTimeout(30),
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
// 使用 Milvus 服务
milvusService := vector.NewMilvusService("localhost:19530")
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

### 6. Word2Vec → Word2VecRecall ✅

**实现类**: `Word2VecRecall`

**核心思想**: "将文本/序列转换为向量，通过向量相似度找到相似物品"

**使用场景**:
- 文本特征向量化：物品标题、描述、标签 → 向量
- 序列向量化：用户行为序列（点击的物品ID序列）→ 向量
- I2I 召回：基于物品文本相似度

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

**完整示例**: 参考 `examples/word2vec/main.go`

### 7. RPC 召回 → RPCRecall ✅

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

## 工程特征对比

| 算法 | 实时性 | 计算复杂度 | 可解释性 | 冷启动 | 工业使用 |
|------|--------|------------|----------|--------|----------|
| User-CF | 较差 | 高 | 强 | 差 | 离线分析 |
| Item-CF | 好 | 可控 | 强 | 中等 | 工业常青树 |
| MF/ALS | 好 | 低 | 中等 | 中等 | 广泛使用 |
| Embedding | 好 | 中等 | 弱 | 好 | 主流方案 |
| Content | 好 | 中等 | 强 | 好 | 冷启动首选 |
