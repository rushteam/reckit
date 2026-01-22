# Reckit 接口与实现完整分析

本文档按照推荐系统流程顺序，列出所有接口、已有实现及其作用。

## 推荐流程概览

```
召回 (Recall) → 过滤 (Filter) → 特征注入 (Feature Enrichment) → 排序 (Rank) → Top-N 截断 → 多样性重排 (Diversity)
```

---

## 1. 召回阶段 (Recall)

### 1.1 核心接口

#### `pipeline.Node` - Pipeline 节点基础接口
**位置**: `pipeline/node.go`

```go
type Node interface {
    Name() string
    Kind() Kind  // recall / filter / rank / rerank / postprocess
    Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error)
}
```

**作用**: 所有 Pipeline 节点的统一接口，支持链式处理。

#### `recall.Source` - 召回源接口
**位置**: `recall/source.go`

```go
type Source interface {
    Name() string
    Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error)
}
```

**作用**: 定义召回源的标准接口，支持多种召回算法实现。

**已有实现**:
- ✅ `recall.Hot` - 热门召回
- ✅ `recall.U2IRecall` - 用户协同过滤召回
- ✅ `recall.I2IRecall` - 物品协同过滤召回
- ✅ `recall.MFRecall` - 矩阵分解召回
- ✅ `recall.ANN` / `recall.EmbRecall` - 向量检索召回
- ✅ `recall.ContentRecall` - 内容推荐召回
- ✅ `recall.UserHistory` - 用户历史召回
- ✅ `recall.RPCRecall` - RPC 召回（外部服务）

#### `recall.MergeStrategy` - 合并策略接口
**位置**: `recall/fanout.go`

```go
type MergeStrategy interface {
    Merge(items []*core.Item, dedup bool) []*core.Item
}
```

**作用**: 定义多路召回结果的合并策略。

**已有实现**:
- ✅ `FirstMergeStrategy` - 按 ID 去重，保留第一个出现的
- ✅ `UnionMergeStrategy` - 并集策略，不去重
- ✅ `PriorityMergeStrategy` - 按优先级合并

#### `recall.ErrorHandler` - 错误处理策略接口
**位置**: `recall/fanout.go`

```go
type ErrorHandler interface {
    HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error)
}
```

**作用**: 定义召回源失败时的错误处理策略。

**已有实现**:
- ✅ `IgnoreErrorHandler` - 忽略错误，返回空结果
- ✅ `RetryErrorHandler` - 重试策略（示例实现）
- ✅ `FallbackErrorHandler` - 降级策略，使用备用召回源

#### `recall.SimilarityCalculator` - 相似度计算接口
**位置**: `recall/collaborative_filtering.go`

```go
type SimilarityCalculator interface {
    Calculate(x, y []float64) float64
}
```

**作用**: 定义相似度计算方法，用于协同过滤。

**已有实现**:
- ✅ `CosineSimilarity` - 余弦相似度
- ✅ `PearsonCorrelation` - 皮尔逊相关系数

#### `recall.CFStore` - 协同过滤存储接口
**位置**: `recall/collaborative_filtering.go`

```go
type CFStore interface {
    GetUserItems(ctx context.Context, userID string) (map[string]float64, error)
    GetItemUsers(ctx context.Context, itemID string) (map[string]float64, error)
    GetAllUsers(ctx context.Context) ([]string, error)
    GetAllItems(ctx context.Context) ([]string, error)
}
```

**作用**: 定义协同过滤所需的数据存储接口。

**已有实现**:
- ✅ `StoreCFAdapter` - 将 `store.Store` 适配为 `CFStore`

#### `recall.VectorStore` - 向量存储接口
**位置**: `recall/ann.go`

```go
type VectorStore interface {
    GetVector(ctx context.Context, itemID string) ([]float64, error)
    ListVectors(ctx context.Context) (map[string][]float64, error)
    Search(ctx context.Context, vector []float64, topK int, metric string) ([]string, []float64, error)
}
```

**作用**: 定义向量存储接口，用于向量检索召回。

**已有实现**:
- ✅ `VectorStoreAdapter` - 将 `vector.ANNService` 适配为 `VectorStore`

#### `recall.ContentStore` - 内容推荐存储接口
**位置**: `recall/content.go`

```go
type ContentStore interface {
    GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)
    GetUserPreferences(ctx context.Context, userID string) (map[string]float64, error)
    GetSimilarItems(ctx context.Context, itemFeatures map[string]float64, topK int) ([]string, error)
    GetAllItems(ctx context.Context) ([]string, error)
}
```

**作用**: 定义内容推荐所需的数据存储接口。

**已有实现**:
- ✅ `StoreContentAdapter` - 将 `store.Store` 适配为 `ContentStore`

#### `recall.MFStore` - 矩阵分解存储接口
**位置**: `recall/matrix_factorization.go`

```go
type MFStore interface {
    GetUserVector(ctx context.Context, userID string) ([]float64, error)
    GetItemVector(ctx context.Context, itemID string) ([]float64, error)
    GetAllItemVectors(ctx context.Context) (map[string][]float64, error)
}
```

**作用**: 定义矩阵分解所需的数据存储接口。

**已有实现**:
- ✅ `StoreMFAdapter` - 将 `store.Store` 适配为 `MFStore`

### 1.2 核心实现

#### `recall.Fanout` - 多路并发召回节点
**位置**: `recall/fanout.go`

**作用**: 
- 并发执行多个召回源
- 支持超时控制和限流
- 自动合并结果并记录召回来源 Label

**特性**:
- ✅ 并发执行（errgroup）
- ✅ 超时控制
- ✅ 限流保护（MaxConcurrent）
- ✅ 自定义合并策略
- ✅ 自定义错误处理策略
- ✅ 自动记录召回来源 Label

---

## 2. 过滤阶段 (Filter)

### 2.1 核心接口

#### `filter.Filter` - 过滤器接口
**位置**: `filter/filter.go`

```go
type Filter interface {
    Name() string
    ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error)
}
```

**作用**: 定义过滤器的标准接口，返回 true 表示过滤（移除），false 表示保留。

**已有实现**:
- ✅ `BlacklistFilter` - 黑名单过滤
- ✅ `UserBlockFilter` - 用户拉黑过滤
- ✅ `ExposedFilter` - 已曝光过滤

#### `filter.BlacklistStore` - 黑名单存储接口
**位置**: `filter/blacklist.go`

```go
type BlacklistStore interface {
    GetBlacklist(ctx context.Context, key string) ([]string, error)
}
```

**作用**: 定义黑名单数据的存储接口。

**已有实现**:
- ✅ `StoreAdapter` - 将 `store.Store` 适配为各种 Filter Store

#### `filter.UserBlockStore` - 用户拉黑存储接口
**位置**: `filter/user_block.go`

```go
type UserBlockStore interface {
    GetUserBlocks(ctx context.Context, userID string, keyPrefix string) ([]string, error)
}
```

**作用**: 定义用户拉黑数据的存储接口。

**已有实现**:
- ✅ `StoreAdapter` - 将 `store.Store` 适配为各种 Filter Store

#### `filter.ExposedStore` - 曝光历史存储接口
**位置**: `filter/exposed.go`

```go
type ExposedStore interface {
    GetExposedItems(ctx context.Context, userID string, keyPrefix string, timeWindow int64) ([]string, error)
}
```

**作用**: 定义曝光历史数据的存储接口。

**已有实现**:
- ✅ `StoreAdapter` - 将 `store.Store` 适配为各种 Filter Store

### 2.2 核心实现

#### `filter.FilterNode` - 组合过滤器节点
**位置**: `filter/node.go`

**作用**: 
- 组合多个过滤器
- 如果任何一个过滤器返回 true，该物品就会被过滤掉
- 自动记录过滤原因 Label

**特性**:
- ✅ 支持多个过滤器组合
- ✅ 自动记录过滤原因
- ✅ 错误容错（过滤器错误不中断流程）

---

## 3. 特征注入阶段 (Feature Enrichment)

### 3.1 核心接口

#### `feature.FeatureService` - 特征服务接口
**位置**: `feature/service.go`

```go
type FeatureService interface {
    Name() string
    GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error)
    BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error)
    GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)
    BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error)
    GetRealtimeFeatures(ctx context.Context, userID, itemID string) (map[string]float64, error)
    BatchGetRealtimeFeatures(ctx context.Context, pairs []UserItemPair) (map[UserItemPair]map[string]float64, error)
    Close() error
}
```

**作用**: 统一的特征服务接口，提供用户特征、物品特征、实时特征的获取能力。

**已有实现**:
- ✅ `BaseFeatureService` - 基础特征服务实现（支持缓存、监控、降级）

#### `feature.FeatureProvider` - 特征提供者接口
**位置**: `feature/service.go`

```go
type FeatureProvider interface {
    Name() string
    GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error)
    BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error)
    GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)
    BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error)
    GetRealtimeFeatures(ctx context.Context, userID, itemID string) (map[string]float64, error)
    BatchGetRealtimeFeatures(ctx context.Context, pairs []UserItemPair) (map[UserItemPair]map[string]float64, error)
}
```

**作用**: 定义特征提供者的抽象接口，支持多种特征源（Redis、HTTP、Memory等）。

**已有实现**:
- ✅ `StoreFeatureProvider` - 基于 Store 的特征提供者（支持 Redis）

#### `feature.FeatureStore` - 特征存储接口
**位置**: `feature/service.go`

```go
type FeatureStore interface {
    Name() string
    GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error)
    BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error)
    GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)
    BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error)
    GetRealtimeFeatures(ctx context.Context, userID, itemID string) (map[string]float64, error)
    BatchGetRealtimeFeatures(ctx context.Context, pairs []UserItemPair) (map[UserItemPair]map[string]float64, error)
    SetUserFeatures(ctx context.Context, userID string, features map[string]float64, ttl time.Duration) error
    SetItemFeatures(ctx context.Context, itemID string, features map[string]float64, ttl time.Duration) error
}
```

**作用**: 定义特征存储的抽象接口，用于特征数据的持久化和读取。

#### `feature.FeatureCache` - 特征缓存接口
**位置**: `feature/service.go`

```go
type FeatureCache interface {
    GetUserFeatures(ctx context.Context, userID string) (map[string]float64, bool)
    SetUserFeatures(ctx context.Context, userID string, features map[string]float64, ttl time.Duration)
    GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, bool)
    SetItemFeatures(ctx context.Context, itemID string, features map[string]float64, ttl time.Duration)
    InvalidateUserFeatures(ctx context.Context, userID string)
    InvalidateItemFeatures(ctx context.Context, itemID string)
    Clear(ctx context.Context)
}
```

**作用**: 定义特征缓存接口，采用装饰器模式，为特征服务添加缓存能力。

**已有实现**:
- ✅ `MemoryFeatureCache` - 内存特征缓存

#### `feature.FeatureMonitor` - 特征监控接口
**位置**: `feature/service.go`

```go
type FeatureMonitor interface {
    RecordFeatureUsage(ctx context.Context, featureName string, value float64)
    RecordFeatureMissing(ctx context.Context, featureName string, entityType string, entityID string)
    RecordFeatureError(ctx context.Context, featureName string, err error)
    GetFeatureStats(ctx context.Context, featureName string) (*FeatureStats, error)
}
```

**作用**: 定义特征监控接口，用于监控特征质量、分布、缺失率等。

**已有实现**:
- ✅ `BaseFeatureMonitor` - 基础特征监控实现

#### `feature.FallbackStrategy` - 降级策略接口
**位置**: `feature/service.go`

```go
type FallbackStrategy interface {
    GetUserFeatures(ctx context.Context, userID string, rctx *core.RecommendContext) (map[string]float64, error)
    GetItemFeatures(ctx context.Context, itemID string, item *core.Item) (map[string]float64, error)
    GetRealtimeFeatures(ctx context.Context, userID, itemID string, rctx *core.RecommendContext, item *core.Item) (map[string]float64, error)
}
```

**作用**: 定义降级策略接口，当特征服务不可用时提供降级方案。

**已有实现**:
- ✅ `DefaultFallbackStrategy` - 默认降级策略

#### `feast.Client` - Feast Feature Store 客户端接口
**位置**: `feast/client.go`

```go
type Client interface {
    GetOnlineFeatures(ctx context.Context, req *GetOnlineFeaturesRequest) (*GetOnlineFeaturesResponse, error)
    GetHistoricalFeatures(ctx context.Context, req *GetHistoricalFeaturesRequest) (*GetHistoricalFeaturesResponse, error)
    Materialize(ctx context.Context, req *MaterializeRequest) error
    ListFeatures(ctx context.Context) ([]Feature, error)
    GetFeatureService(ctx context.Context) (*FeatureServiceInfo, error)
    Close() error
}
```

**作用**: 定义 Feast Feature Store 的客户端接口，支持在线特征、历史特征、特征物化等功能。

**已有实现**:
- ✅ `HTTPClient` - HTTP 客户端实现
- ✅ `GrpcClient` - gRPC 客户端实现
- ✅ `FeatureServiceAdapter` - 将 Feast Client 适配为 `feature.FeatureService`

### 3.2 核心实现

#### `feature.EnrichNode` - 特征注入节点
**位置**: `feature/enrich.go`

**作用**: 
- 将用户特征、物品特征、交叉特征组合
- 支持批量特征获取
- 支持特征前缀（user_、item_、cross_）

**特性**:
- ✅ 支持 FeatureService 模式（推荐）
- ✅ 支持传统提取器模式
- ✅ 自动生成交叉特征
- ✅ 批量获取物品特征

---

## 4. 排序阶段 (Rank)

### 4.1 核心接口

#### `model.RankModel` - 排序模型接口
**位置**: `model/model.go`

```go
type RankModel interface {
    Name() string
    Predict(features map[string]float64) (float64, error)
}
```

**作用**: 定义排序模型的标准接口，输入特征，输出可比较的分数。

**已有实现**:
- ✅ `LRModel` - 线性回归模型
- ✅ `RPCModel` - RPC 模型（支持批量预测）
- ✅ `DNNModel` - 深度神经网络模型
- ✅ `DINModel` - Deep Interest Network 模型
- ✅ `WideDeepModel` - Wide&Deep 模型
- ✅ `TwoTowerModel` - 双塔模型

#### `rank.SortStrategy` - 排序策略接口
**位置**: `rank/lr_node.go`

```go
type SortStrategy interface {
    Sort(items []*core.Item)
}
```

**作用**: 定义物品排序逻辑，支持多种排序策略。

**已有实现**:
- ✅ `ScoreDescSortStrategy` - 按分数降序排序（默认）
- ✅ `ScoreAscSortStrategy` - 按分数升序排序
- ✅ `MultiFieldSortStrategy` - 多字段排序策略

### 4.2 核心实现

#### `rank.LRNode` - LR 排序节点
**位置**: `rank/lr_node.go`

**作用**: 
- 使用 RankModel 进行预测
- 更新 item.Score
- 按分数排序
- 记录 rank_model Label

**特性**:
- ✅ 支持自定义排序策略
- ✅ 自动记录排序模型 Label

#### `rank.RPCNode` - RPC 排序节点
**位置**: `rank/rpc_node.go`

**作用**: 
- 通过 RPC 调用外部模型服务
- 支持批量预测
- 支持 TensorFlow Serving、TorchServe 等

**特性**:
- ✅ 支持批量预测（PredictBatch）
- ✅ 特征名标准化（去掉前缀）
- ✅ 自动排序

#### `rank.DNNNode` - DNN 排序节点
**位置**: `rank/dnn_node.go`

**作用**: 使用 DNN 模型进行排序。

#### `rank.DINNode` - DIN 排序节点
**位置**: `rank/din_node.go`

**作用**: 
- 使用 DIN 模型进行排序
- 支持用户行为序列（从 UserProfile.RecentClicks 获取）

**特性**:
- ✅ 支持最大行为序列长度配置

#### `rank.WideDeepNode` - Wide&Deep 排序节点
**位置**: `rank/wide_deep_node.go`

**作用**: 使用 Wide&Deep 模型进行排序。

#### `rank.TwoTowerNode` - 双塔排序节点
**位置**: `rank/two_tower_node.go`

**作用**: 使用两塔模型进行排序，分别学习用户表示和物品表示。

---

## 5. Top-N 截断阶段

### 5.1 核心实现

#### `rerank.TopNNode` - Top-N 截断节点
**位置**: `rerank/topn.go`

**作用**: 
- 在排序后截取前 N 个物品
- 控制推荐结果数量
- 提升性能

**特性**:
- ✅ 支持边界情况处理（N <= 0 或 N > len(items)）
- ✅ 简单高效

---

## 6. 多样性重排阶段 (ReRank)

### 6.1 核心实现

#### `rerank.Diversity` - 多样性重排节点
**位置**: `rerank/diversity.go`

**作用**: 
- 按类别去重（保留首个出现的类别）
- 提升推荐结果的多样性

**特性**:
- ✅ 支持从 Label 或 Meta 读取类别
- ✅ 自动去重

---

## 7. Pipeline 基础设施

### 7.1 核心接口

#### `pipeline.PipelineHook` - Pipeline Hook 接口
**位置**: `pipeline/pipeline.go`

```go
type PipelineHook interface {
    BeforeNode(ctx context.Context, rctx *core.RecommendContext, node Node, items []*core.Item) ([]*core.Item, error)
    AfterNode(ctx context.Context, rctx *core.RecommendContext, node Node, items []*core.Item, err error) ([]*core.Item, error)
}
```

**作用**: 定义 Pipeline 执行过程中的 Hook 接口，用于实现中间件功能（日志、监控、缓存等）。

#### `pipeline.NodeBuilder` - Node 构建器函数类型
**位置**: `pipeline/config.go`

```go
type NodeBuilder func(map[string]interface{}) (Node, error)
```

**作用**: 定义 Node 构建器函数类型，用于从配置构建 Node 实例。

### 7.2 核心实现

#### `pipeline.Pipeline` - Pipeline 执行器
**位置**: `pipeline/pipeline.go`

**作用**: 
- 链式执行多个 Node
- 支持 Hook 机制
- 支持错误处理

**特性**:
- ✅ 支持 Hook 机制（BeforeNode、AfterNode）
- ✅ 链式处理
- ✅ 错误传播

#### `pipeline.NodeFactory` - Node 工厂
**位置**: `pipeline/config.go`

**作用**: 
- 根据配置构建 Node 实例
- 支持动态注册自定义 Node 类型
- 线程安全

**特性**:
- ✅ 线程安全的动态注册
- ✅ 支持从 YAML/JSON 配置加载
- ✅ 支持自定义 Node 类型注册

**已有实现**:
- ✅ `config.DefaultFactory` - 默认工厂，包含所有内置 Node 的构建器

---

## 8. 存储抽象

### 8.1 核心接口

#### `store.Store` - 存储接口
**位置**: `store/store.go`

```go
type Store interface {
    Name() string
    Get(ctx context.Context, key string) ([]byte, error)
    Set(ctx context.Context, key string, value []byte, ttl ...int) error
    Delete(ctx context.Context, key string) error
    BatchGet(ctx context.Context, keys []string) (map[string][]byte, error)
    BatchSet(ctx context.Context, kvs map[string][]byte, ttl ...int) error
    Close() error
}
```

**作用**: 定义存储的抽象接口，支持多种存储后端。

**已有实现**:
- ✅ `MemoryStore` - 内存存储
- ✅ `RedisStore` - Redis 存储

#### `store.KeyValueStore` - 键值存储接口
**位置**: `store/store.go`

```go
type KeyValueStore interface {
    Store
    ZAdd(ctx context.Context, key string, score float64, member string) error
    ZRange(ctx context.Context, key string, start, stop int64) ([]string, error)
    ZScore(ctx context.Context, key string, member string) (float64, error)
    HGet(ctx context.Context, key, field string) ([]byte, error)
    HSet(ctx context.Context, key, field string, value []byte) error
    HGetAll(ctx context.Context, key string) (map[string][]byte, error)
}
```

**作用**: 扩展 Store 接口，支持有序集合和哈希表操作。

**已有实现**:
- ✅ `MemoryStore` - 内存存储（实现 KeyValueStore）
- ✅ `RedisStore` - Redis 存储（实现 KeyValueStore）

---

## 9. 向量服务

### 9.1 核心接口

#### `vector.ANNService` - 向量检索服务接口
**位置**: `vector/ann_service.go`

```go
type ANNService interface {
    Search(ctx context.Context, vector []float64, topK int, metric string) ([]string, []float64, error)
}
```

**作用**: 定义向量检索服务的接口。

**已有实现**:
- ✅ `MilvusClient` - Milvus 向量数据库客户端
- ✅ `ANNServiceClient` - 外部 ANN 服务客户端

---

## 10. ML 服务

### 10.1 核心接口

#### `service.MLService` - 机器学习服务接口
**位置**: `service/ml_service.go`

```go
type MLService interface {
    Predict(ctx context.Context, req *PredictRequest) (*PredictResponse, error)
    Health(ctx context.Context) error
    Close() error
}
```

**作用**: 统一的机器学习服务接口，用于对接 TF Serving、TorchServe、自定义模型服务等。

**已有实现**:
- ✅ `TFServingClient` - TensorFlow Serving 客户端（REST API）
- ✅ `TorchServeClient` - TorchServe 客户端（REST API）
- ✅ `ANNServiceClient` - ANN 服务客户端

---

## 11. 配置接口

### 11.1 核心接口

#### `core.RecallConfig` - 召回配置接口
**位置**: `core/config.go`

```go
type RecallConfig interface {
    DefaultTopKSimilarUsers() int
    DefaultTopKItems() int
    DefaultMinCommonItems() int
    DefaultMinCommonUsers() int
    DefaultTimeout() time.Duration
}
```

**作用**: 定义召回算法的默认配置接口。

**已有实现**:
- ✅ `DefaultRecallConfig` - 默认召回配置

---

## 12. 工具接口

### 12.1 核心接口

#### `utils.LabelMergeStrategy` - Label 合并策略接口
**位置**: `pkg/utils/label.go`

```go
type LabelMergeStrategy interface {
    Merge(existing, incoming Label) Label
}
```

**作用**: 定义 Label 合并策略，用于自定义 Label 合并逻辑。

**已有实现**:
- ✅ `DefaultLabelMergeStrategy` - 默认合并策略（Value 以 '|' 累积，Source 以 ',' 累积）
- ✅ `PriorityLabelMergeStrategy` - 优先级合并策略（按优先级覆盖）
- ✅ `AccumulateLabelMergeStrategy` - 累加策略（适用于数值型 Value）

### 12.2 DSL 表达式引擎

#### `dsl.Eval` - Label DSL 解释器
**位置**: `pkg/dsl/eval.go`

**作用**: 
- 使用 CEL (Common Expression Language) 实现 Label DSL 表达式引擎
- 支持基于 Label、Item、Context 的条件判断
- 用于策略驱动和可解释性

**支持的语法**:
- 基础比较：`label.recall_source == "hot"` / `item.score > 0.7`
- 逻辑运算：`label.category == "A" && item.score > 0.8`
- 存在性检查：`label.recall_source != null`
- 包含判断：`label.recall_source.contains("hot")` 或 `"hot" in label.recall_source`

**使用示例**:
```go
eval := dsl.NewEval(item, rctx)
result, _ := eval.Evaluate(`label.recall_source == "hot" && item.score > 0.7`)
```

---

## 总结

### 接口统计

| 模块 | 接口数量 | 实现数量 |
|------|---------|---------|
| Pipeline | 3 | 2 |
| Recall | 8 | 10+ |
| Filter | 4 | 4 |
| Feature | 6 | 5+ |
| Feast | 1 | 3 |
| Rank | 2 | 6 |
| ReRank | 0 | 2 |
| Store | 2 | 2 |
| Vector | 1 | 2 |
| Service | 1 | 3 |
| Config | 1 | 1 |
| Utils | 1 | 2 |

### 核心特性

1. **高度可扩展**: 所有策略都通过接口实现，支持自定义扩展
2. **批量处理**: 支持批量特征获取、批量模型预测
3. **错误处理**: 完善的错误处理和降级策略
4. **监控支持**: 支持特征监控、Pipeline Hook
5. **缓存支持**: 支持特征缓存，提升性能
6. **Labels-first**: 全链路标签追踪，支持可解释性

### 推荐流程完整支持

✅ **召回**: 支持 7+ 种召回算法，多路并发召回  
✅ **过滤**: 支持黑名单、用户拉黑、已曝光过滤  
✅ **特征注入**: 支持批量特征获取、交叉特征生成  
✅ **排序**: 支持 6+ 种排序模型，多种排序策略  
✅ **Top-N**: 支持 Top-N 截断  
✅ **多样性重排**: 支持类别去重、多样性提升  
