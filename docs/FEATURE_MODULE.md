# Feature 模块指南

本文档介绍 Reckit 中特征相关模块的职责分工和协作关系。

## 模块概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Pipeline 执行流程                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Request                                                           │
│      │                                                              │
│      ▼                                                              │
│   ┌──────────────────┐                                             │
│   │  RecommendContext │  ← Params (请求参数)                        │
│   │  (UserID, Params) │                                             │
│   └────────┬─────────┘                                             │
│            │                                                        │
│            ▼                                                        │
│   ┌──────────────────┐     ┌──────────────────┐                    │
│   │  TwoTowerRecall  │ ──► │ FeatureExtractor │ ← 从 Context 提取   │
│   │  (召回源)         │     │ (特征抽取器)       │   特征给用户塔     │
│   └────────┬─────────┘     └──────────────────┘                    │
│            │                                                        │
│            ▼                                                        │
│   ┌──────────────────┐     ┌──────────────────┐                    │
│   │   EnrichNode     │ ──► │ FeatureService   │ ← 从存储获取        │
│   │  (特征注入节点)    │     │ (特征服务)        │   用户/物品特征    │
│   └────────┬─────────┘     └──────────────────┘                    │
│            │                                                        │
│            ▼                                                        │
│      Item.Features                                                  │
│      (完整特征)                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. FeatureExtractor（特征抽取器）

**职责**：从 `RecommendContext` 中提取特征，轻量级、无 IO 操作。

**数据来源**：
- `rctx.User`（强类型用户画像）
- `rctx.UserProfile`（map 形式用户画像）
- `rctx.Params`（请求参数）

**使用场景**：召回阶段，需要快速从请求上下文提取特征。

**代码位置**：`feature/extractor.go`

```go
// 创建默认抽取器（从 Params 提取，无前缀）
extractor := feature.NewDefaultFeatureExtractor()

// 自定义前缀
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithParamsPrefix("ctx_"),
)

// 只提取指定的 key
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithParamsKeys([]string{"latitude", "longitude", "time_of_day"}),
)

// 使用
features, err := extractor.Extract(ctx, rctx)
```

**内置实现**：
| 实现 | 说明 |
|------|------|
| `DefaultFeatureExtractor` | 从 User/UserProfile/Params 提取 |
| `CustomFeatureExtractor` | 完全自定义抽取逻辑 |
| `CompositeFeatureExtractor` | 组合多个抽取器 |
| `QueryFeatureExtractor` | Query 特征（用于 DSSM） |
| `HistoryExtractor` | 历史行为序列（用于 YouTube DNN） |

### 2. FeatureService（特征服务）

**职责**：从外部存储（Redis/Feast/HTTP）获取预计算特征，重量级、有 IO 操作。

**数据来源**：
- Redis/Memcached
- Feast Feature Store
- HTTP API

**使用场景**：需要获取持久化的用户/物品特征。

**代码位置**：
- 接口定义：`core/feature_service.go`
- 实现：`feature/base_service.go`

```go
// 从 Store 创建特征服务
factory := feature.NewFeatureServiceFactory()
service := factory.CreateFromStore(redisStore)

// 带缓存
service := factory.CreateWithCache(redisStore, 1000, 5*time.Minute)

// 使用
userFeatures, err := service.GetUserFeatures(ctx, userID)
itemFeatures, err := service.BatchGetItemFeatures(ctx, itemIDs)
```

**接口定义**：

```go
type FeatureService interface {
    Name() string
    GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error)
    BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error)
    GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)
    BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error)
    Close(ctx context.Context) error
}
```

### 3. EnrichNode（特征注入节点）

**职责**：Pipeline Node，将用户特征、物品特征、交叉特征注入到 Item 中。

**数据来源**：
- `FeatureService`（外部存储）
- `RecommendContext`（请求上下文）

**使用场景**：排序前的特征准备，为排序模型提供完整特征。

**代码位置**：`feature/enrich.go`

```go
// 使用 FeatureService
enrichNode := &feature.EnrichNode{
    FeatureService: featureService,
}

// 自定义特征提取器
enrichNode := &feature.EnrichNode{
    UserFeatureExtractor: func(rctx *core.RecommendContext) map[string]float64 {
        return map[string]float64{"age": float64(rctx.User.Age)}
    },
    ItemFeatureExtractor: func(item *core.Item) map[string]float64 {
        return item.Features
    },
}

// 在 Pipeline 中使用
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        recallNode,
        enrichNode,  // 注入特征
        rankNode,    // 使用 Item.Features 排序
    },
}
```

**特征前缀**：
- `user_*`：用户特征
- `item_*`：物品特征
- `scene_*`：场景特征
- `cross_*`：交叉特征

## 职责对比

| 模块 | 职责 | IO | 数据来源 | 使用阶段 |
|------|------|-----|---------|---------|
| **FeatureExtractor** | 从请求上下文提取特征 | 无 | `RecommendContext` | 召回 |
| **FeatureService** | 从外部存储获取特征 | 有 | Redis/Feast/HTTP | 任意 |
| **EnrichNode** | 将特征注入到 Item | 有 | Service + Context | 排序前 |

## 典型使用场景

### 场景 1：双塔召回

召回阶段需要快速从请求参数提取用户特征，传给用户塔模型。

```go
// 使用 Extractor（轻量、无 IO）
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithParamsKeys([]string{"latitude", "longitude", "time_of_day"}),
)

twoTowerRecall := recall.NewTwoTowerRecall(
    nil,  // 不需要 FeatureService
    userTowerService,
    vectorService,
    recall.WithTwoTowerUserFeatureExtractor(extractor),
)
```

### 场景 2：精排阶段

排序阶段需要完整的用户特征 + 物品特征 + 交叉特征。

```go
// 使用 EnrichNode + FeatureService
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{Sources: sources},
        &filter.FilterNode{Filters: filters},
        &feature.EnrichNode{
            FeatureService: featureService,  // 从存储获取
        },
        &rank.DNNNode{Model: dnnModel},  // 使用 Item.Features
    },
}
```

### 场景 3：组合使用

同时使用 FeatureService（获取持久化特征）和 Params（获取请求上下文）。

```go
// 组合多个 Extractor
baseExtractor := feature.NewDefaultFeatureExtractor(
    feature.WithFeatureService(featureService),  // 持久化特征
    feature.WithIncludeParams(false),
)

paramsExtractor := feature.NewDefaultFeatureExtractor(
    feature.WithParamsPrefix("ctx_"),  // 请求上下文
)

compositeExtractor := feature.NewCompositeFeatureExtractor(
    "combined",
    baseExtractor,
    paramsExtractor,
)
```

## 相关文档

- [特征处理](./FEATURE_PROCESSING.md) - 特征归一化、编码等
- [双塔指南](./TWO_TOWER_GUIDE.md) - 双塔模型使用
- [架构设计](./ARCHITECTURE.md) - 整体架构
