# 特征抽取器使用指南

## 概述

作为推荐脚手架，不同模型可能需要不同的特征抽取逻辑。Reckit 提供了统一的 `FeatureExtractor` 接口，支持灵活自定义特征抽取策略。

## 核心接口

```go
type FeatureExtractor interface {
    Extract(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error)
    Name() string
}
```

## 内置实现

### 1. DefaultFeatureExtractor（默认抽取器）

从 `RecommendContext` 中提取特征，支持：
- 从 `UserProfile`（强类型）提取：`age`, `gender`, `interest_<tag>`
- 从 `UserProfileMap` 提取：所有可转换为 `float64` 的值
- 从 `Realtime` 提取：所有可转换为 `float64` 的值（添加 `realtime_` 前缀）

**使用示例**：

```go
import "github.com/rushteam/reckit/feature"

// 基础用法（无前缀）
extractor := feature.NewDefaultFeatureExtractor()

// 带前缀（如 "user_"）
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithFieldPrefix("user_"),
)

// 优先使用 FeatureService
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithFeatureService(featureService),
    feature.WithFieldPrefix("user_"),
)

// 不包含实时特征
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithIncludeRealtime(false),
)
```

### 2. CustomFeatureExtractor（自定义抽取器）

完全自定义抽取逻辑：

```go
customExtractor := feature.NewCustomFeatureExtractor(
    "my_custom",
    func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
        features := make(map[string]float64)
        // 自定义逻辑：从多个源组合特征
        if rctx.User != nil {
            features["age"] = float64(rctx.User.Age)
            // ... 其他逻辑
        }
        // 从外部服务获取特征
        externalFeatures, _ := externalService.GetFeatures(ctx, rctx.UserID)
        for k, v := range externalFeatures {
            features["external_"+k] = v
        }
        return features, nil
    },
)
```

### 3. CompositeFeatureExtractor（组合抽取器）

从多个抽取器组合特征：

```go
// 从 FeatureService 获取基础特征
baseExtractor := feature.NewDefaultFeatureExtractor(
    feature.WithFeatureService(featureService),
)

// 从 Context 获取实时特征
realtimeExtractor := feature.NewDefaultFeatureExtractor(
    feature.WithIncludeRealtime(true),
)

// 组合
compositeExtractor := feature.NewCompositeFeatureExtractor(
    "composite",
    baseExtractor,
    realtimeExtractor,
)

// 自定义合并策略（默认：后覆盖前）
compositeExtractor.WithMergeFunc(func(featuresList []map[string]float64) map[string]float64 {
    result := make(map[string]float64)
    for _, features := range featuresList {
        for k, v := range features {
            // 自定义合并逻辑：加权平均
            if old, ok := result[k]; ok {
                result[k] = (old + v) / 2.0
            } else {
                result[k] = v
            }
        }
    }
    return result
})
```

### 4. QueryFeatureExtractor（Query 特征抽取器）

用于 DSSM 等 Query-Doc 匹配场景：

```go
// 基础用法：从 Params["query_features"] 获取
queryExtractor := feature.NewQueryFeatureExtractor()

// 自定义 key
queryExtractor := feature.NewQueryFeatureExtractor(
    feature.WithQueryFeaturesKey("my_query_features"),
)

// 从 query 文本构建特征
queryExtractor := feature.NewQueryFeatureExtractor(
    feature.WithQueryTextKey("query"),
    feature.WithTextFeatureBuilder(func(queryText string) map[string]float64 {
        // 文本特征化：词频、TF-IDF 等
        features := make(map[string]float64)
        words := strings.Fields(queryText)
        for _, word := range words {
            features["word_"+word] = 1.0
        }
        return features
    }),
)
```

### 5. HistoryExtractor（历史行为抽取器）

用于 YouTube DNN 等需要历史序列的场景：

```go
// 基础用法：从 User.RecentClicks 获取
historyExtractor := feature.NewHistoryExtractor()

// 自定义 key 和最大长度
historyExtractor := feature.NewHistoryExtractor(
    feature.WithHistoryKey("recent_views"),
    feature.WithMaxLength(100),
)

// 完全自定义
historyExtractor := feature.NewHistoryExtractor(
    feature.WithCustomHistoryExtractor(func(rctx *core.RecommendContext) []string {
        // 自定义逻辑：从多个源组合历史
        var history []string
        if rctx.User != nil {
            history = append(history, rctx.User.RecentClicks...)
        }
        if rctx.Params != nil {
            if views, ok := rctx.Params["recent_views"].([]string); ok {
                history = append(history, views...)
            }
        }
        return history
    }),
)
```

## 在召回源中使用

### TwoTowerRecall

```go
import "github.com/rushteam/reckit/recall"
import "github.com/rushteam/reckit/feature"

// 方式 1：使用默认抽取器（带前缀）
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithFeatureService(featureService),
    feature.WithFieldPrefix("user_"),
)
twoTowerRecall := recall.NewTwoTowerRecall(
    featureService,
    userTowerService,
    vectorService,
    recall.WithTwoTowerUserFeatureExtractor(extractor),
)

// 方式 2：使用自定义抽取器
customExtractor := feature.NewCustomFeatureExtractor(
    "my_extractor",
    func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
        // 自定义逻辑
        return features, nil
    },
)
twoTowerRecall := recall.NewTwoTowerRecall(
    featureService,
    userTowerService,
    vectorService,
    recall.WithTwoTowerUserFeatureExtractor(customExtractor),
)

// 方式 3：向后兼容（函数类型自动适配）
twoTowerRecall := recall.NewTwoTowerRecall(
    featureService,
    userTowerService,
    vectorService,
    recall.WithTwoTowerUserFeatureExtractor(func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
        // 函数类型会自动包装为 CustomFeatureExtractor
        return features, nil
    }),
)
```

### YouTubeDNNRecall

```go
// 用户特征抽取器
userExtractor := feature.NewDefaultFeatureExtractor(
    feature.WithFeatureService(featureService),
    feature.WithFieldPrefix("user_"),
)

// 历史行为抽取器
historyExtractor := feature.NewHistoryExtractor(
    feature.WithMaxLength(50),
)

youtubeDNNRecall := &recall.YouTubeDNNRecall{
    FeatureService:      featureService,
    UserEmbeddingURL:    "http://localhost:8082/user_embedding",
    UserFeatureExtractor: userExtractor,
    HistoryExtractor:     historyExtractor,
    VectorService:        vectorService,
    TopK:                 100,
    Collection:           "youtube_dnn_items",
}
```

### DSSMRecall

```go
// Query 特征抽取器
queryExtractor := feature.NewQueryFeatureExtractor(
    feature.WithQueryFeaturesKey("query_features"),
    feature.WithTextFeatureBuilder(func(queryText string) map[string]float64 {
        // 文本特征化
        return features
    }),
)

dssmRecall := &recall.DSSMRecall{
    QueryEmbeddingURL:    "http://localhost:8083/query_embedding",
    QueryFeatureExtractor: queryExtractor,
    VectorService:         vectorService,
    TopK:                  100,
    Collection:            "dssm_docs",
}
```

## 向后兼容

为了保持向后兼容，`WithTwoTowerUserFeatureExtractor` 等函数支持传入函数类型，会自动适配为 `CustomFeatureExtractor`：

```go
// 旧代码（仍然有效）
recall.WithTwoTowerUserFeatureExtractor(func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
    // ...
})

// 新代码（推荐）
extractor := feature.NewDefaultFeatureExtractor(...)
recall.WithTwoTowerUserFeatureExtractor(extractor)
```

## 最佳实践

1. **优先使用 FeatureService**：如果已有 `FeatureService`，优先使用 `DefaultFeatureExtractor` 配合 `FeatureService`。
2. **字段命名一致性**：使用 `WithFieldPrefix` 保持字段命名与训练时一致（如 `user_age`, `item_ctr`）。
3. **组合多个源**：使用 `CompositeFeatureExtractor` 从多个源组合特征。
4. **自定义逻辑**：对于复杂场景，使用 `CustomFeatureExtractor` 完全自定义。
5. **向后兼容**：函数类型会自动适配，无需立即修改现有代码。

## 相关文档

- [特征服务文档](./README.md)
- [双塔召回指南](../docs/TWO_TOWER_GUIDE.md)
- [架构设计文档](../docs/ARCHITECTURE.md)
