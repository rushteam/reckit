# 特征抽取器使用指南

> **注意**：本文档已整合到 [docs/FEATURE_MODULE.md](../docs/FEATURE_MODULE.md)。
> 
> 请参考 [Feature 模块指南](../docs/FEATURE_MODULE.md) 了解 Extractor、Service、EnrichNode 的完整职责和使用方法。

## 快速参考

### DefaultFeatureExtractor

从 `RecommendContext` 中提取特征。

```go
// 默认用法（从 Params 提取，无前缀）
extractor := feature.NewDefaultFeatureExtractor()

// 自定义前缀
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithParamsPrefix("ctx_"),
)

// 只提取指定的 key
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithParamsKeys([]string{"latitude", "longitude", "time_of_day"}),
)

// 使用 FeatureService
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithFeatureService(featureService),
)
```

### 在双塔召回中使用

```go
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithParamsKeys([]string{"latitude", "longitude"}),
)

twoTowerRecall := recall.NewTwoTowerRecall(
    nil,
    userTowerService,
    vectorService,
    recall.WithTwoTowerUserFeatureExtractor(extractor),
)
```

## 详细文档

- [Feature 模块指南](../docs/FEATURE_MODULE.md) - 完整的模块职责和使用说明
- [特征处理](../docs/FEATURE_PROCESSING.md) - 特征归一化、编码等
