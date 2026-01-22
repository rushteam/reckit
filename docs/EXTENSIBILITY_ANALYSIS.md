# Reckit 可扩展性分析报告

## 概述

本文档分析 Reckit 作为工业级推荐系统工具库的可扩展性，识别需要改进的地方，确保用户可以通过接口、Hook、透传等方式扩展功能，而无需修改库代码。

**设计原则**：所有策略、算法、配置都通过接口抽象，不提供字符串匹配的硬编码方式。

## 当前可扩展性状态

### ✅ 已实现的可扩展功能

Reckit 已经实现了完整的可扩展性设计：

1. **接口抽象**：核心功能都通过接口定义
   - `recall.Source`：召回源接口
   - `filter.Filter`：过滤器接口
   - `model.RankModel`：排序模型接口
   - `feature.FeatureService`：特征服务接口
   - `store.Store`：存储接口

2. **策略模式**：所有策略都通过接口实现
   - `MergeStrategy`：合并策略接口
   - `ErrorHandler`：错误处理接口
   - `SortStrategy`：排序策略接口
   - `SimilarityCalculator`：相似度计算接口
   - `LabelMergeStrategy`：Label 合并策略接口

3. **工厂模式**：支持动态注册
   - `NodeFactory` 支持线程安全的动态注册
   - 用户可以在运行时注册自定义 Node 类型

4. **Hook 机制**：Pipeline 支持 Hook
   - `PipelineHook` 接口支持在执行前后插入逻辑
   - 可以用于日志、监控、缓存等场景

5. **配置接口**：所有默认值都通过配置接口提供
   - `RecallConfig`：召回配置接口
   - `FeatureConfig`：特征配置接口

## 使用示例

### 自定义合并策略

```go
type CustomMergeStrategy struct{}

func (s *CustomMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
    // 自定义合并逻辑
    // 例如：只保留分数最高的物品
    if !dedup {
        return items
    }
    
    bestItems := make(map[string]*core.Item)
    for _, item := range items {
        if item == nil {
            continue
        }
        if existing, ok := bestItems[item.ID]; ok {
            if item.Score > existing.Score {
                bestItems[item.ID] = item
            }
        } else {
            bestItems[item.ID] = item
        }
    }
    
    result := make([]*core.Item, 0, len(bestItems))
    for _, item := range bestItems {
        result = append(result, item)
    }
    return result
}

// 使用
fanout := &recall.Fanout{
    Sources: []recall.Source{...},
    Dedup: true,
    MergeStrategy: &CustomMergeStrategy{},
}
```

### 自定义相似度计算器

```go
type JaccardSimilarity struct{}

func (j *JaccardSimilarity) Calculate(x, y []float64) float64 {
    // 实现 Jaccard 相似度
    // ...
}

// 使用
config := &core.DefaultRecallConfig{}
u2i := &recall.U2IRecall{
    Store:                cfStore,
    SimilarityCalculator: &JaccardSimilarity{},
    Config:                config,
}
```

### 动态注册自定义 Node

```go
factory := pipeline.NewNodeFactory()

// 注册自定义 Node
factory.Register("my.custom.node", func(config map[string]interface{}) (pipeline.Node, error) {
    return &MyCustomNode{}, nil
})

// 构建 Node
node, err := factory.Build("my.custom.node", config)
```

### Pipeline Hook

```go
type LoggingHook struct{}

func (h *LoggingHook) BeforeNode(ctx context.Context, rctx *core.RecommendContext, node pipeline.Node, items []*core.Item) ([]*core.Item, error) {
    fmt.Printf("[Hook] Before %s: %d items\n", node.Name(), len(items))
    return items, nil
}

func (h *LoggingHook) AfterNode(ctx context.Context, rctx *core.RecommendContext, node pipeline.Node, items []*core.Item, err error) ([]*core.Item, error) {
    if err != nil {
        fmt.Printf("[Hook] After %s: error=%v\n", node.Name(), err)
    } else {
        fmt.Printf("[Hook] After %s: %d items\n", node.Name(), len(items))
    }
    return items, err
}

// 使用
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{...},
    Hooks: []pipeline.PipelineHook{
        &LoggingHook{},
    },
}
```

### 自定义配置

```go
type MyRecallConfig struct {
    core.DefaultRecallConfig
}

func (c *MyRecallConfig) DefaultTopKItems() int {
    return 50 // 自定义默认值
}

// 使用
config := &MyRecallConfig{}
u2i := &recall.U2IRecall{
    Config: config,
}
```

## 扩展指南

### 添加新的召回算法

1. 实现 `recall.Source` 接口：
```go
type MyRecall struct{}

func (r *MyRecall) Name() string { return "my_recall" }
func (r *MyRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
    // 实现召回逻辑
}
```

2. 在工厂中注册（可选）：
```go
factory.Register("recall.my", buildMyRecallNode)
```

### 添加新的过滤器

1. 实现 `filter.Filter` 接口：
```go
type MyFilter struct{}

func (f *MyFilter) Name() string { return "filter.my" }
func (f *MyFilter) ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error) {
    // 返回 true 表示过滤，false 表示保留
}
```

### 添加新的排序模型

1. 实现 `model.RankModel` 接口：
```go
type MyModel struct{}

func (m *MyModel) Name() string { return "my_model" }
func (m *MyModel) Predict(features map[string]float64) (float64, error) {
    // 实现预测逻辑
}
```

2. 创建对应的 Rank Node：
```go
type MyRankNode struct {
    Model model.RankModel
}

func (n *MyRankNode) Name() string { return "rank.my" }
func (n *MyRankNode) Kind() pipeline.Kind { return pipeline.KindRank }
func (n *MyRankNode) Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
    // 实现排序逻辑
}
```

## 总结

Reckit 作为推荐系统工具库，已经实现了完整的可扩展性设计：

- ✅ 所有核心功能都通过接口抽象
- ✅ 所有策略都通过接口实现，支持自定义
- ✅ 工厂模式支持动态注册
- ✅ Pipeline 支持 Hook 机制
- ✅ 所有默认值都通过配置接口提供
- ✅ 无硬编码，无字符串匹配

用户可以通过实现接口、注册工厂、添加 Hook 等方式扩展所有功能，无需修改库代码。
