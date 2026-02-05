# Reckit 可扩展性指南

本文档介绍如何通过接口、Hook、透传等方式扩展 Reckit 功能，而无需修改库代码。

**设计原则**：所有策略、算法、配置都通过接口抽象，不提供字符串匹配的硬编码方式。

---

## 扩展方式

### 1. 实现接口

所有核心功能都通过接口定义，实现接口即可扩展：

- `recall.Source`：召回源接口
- `filter.Filter`：过滤器接口
- `model.RankModel`：排序模型接口
- `feature.FeatureService`：特征服务接口
- `core.Store`：存储接口

详细接口定义请参考：[接口与实现完整分析](./INTERFACES_AND_IMPLEMENTATIONS.md)

### 2. 自定义策略

所有策略都通过接口实现，支持自定义：

- `MergeStrategy`：合并策略接口
- `ErrorHandler`：错误处理接口
- `SortStrategy`：排序策略接口
- `SimilarityCalculator`：相似度计算接口
- `LabelMergeStrategy`：Label 合并策略接口

### 3. 工厂模式

支持动态注册自定义 Node：

```go
factory := pipeline.NewNodeFactory()

// 注册自定义 Node
factory.Register("my.custom.node", func(config map[string]interface{}) (pipeline.Node, error) {
    return &MyCustomNode{}, nil
})

// 构建 Node
node, err := factory.Build("my.custom.node", config)
```

### 4. Pipeline Hook

Pipeline 支持 Hook，可以在执行前后插入自定义逻辑：

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

### 5. 配置接口

所有默认值都通过配置接口提供：

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

---

## 扩展示例

### 添加新的召回算法

```go
type MyRecall struct{}

func (r *MyRecall) Name() string { return "my_recall" }
func (r *MyRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
    // 实现召回逻辑
    items := []*core.Item{}
    // ...
    return items, nil
}
```

### 添加新的过滤器

```go
type MyFilter struct{}

func (f *MyFilter) Name() string { return "filter.my" }
func (f *MyFilter) ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error) {
    // 返回 true 表示过滤，false 表示保留
    return false, nil
}
```

### 添加新的排序模型

```go
type MyModel struct{}

func (m *MyModel) Name() string { return "my_model" }
func (m *MyModel) Predict(features map[string]float64) (float64, error) {
    // 实现预测逻辑
    return 0.5, nil
}

// 创建对应的 Rank Node
type MyRankNode struct {
    Model model.RankModel
}

func (n *MyRankNode) Name() string { return "rank.my" }
func (n *MyRankNode) Kind() pipeline.Kind { return pipeline.KindRank }
func (n *MyRankNode) Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
    for _, item := range items {
        score, _ := n.Model.Predict(item.Features)
        item.Score = score
    }
    return items, nil
}
```

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

---

## 相关文档

- [架构设计](./ARCHITECTURE.md) - 架构设计说明
- [Feature 模块](./FEATURE_MODULE.md) - 特征模块职责
- [召回算法](./RECALL_ALGORITHMS.md) - 召回算法使用
