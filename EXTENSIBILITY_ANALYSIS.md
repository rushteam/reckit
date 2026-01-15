# Reckit 可扩展性分析报告

## 概述

本文档分析 Reckit 作为工业级推荐系统工具库的可扩展性，识别需要修改包本身才能实现的功能，并提出改进建议，确保用户可以通过接口、Hook、透传等方式扩展功能，而无需修改库代码。

## ✅ 已修复的核心问题

### 1. Fanout 合并策略接口化 ✅

**状态**：已实现

**改进内容**：
- 定义了 `MergeStrategy` 接口，支持自定义合并策略
- 提供了三个内置实现：`FirstMergeStrategy`、`UnionMergeStrategy`、`PriorityMergeStrategy`
- 保持向后兼容：支持通过 `MergeStrategyName` 字符串指定策略

**使用示例**：
```go
// 使用自定义合并策略
fanout := &recall.Fanout{
    Sources: []recall.Source{...},
    Dedup: true,
    MergeStrategy: &CustomMergeStrategy{}, // 自定义策略
}

// 或使用内置策略（向后兼容）
fanout := &recall.Fanout{
    Sources: []recall.Source{...},
    Dedup: true,
    MergeStrategyName: "priority", // 字符串方式
}
```

### 2. 工厂模式支持动态注册 ✅

**状态**：已实现

**改进内容**：
- `NodeFactory` 支持线程安全的动态注册
- 添加了 `Unregister` 和 `ListRegisteredTypes` 方法
- 用户可以在运行时注册自定义 Node 类型，无需修改库代码

**使用示例**：
```go
factory := pipeline.NewNodeFactory()

// 注册自定义 Node
factory.Register("my.custom.node", func(config map[string]interface{}) (pipeline.Node, error) {
    return &MyCustomNode{}, nil
})

// 查看已注册的类型
types := factory.ListRegisteredTypes()

// 构建 Node
node, err := factory.Build("my.custom.node", config)
```

### 3. Pipeline Hook 机制 ✅

**状态**：已实现

**改进内容**：
- 定义了 `PipelineHook` 接口，支持在执行前后插入逻辑
- Pipeline 支持多个 Hook，按顺序执行
- 可以用于日志、监控、缓存、性能分析等场景

**使用示例**：
```go
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{...},
    Hooks: []pipeline.PipelineHook{
        &LoggingHook{},    // 日志记录
        &MetricsHook{},    // 性能监控
        &CacheHook{},      // 缓存
    },
}
```

**示例代码**：见 `examples/extensibility/main.go`

## 设计原则检查

### ✅ 已实现的设计模式

1. **接口抽象**：核心功能都通过接口定义
   - `recall.Source`：召回源接口
   - `filter.Filter`：过滤器接口
   - `model.RankModel`：排序模型接口
   - `feature.FeatureService`：特征服务接口
   - `store.Store`：存储接口

2. **策略模式**：支持多种实现策略
   - 多种召回算法（U2I、I2I、MF、ANN、Content）
   - 多种排序模型（LR、DNN、RPC）
   - 多种存储实现（Memory、Redis）

3. **工厂模式**：通过工厂创建实例
   - `config.DefaultFactory()`：配置化工厂
   - `feature.NewFeatureServiceFactory()`：特征服务工厂

4. **适配器模式**：适配不同系统
   - `VectorStoreAdapter`：适配向量服务
   - `FeatureServiceAdapter`：适配 Feast

## ❌ 需要改进的问题

### 1. Fanout 合并策略硬编码

**问题位置**：`recall/fanout.go`

**问题描述**：
- 合并策略通过字符串匹配（"first", "union", "priority"）实现
- 无法扩展自定义合并策略
- 优先级计算固定使用数组索引，无法自定义优先级权重

**当前实现**：
```go
switch n.MergeStrategy {
case "priority":
    return n.mergeByPriority(all), nil
case "union":
    return n.mergeUnion(all), nil
default:
    return n.mergeFirst(all), nil
}
```

**改进方案**：
```go
// 定义合并策略接口
type MergeStrategy interface {
    Merge(items []*core.Item, dedup bool) []*core.Item
}

// Fanout 支持自定义合并策略
type Fanout struct {
    Sources       []Source
    Dedup         bool
    Timeout       time.Duration
    MaxConcurrent int
    MergeStrategy MergeStrategy  // 改为接口
    // 保留字符串字段用于向后兼容
    MergeStrategyName string
}

// 默认实现
type FirstMergeStrategy struct{}
type UnionMergeStrategy struct{}
type PriorityMergeStrategy struct {
    PriorityWeights map[string]int  // 自定义优先级权重
}
```

### 2. Label 合并规则硬编码

**问题位置**：`pkg/utils/label.go`

**问题描述**：
- `MergeLabel` 函数使用固定的合并规则（"|" 分隔 Value，"," 分隔 Source）
- 无法自定义合并逻辑（如优先级覆盖、数值累加等）

**当前实现**：
```go
func MergeLabel(existing Label, incoming Label) Label {
    merged.Value = existing.Value + "|" + incoming.Value
    merged.Source = existing.Source + "," + incoming.Source
    return merged
}
```

**改进方案**：
```go
// 定义 Label 合并策略接口
type LabelMergeStrategy interface {
    Merge(existing, incoming Label) Label
}

// 默认实现
type DefaultLabelMergeStrategy struct{}
type PriorityLabelMergeStrategy struct {
    Priority map[string]int  // Source 优先级
}
type AccumulateLabelMergeStrategy struct{}  // 数值累加

// Item 支持自定义合并策略
type Item struct {
    // ...
    LabelMergeStrategy LabelMergeStrategy  // 可选，nil 使用默认策略
}
```

### 3. 特征提取器硬编码

**问题位置**：`feature/enrich.go`

**问题描述**：
- `defaultCrossFeatures` 硬编码了关键特征列表
- 无法自定义交叉特征生成规则

**当前实现**：
```go
keyUserFeatures := []string{"age", "gender", "user_id"}
keyItemFeatures := []string{"ctr", "cvr", "price", "score"}
```

**改进方案**：
```go
type EnrichNode struct {
    // ...
    // 支持自定义关键特征列表
    KeyUserFeatures []string
    KeyItemFeatures []string
    // 或使用函数
    KeyFeatureSelector func(features map[string]float64) []string
}
```

### 4. 错误处理策略硬编码

**问题位置**：`recall/fanout.go`

**问题描述**：
- 召回源失败时固定返回空结果，无法自定义错误处理策略
- 无法实现重试、降级、部分结果返回等策略

**当前实现**：
```go
items, err := s.Recall(recallCtx, rctx)
if err != nil {
    return nil  // 固定返回空结果
}
```

**改进方案**：
```go
// 定义错误处理策略接口
type ErrorHandler interface {
    HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error)
}

// 默认实现
type IgnoreErrorHandler struct{}  // 忽略错误，返回空
type RetryErrorHandler struct {
    MaxRetries int
    RetryDelay time.Duration
}
type FallbackErrorHandler struct {
    FallbackSource Source
}

type Fanout struct {
    // ...
    ErrorHandler ErrorHandler  // 自定义错误处理
}
```

### 5. 排序逻辑硬编码

**问题位置**：`rank/lr_node.go`

**问题描述**：
- 排序固定为降序，无法自定义排序规则
- 无法实现多维度排序、稳定性排序等

**当前实现**：
```go
sort.SliceStable(items, func(i, j int) bool {
    return items[i].Score > items[j].Score
})
```

**改进方案**：
```go
// 定义排序策略接口
type SortStrategy interface {
    Sort(items []*core.Item)
}

// 默认实现
type ScoreDescSortStrategy struct{}
type ScoreAscSortStrategy struct{}
type MultiFieldSortStrategy struct {
    Fields []SortField
}

type LRNode struct {
    Model model.RankModel
    SortStrategy SortStrategy  // 可选，nil 使用默认降序
}
```

### 6. 工厂模式硬编码

**问题位置**：`config/factory.go`

**问题描述**：
- `DefaultFactory` 中硬编码了所有 Node 类型的构建器
- 无法扩展新的 Node 类型，必须修改包代码

**当前实现**：
```go
func DefaultFactory() *pipeline.NodeFactory {
    factory.Register("recall.fanout", buildFanoutNode)
    factory.Register("recall.hot", buildHotNode)
    // ...
}
```

**改进方案**：
```go
// 工厂支持动态注册
type NodeFactory struct {
    builders map[string]NodeBuilder
    mutex    sync.RWMutex
}

func (f *NodeFactory) Register(name string, builder NodeBuilder) {
    f.mutex.Lock()
    defer f.mutex.Unlock()
    f.builders[name] = builder
}

// 用户可以在自己的代码中注册自定义 Node
factory := config.DefaultFactory()
factory.Register("my.custom.node", buildMyCustomNode)
```

### 7. 超时和默认值硬编码

**问题位置**：多处

**问题描述**：
- 多处硬编码了默认值（超时时间、前缀、TopK 等）
- 无法全局配置默认值

**改进方案**：
```go
// 定义全局配置接口
type GlobalConfig interface {
    DefaultTimeout() time.Duration
    DefaultMaxConcurrent() int
    DefaultFeaturePrefixes() FeaturePrefixes
}

// 支持从环境变量或配置文件加载
type Config struct {
    DefaultTimeout time.Duration
    // ...
}

// Pipeline 支持全局配置
type Pipeline struct {
    Nodes []Node
    Config *Config  // 全局配置
}
```

### 8. Pipeline 执行流程硬编码

**问题位置**：`pipeline/pipeline.go`

**问题描述**：
- Pipeline 执行是固定的顺序执行，无法插入 Hook
- 无法实现执行前/后的拦截器、中间件等

**当前实现**：
```go
func (p *Pipeline) Run(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
    for _, node := range p.Nodes {
        next, err := node.Process(ctx, rctx, cur)
        // ...
    }
}
```

**改进方案**：
```go
// 定义 Pipeline Hook 接口
type PipelineHook interface {
    BeforeNode(ctx context.Context, rctx *core.RecommendContext, node Node, items []*core.Item) ([]*core.Item, error)
    AfterNode(ctx context.Context, rctx *core.RecommendContext, node Node, items []*core.Item) ([]*core.Item, error)
}

type Pipeline struct {
    Nodes []Node
    Hooks []PipelineHook  // 支持多个 Hook
}

// 实现示例：日志 Hook、监控 Hook、缓存 Hook
type LoggingHook struct{}
type MetricsHook struct{}
type CacheHook struct{}
```

### 9. 特征前缀硬编码

**问题位置**：`feature/enrich.go`

**问题描述**：
- 特征前缀有默认值（"user_", "item_", "cross_"），但无法全局配置
- 每个 EnrichNode 都需要单独配置

**改进方案**：
```go
// 支持全局默认前缀配置
type FeatureConfig struct {
    DefaultUserPrefix  string
    DefaultItemPrefix  string
    DefaultCrossPrefix string
}

// EnrichNode 支持从全局配置读取
type EnrichNode struct {
    // ...
    Config *FeatureConfig  // 可选，nil 使用默认值
}
```

### 10. 优先级计算硬编码

**问题位置**：`recall/fanout.go`

**问题描述**：
- 优先级固定使用数组索引（索引越小优先级越高）
- 无法为不同召回源设置不同的优先级权重

**当前实现**：
```go
priority := i  // 索引即优先级
```

**改进方案**：
```go
// Source 支持优先级配置
type SourceWithPriority struct {
    Source   Source
    Priority int  // 自定义优先级
}

type Fanout struct {
    Sources []Source  // 或 []SourceWithPriority
    // 或使用映射
    SourcePriorities map[string]int
}
```

### 11. 相似度计算硬编码

**问题位置**：`recall/collaborative_filtering.go`

**问题描述**：
- 相似度计算通过字符串匹配（"cosine", "pearson"）实现
- 无法扩展自定义相似度计算方法

**当前实现**：
```go
switch metric {
case "pearson":
    sim = pearsonCorrelation(targetScores, userScores)
case "cosine":
    fallthrough
default:
    sim = cosineSimilarityVector(targetScores, userScores)
}
```

**改进方案**：
```go
// 定义相似度计算接口
type SimilarityCalculator interface {
    Calculate(x, y []float64) float64
}

// 默认实现
type CosineSimilarity struct{}
type PearsonCorrelation struct{}

type UserBasedCF struct {
    Store CFStore
    // ...
    SimilarityCalculator SimilarityCalculator  // 可选，nil 使用默认
    SimilarityMetric      string                // 向后兼容
}
```

### 12. 默认值硬编码（多处）

**问题位置**：多个文件

**问题描述**：
- 多处硬编码了默认值，无法全局配置
- 例如：TopK 默认值、超时时间、最小共同物品数等

**当前实现示例**：
```go
topKSimilar := r.TopKSimilarUsers
if topKSimilar <= 0 {
    topKSimilar = 50  // 硬编码默认值
}

minCommon := r.MinCommonItems
if minCommon <= 0 {
    minCommon = 2  // 硬编码默认值
}
```

**改进方案**：
```go
// 定义配置接口
type RecallConfig interface {
    DefaultTopKSimilarUsers() int
    DefaultMinCommonItems() int
    DefaultTopKItems() int
}

// 支持从配置读取默认值
type UserBasedCF struct {
    Store CFStore
    Config RecallConfig  // 可选，nil 使用硬编码默认值
    // ...
}
```

## 改进优先级

### 高优先级（必须改进）

1. **Fanout 合并策略接口化**：影响多路召回的核心功能
2. **工厂模式支持动态注册**：影响扩展性
3. **Pipeline Hook 机制**：影响可观测性和中间件能力

### 中优先级（建议改进）

4. **Label 合并策略接口化**：影响 Label 的灵活性
5. **错误处理策略接口化**：影响容错能力
6. **排序策略接口化**：影响排序灵活性

### 低优先级（可选改进）

7. **全局配置支持**：提升易用性
8. **特征提取器可配置**：提升灵活性

## 改进实施建议

### 阶段一：核心接口抽象

1. 定义合并策略接口 `MergeStrategy`
2. 定义 Pipeline Hook 接口 `PipelineHook`
3. 定义错误处理接口 `ErrorHandler`

### 阶段二：工厂模式增强

1. 工厂支持动态注册
2. 提供注册示例和文档

### 阶段三：策略模式完善

1. Label 合并策略接口化
2. 排序策略接口化
3. 错误处理策略接口化

### 阶段四：配置和 Hook

1. 全局配置支持
2. Pipeline Hook 机制实现
3. 提供常用 Hook 实现（日志、监控）

## 向后兼容性

所有改进都应保持向后兼容：

1. **保留原有字段**：新增接口字段的同时保留原有字符串/布尔字段
2. **默认实现**：提供默认实现，确保原有代码继续工作
3. **渐进式迁移**：提供迁移指南和示例

## 示例：改进后的使用方式

```go
// 1. 自定义合并策略
type MyMergeStrategy struct {
    CustomRules map[string]int
}
func (s *MyMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
    // 自定义逻辑
}

// 2. 注册自定义 Node
factory := config.DefaultFactory()
factory.Register("my.recall", buildMyRecallNode)

// 3. 添加 Pipeline Hook
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{...},
    Hooks: []pipeline.PipelineHook{
        &MyLoggingHook{},
        &MyMetricsHook{},
    },
}

// 4. 自定义错误处理
fanout := &recall.Fanout{
    Sources: []recall.Source{...},
    ErrorHandler: &MyRetryErrorHandler{
        MaxRetries: 3,
    },
}
```

## 总结

Reckit 作为推荐系统工具库，在接口抽象方面已经做得不错，但在策略可扩展性、Hook 机制、工厂模式等方面还有改进空间。通过上述改进，可以确保用户无需修改库代码即可实现所有自定义需求，真正实现"开箱即用，灵活扩展"的设计目标。
