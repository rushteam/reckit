# Reckit 架构设计

## 概述

Reckit 是一个工业级推荐系统工具库，采用 **Pipeline + Node** 架构，通过接口抽象实现高度可扩展性。

**设计目标**：
- **Golang**：负责高并发推荐服务、低延迟在线推理、统计计算
- **PyTorch/TensorFlow**：负责深度模型训练、复杂梯度更新

## 核心架构

```
Request → Context → Recall → Filter → Rank → ReRank → Response
                      ↓        ↓       ↓       ↓
                   Source   Filter   Model   Diversity
```

### 架构层次

```
┌─────────────────────────────────────────┐
│         应用层（Application）            │
│  - 业务逻辑 & 推荐流水线配置              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         领域层（Domain）                 │
│  - core: Context, UserProfile, Item     │
│  - pipeline: Pipeline, Node            │
│  - recall: Source                       │
│  - rank: RankModel                      │
│  - feature: FeatureService              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      基础设施层（Infrastructure）        │
│  - store: Redis, Memory                │
│  - vector: Milvus                       │
│  - service: TF Serving, TorchServe     │
│  - ext/feast: Feast Client             │
└─────────────────────────────────────────┘
```

## 核心概念

### 1. RecommendContext

推荐请求的上下文，贯穿整个 Pipeline。

```go
type RecommendContext struct {
    UserID      string              // 用户 ID
    DeviceID    string              // 设备 ID
    Scene       string              // 场景（feed/search/detail）
    User        *UserProfile        // 强类型用户画像
    UserProfile map[string]any      // Map 形式用户画像
    Labels      map[string]Label    // 用户级标签
    Params      map[string]any      // 请求参数（latitude, time_of_day 等）
}
```

### 2. Item

推荐物品，携带特征和标签。

```go
type Item struct {
    ID       string                 // 物品 ID
    Score    float64                // 排序分数
    Features map[string]float64     // 特征（供排序模型使用）
    Meta     map[string]any         // 元数据
    Labels   map[string]Label       // 标签（召回来源、模型名称等）
}
```

### 3. Pipeline & Node

Pipeline 由多个 Node 组成，Node 是处理单元。

```go
type Node interface {
    Name() string
    Kind() Kind  // recall / filter / rank / rerank
    Process(ctx context.Context, rctx *RecommendContext, items []*Item) ([]*Item, error)
}
```

### 4. UserProfile

用户画像，包含静态属性、兴趣、行为等。

```go
type UserProfile struct {
    UserID        string
    Gender, Age   string, int
    Interests     map[string]float64   // 长期兴趣
    RecentClicks  []string             // 短期行为
    Buckets       map[string]string    // A/B 实验桶
    Extras        map[string]any       // 扩展属性
}
```

## 设计原则

### 1. 接口优先

所有策略都通过接口实现，不使用字符串匹配。用户可以通过实现接口扩展功能。

```go
// 召回源接口
type Source interface {
    Name() string
    Recall(ctx context.Context, rctx *RecommendContext) ([]*Item, error)
}

// 排序模型接口
type RankModel interface {
    Name() string
    Predict(features map[string]float64) (float64, error)
}

// 过滤器接口
type Filter interface {
    Name() string
    ShouldFilter(ctx context.Context, rctx *RecommendContext, item *Item) (bool, error)
}
```

### 2. 策略模式

不同场景使用不同策略，通过接口注入。

| 策略接口 | 说明 | 内置实现 |
|---------|------|---------|
| `MergeStrategy` | 召回合并 | First, Union, Priority |
| `ErrorHandler` | 错误处理 | Ignore, Retry, Fallback |
| `SortStrategy` | 排序策略 | ScoreDesc, ScoreAsc, MultiField |
| `SimilarityCalculator` | 相似度计算 | Cosine, Pearson |

### 3. 通用 ID 类型

所有 ID 使用 `string` 类型，支持 UUID、数字 ID、任意字符串。

### 4. 无外部依赖

核心包无外部依赖，具体实现位于 `ext/` 扩展包：
- `ext/store/redis` - Redis 存储
- `ext/feast` - Feast 特征服务
- `ext/vector/milvus` - Milvus 向量库

## Pipeline 生命周期

### 常驻部分（服务启动时初始化）

- Pipeline 实例
- Store 连接
- FeatureService
- MLService 客户端
- 配置参数（TopK、Timeout 等）

### 请求部分（每次请求创建）

- RecommendContext
- UserProfile（从存储加载）
- Items 列表

```go
// 初始化（服务启动时）
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{Sources: sources},
        &filter.FilterNode{Filters: filters},
        &feature.EnrichNode{FeatureService: featureService},
        &rank.DNNNode{Model: dnnModel},
    },
}

// 请求处理（每次请求）
func HandleRequest(userID string) ([]*core.Item, error) {
    rctx := &core.RecommendContext{
        UserID: userID,
        Params: map[string]any{"latitude": 39.9, "time_of_day": 14.5},
    }
    return pipeline.Run(ctx, rctx, nil)
}
```

## 目录结构

```
github.com/rushteam/reckit/
├── core/           # 核心数据结构（Item, Context, UserProfile）
├── pipeline/       # Pipeline 和 Node 接口
├── recall/         # 召回模块（Source, Fanout, CF, ANN）
├── filter/         # 过滤模块（Blacklist, Exposed）
├── rank/           # 排序模块（LR, DNN, DIN, RPC）
├── rerank/         # 重排模块（Diversity, TopN）
├── model/          # 排序模型抽象
├── feature/        # 特征服务
├── service/        # ML 服务（TF Serving, TorchServe）
├── store/          # 存储抽象（Memory）
├── config/         # Pipeline 配置工厂
├── pkg/            # 工具包（conv, dsl, utils）
├── ext/            # 扩展包（Redis, Feast, Milvus）
└── python/         # Python 训练和服务
```

## 相关文档

- [扩展指南](./EXTENSIBILITY.md) - 如何扩展自定义功能
- [Feature 模块](./FEATURE_MODULE.md) - 特征相关模块职责
- [召回算法](./RECALL_ALGORITHMS.md) - 所有召回算法
- [排序模型](./RANK_MODELS.md) - 所有排序模型
