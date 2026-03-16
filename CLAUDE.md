# Reckit - AI Coding 指南

## 项目概述

Reckit 是一个工业级推荐系统工具库，采用 **Pipeline + Node** 架构，通过接口抽象实现高度可扩展性。

**包名**：`github.com/rushteam/reckit`

### 工程目标

场景分工：
- **深度模型、复杂梯度更新** → 适用工具：**PyTorch/Tensorflow**
- **高并发、低延迟、统计计算** → 适用工具：**Golang(Reckit)**

本项目采用 **PyTorch/Tensorflow + Golang** 的分工模式：
- **PyTorch/Tensorflow**：负责深度模型的训练、复杂梯度更新等机器学习任务
- **Golang (Reckit)**：负责高并发推荐服务、低延迟在线推理、统计计算等生产环境任务

**核心设计原则**：
- 所有策略、算法、配置都通过**接口**实现，不使用字符串匹配
- 用户可以通过实现接口扩展功能，**无需修改库代码**
- 使用 `string` 类型作为通用 ID（支持所有 ID 格式）
- 所有默认值通过配置接口提供，无硬编码

## 核心架构

```
Request → Context → Recall → Filter → Rank → ReRank → PostProcess → Response
```

### 架构层次

1. **应用层**：业务逻辑和 Pipeline 配置
2. **领域层**：核心抽象（Node、Source、Model、Service）
3. **基础设施层**：存储、向量库、ML 服务集成

## 关键接口

### 核心接口（领域层接口在 core 包）

```go
// Pipeline Node（所有处理单元的基础接口）
type Node interface {
    Name() string
    Kind() Kind  // recall / filter / rank / rerank / postprocess
    Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error)
}

// 召回源接口
type Source interface {
    Name() string
    Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error)
}

// 排序模型接口
type RankModel interface {
    Name() string
    Predict(features map[string]float64) (float64, error)
}

// 过滤器接口（逐条过滤）
type Filter interface {
    Name() string
    ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error)
}

// 批量过滤器接口（可选实现，适用于需要批量查询外部服务的场景）
// FilterNode 优先调用 BatchFilter.FilterBatch，降级到 Filter.ShouldFilter
type BatchFilter interface {
    Filter
    FilterBatch(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error)
}

// 存储接口（领域层接口，在 core 包）
type Store interface {
    Get(ctx context.Context, key string) ([]byte, error)
    Set(ctx context.Context, key string, value []byte, ttl int64) error
    // ...
}

// 特征服务接口（领域层接口，在 core 包）
type FeatureService interface {
    GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error)
    BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error)
    GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)
    BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error)
    GetRealtimeFeatures(ctx context.Context, userID, itemID string) (map[string]float64, error)
    BatchGetRealtimeFeatures(ctx context.Context, pairs []FeatureUserItemPair) (map[FeatureUserItemPair]map[string]float64, error)
    Close() error
}

// 向量服务接口（领域层接口，在 core 包）
type VectorService interface {
    Search(ctx context.Context, req *VectorSearchRequest) (*VectorSearchResult, error)
    Close() error
}

// 向量数据库服务接口（领域层接口，在 core 包）
type VectorDatabaseService interface {
    VectorService  // 嵌入召回场景接口
    Insert(ctx context.Context, req *VectorInsertRequest) error
    Update(ctx context.Context, req *VectorUpdateRequest) error
    Delete(ctx context.Context, req *VectorDeleteRequest) error
    CreateCollection(ctx context.Context, req *VectorCreateCollectionRequest) error
    DropCollection(ctx context.Context, collection string) error
    HasCollection(ctx context.Context, collection string) (bool, error)
}

// ML 服务接口（领域层接口，在 core 包）
type MLService interface {
    Predict(ctx context.Context, req *MLPredictRequest) (*MLPredictResponse, error)
    Health(ctx context.Context) error
    Close() error
}
```

### 策略接口（可扩展）

```go
// 合并策略（recall/fanout.go + recall/merge_strategy.go）
type MergeStrategy interface {
    Merge(items []*core.Item, dedup bool) []*core.Item
}
// 去重策略：FirstMergeStrategy, UnionMergeStrategy, PriorityMergeStrategy
// 混排策略：WeightedScoreMergeStrategy, QuotaMergeStrategy, RatioMergeStrategy, RoundRobinMergeStrategy, WaterfallMergeStrategy
// 组合策略：ChainMergeStrategy（串联多个策略，前一个输出作为下一个输入）

// 错误处理策略（recall/fanout.go）
type ErrorHandler interface {
    HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error)
}
// 内置实现：IgnoreErrorHandler, RetryErrorHandler, FallbackErrorHandler

// 排序策略（rank/lr_node.go）
type SortStrategy interface {
    Sort(items []*core.Item)
}
// 内置实现：ScoreDescSortStrategy, ScoreAscSortStrategy, MultiFieldSortStrategy

// 相似度计算器（recall/collaborative_filtering.go）
type SimilarityCalculator interface {
    Calculate(x, y []float64) float64
}
// 内置实现：CosineSimilarity, PearsonCorrelation

// Label 合并策略（pkg/utils/label.go）
type LabelMergeStrategy interface {
    Merge(existing, incoming Label) Label
}
// 内置实现：DefaultLabelMergeStrategy, PriorityLabelMergeStrategy

// Pipeline Hook（pipeline/pipeline.go）
type PipelineHook interface {
    BeforeNode(ctx context.Context, rctx *core.RecommendContext, node Node, items []*core.Item) ([]*core.Item, error)
    AfterNode(ctx context.Context, rctx *core.RecommendContext, node Node, items []*core.Item, err error) ([]*core.Item, error)
}

// 特征元数据加载器接口（feature/metadata_loader.go）
type MetadataLoader interface {
    Load(ctx context.Context, source string) (*FeatureMetadata, error)
}
// 内置实现：FileMetadataLoader, HTTPMetadataLoader, S3MetadataLoader

// 特征标准化器加载器接口（feature/metadata_loader.go）
type ScalerLoader interface {
    Load(ctx context.Context, source string) (FeatureScaler, error)
}
// 内置实现：FileScalerLoader, HTTPScalerLoader, S3ScalerLoader

// S3 兼容协议客户端接口（feature/oss_loader.go）
type S3Client interface {
    GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error)
}
// 支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等
```

### 配置接口

```go
// 召回配置（core/config.go）
type RecallConfig interface {
    DefaultTopKSimilarUsers() int
    DefaultTopKItems() int
    DefaultMinCommonItems() int
    DefaultMinCommonUsers() int
    DefaultTimeout() time.Duration
}
// 默认实现：DefaultRecallConfig
```

## 核心数据结构

### RecommendContext

```go
type RecommendContext struct {
    UserID   string
    DeviceID string
    Scene    string
    
    User        *UserProfile        // 强类型用户画像
    UserProfile map[string]any     // Map 形式用户画像
    Labels      map[string]utils.Label  // 用户级标签
    Realtime    map[string]any
    Params      map[string]any
}
```

### Item

```go
type Item struct {
    ID       string
    Score    float64
    Features map[string]float64
    Meta     map[string]any
    Labels   map[string]utils.Label
    
    LabelMergeStrategy utils.LabelMergeStrategy  // 可选，自定义 Label 合并策略
}
```

**Item 三字段分工**：

| 字段 | 类型 | 语义 | 消费者 | 典型内容 |
|------|------|------|--------|----------|
| `Features` | `map[string]float64` | 模型特征（数值型） | Rank 模型 (`model.Predict`) | `ctr`, `cvr`, `price`, `user_age` |
| `Meta` | `map[string]any` | 业务元数据（任意类型） | 业务逻辑、展示层、过滤 | `title`, `category`, `author`, `cover_url` |
| `Labels` | `map[string]Label` | 策略标签（带来源追踪） | 策略引擎、Diversity、DSL、可解释性 | `recall_source`, `recall_priority`, `boost_reason` |

- **Features**：由 `feature.EnrichNode` 注入，供排序模型直接消费，不带来源信息
- **Meta**：业务侧自行填充，存储展示、过滤所需的结构化信息，值类型不限
- **Labels**：推荐链路各阶段（recall / rank / rerank / rule）写入，每个 Label 记录 `Value` + `Source`，同名 key 按合并策略（默认 `|` 拼接）自动合并

**字段查找优先级**：`Diversity` 等节点通过 `getValue(item, key)` 读取字段时，按 **Labels > Meta > Features** 的优先级查找。离散类别（如 category、author）建议放 Labels 或 Meta；仅当特征编码为数值放在 Features 时，才会 fallback 到 Features（float64 自动转为字符串）。

### UserProfile

```go
type UserProfile struct {
    UserID        string
    Gender        string
    Age           int
    Location      string
    Interests     map[string]float64
    RecentClicks  []string
    RecentImpress []string
    PreferTags    map[string]float64
    Buckets       map[string]string
    Extras        map[string]any  // 扩展字段（用户自定义属性）
    UpdateTime    time.Time
}
```

## 目录结构

```
github.com/rushteam/reckit/
├── core/              # 核心数据结构（Item, Context, UserProfile, Config）
├── pipeline/          # Pipeline 和 Node 接口
├── recall/            # 召回模块（Source, Fanout, CF, ANN, Content 等）
├── filter/            # 过滤模块（Blacklist, UserBlock, Exposed）
├── rank/              # 排序模块（LR, DNN, DIN, RPC 等）
├── rerank/            # 重排模块（Diversity）
├── model/             # 排序模型抽象和实现
├── feature/           # 特征服务（Enrich, Service, Provider）
├── store/             # 存储抽象（Memory，Redis 移至扩展包）
├── vector/             # 向量服务接口（Milvus 移至扩展包）
├── service/           # ML 服务（TF Serving, ANN Service）
├── feature/           # 特征服务实现（接口在 core.FeatureService）
├── config/            # Pipeline 配置工厂
├── ext/                # 扩展包目录（独立 go.mod）
│   ├── store/
│   │   └── redis/     # Redis 存储实现
│   ├── feast/
│   │   ├── http/      # Feast HTTP 客户端实现（适配为 core.FeatureService）
│   │   └── grpc/      # Feast gRPC 客户端实现（适配为 core.FeatureService）
│   └── vector/
│       └── milvus/    # Milvus 向量数据库实现
└── pkg/
    ├── utils/         # Label 工具
    ├── dsl/           # Label DSL 表达式引擎
    └── conv/          # 类型转换与泛型工具（ToFloat64、ConfigGet、MapToFloat64 等）
```

### 扩展包说明

核心包 `github.com/rushteam/reckit` **无外部依赖**，只保留工具库（CEL、YAML、sync）。

具体实现已移至 `ext/` 扩展包，每个扩展包有独立的 `go.mod`，用户按需引入：

- **Redis Store**: `go get github.com/rushteam/reckit/ext/store/redis`
- **Feast gRPC**: `go get github.com/rushteam/reckit/ext/feast/grpc`
- **Milvus Vector**: `go get github.com/rushteam/reckit/ext/vector/milvus`

详见 `ext/README.md`。

## 关键文件位置

### 核心文件

- `core/item.go` - Item 定义和 Label 操作
- `core/context.go` - RecommendContext 定义
- `core/user_profile.go` - UserProfile 定义
- `core/config.go` - 配置接口定义
- `pipeline/node.go` - Node 接口定义
- `pipeline/pipeline.go` - Pipeline 执行器和 Hook
- `pipeline/config.go` - 配置加载和工厂

### 召回模块

- `recall/source.go` - Source 接口
- `recall/fanout.go` - 多路并发召回和去重合并策略（First/Union/Priority），Fanout 同时实现 Source 接口支持嵌套
- `recall/merge_strategy.go` - 混排合并策略（WeightedScore/Quota/Ratio/RoundRobin/Waterfall）和组合策略（Chain）
- `recall/collaborative_filtering.go` - U2I/I2I 协同过滤
- `recall/ann.go` - Embedding ANN 召回
- `recall/content.go` - 内容推荐
- `recall/matrix_factorization.go` - 矩阵分解召回
- `recall/word2vec_recall.go` - Word2Vec / Item2Vec 召回（文本模式 + 序列模式）
- `recall/bert_recall.go` - BERT 召回（基于语义相似度）
- `recall/hot.go` - 热门召回
- `recall/user_history.go` - 用户历史召回

### 排序模块

- `rank/lr_node.go` - LR 排序节点（含排序策略）
- `rank/rpc_node.go` - RPC 排序节点
- `rank/dnn_node.go` - DNN 排序节点
- `rank/din_node.go` - DIN 排序节点
- `rank/wide_deep_node.go` - Wide&Deep 排序节点
- `rank/two_tower_node.go` - Two Tower 排序节点
- `model/model.go` - RankModel 接口
- `model/lr.go` - LR 模型实现
- `model/rpc.go` - RPC 模型实现
- `model/word2vec.go` - Word2Vec 模型实现
- `model/bert.go` - BERT 模型实现

### 重排模块

- `rerank/diversity.go` - 多样性重排（类别去重 + 作者打散，字段查找优先级：Labels > Meta > Features）
- `rerank/topn.go` - TopN 截断

### 特征模块

- `core/feature_service.go` - FeatureService 领域接口
- `feature/service.go` - FeatureService 实现（BaseFeatureService）
- `feature/enrich.go` - 特征注入节点
- `feature/store_provider.go` - 存储特征提供者
- `feature/metadata.go` - 特征元数据和标准化器定义
- `feature/metadata_loader.go` - 特征元数据加载器接口和文件实现
- `feature/http_loader.go` - HTTP 接口加载器实现
- `feature/oss_loader.go` - S3 兼容协议加载器实现（支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等）
- `feature/processing.go` - 特征处理工具类（归一化、分箱、交叉特征等）
- `feature/encoder.go` - 特征编码工具类（One-Hot、Label、Hash、Embedding 等）

### 工具模块

- `pkg/utils/label.go` - Label 定义和合并策略
- `pkg/dsl/eval.go` - Label DSL 表达式引擎
- `pkg/conv/conv.go` - 类型转换与泛型工具（ToFloat64、ToInt、ToString、ConfigGet、MapToFloat64、SliceAnyToString 等）

## 使用模式

### 1. 创建 Pipeline

```go
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        // 召回
        &recall.Fanout{
            Sources: []recall.Source{
                &recall.Hot{IDs: []string{"1", "2", "3"}},
                &recall.U2IRecall{...},
            },
            Dedup:         true,
            MergeStrategy: &recall.PriorityMergeStrategy{},
        },
        // 过滤
        &filter.FilterNode{
            Filters: []filter.Filter{
                filter.NewBlacklistFilter(...),
            },
        },
        // 特征注入
        &feature.EnrichNode{
            FeatureService: featureService, // core.FeatureService 接口
        },
        // 排序
        &rank.LRNode{
            Model:        lrModel,
            SortStrategy: &rank.ScoreDescSortStrategy{},
        },
    },
    Hooks: []pipeline.PipelineHook{
        &LoggingHook{},
    },
}

items, err := p.Run(ctx, rctx, nil)
```

### 2. 创建召回源

```go
config := &core.DefaultRecallConfig{}
u2i := &recall.U2IRecall{
    Store:                cfStore,
    TopKSimilarUsers:     10,
    TopKItems:            20,
    SimilarityCalculator: &recall.CosineSimilarity{},
    Config:                config,
}
```

### 3. 自定义特征抽取器

```go
import "github.com/rushteam/reckit/feature"

// 方式 1：使用默认抽取器（带前缀）
extractor := feature.NewDefaultFeatureExtractor(
    feature.WithFeatureService(featureService),
    feature.WithFieldPrefix("user_"),
)

// 方式 2：完全自定义
customExtractor := feature.NewCustomFeatureExtractor(
    "my_extractor",
    func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
        features := make(map[string]float64)
        // 自定义逻辑：从多个源组合特征
        if rctx.User != nil {
            features["age"] = float64(rctx.User.Age)
        }
        // 从外部服务获取
        externalFeatures, _ := externalService.GetFeatures(ctx, rctx.UserID)
        for k, v := range externalFeatures {
            features["external_"+k] = v
        }
        return features, nil
    },
)

// 在召回源中使用
twoTowerRecall := recall.NewTwoTowerRecall(
    featureService,
    userTowerService,
    vectorService,
    recall.WithTwoTowerUserFeatureExtractor(extractor), // 或 customExtractor
)
```

**详细文档**：见 `feature/EXTRACTOR_GUIDE.md`

### 4. 使用内置合并策略

```go
// 加权分数：按源权重调整分数后排序
fanout := &recall.Fanout{
    Sources: sources, Dedup: true,
    MergeStrategy: &recall.WeightedScoreMergeStrategy{
        SourceWeights: map[string]float64{"recall.hot": 2.0, "recall.cf": 1.0, "recall.ann": 1.5},
        TopN: 100,
    },
}

// 固定配额：每源取固定数量
fanout.MergeStrategy = &recall.QuotaMergeStrategy{
    SourceQuotas: map[string]int{"recall.hot": 10, "recall.cf": 20, "recall.ann": 15},
}

// 比例配比：按比例分配总量
fanout.MergeStrategy = &recall.RatioMergeStrategy{
    SourceRatios: map[string]float64{"recall.hot": 0.2, "recall.cf": 0.3, "recall.ann": 0.5},
    TotalLimit:   100,
}

// 轮询交叉：各源轮流取（适合信息流多样性）
fanout.MergeStrategy = &recall.RoundRobinMergeStrategy{
    SourceOrder: []string{"recall.cf", "recall.ann", "recall.hot"},
    TopN: 50,
}

// 瀑布级联：高优先级源优先填满，不足时低优先级源兜底
fanout.MergeStrategy = &recall.WaterfallMergeStrategy{
    SourcePriority: []string{"recall.cf", "recall.ann", "recall.hot"},
    TotalLimit:     100,
    SourceLimits:   map[string]int{"recall.hot": 20}, // 热门最多 20 条
}

// 组合策略：先加权调分，再按配额截取
fanout.MergeStrategy = &recall.ChainMergeStrategy{
    Strategies: []recall.MergeStrategy{
        &recall.WeightedScoreMergeStrategy{
            SourceWeights: map[string]float64{"recall.hot": 2.0, "recall.cf": 1.0},
        },
        &recall.QuotaMergeStrategy{
            SourceQuotas: map[string]int{"recall.hot": 10, "recall.cf": 30},
        },
    },
}
```

### 5. 嵌套 Fanout（树形召回拓扑）

Fanout 同时实现 Node 和 Source 接口，可嵌套在另一个 Fanout 中作为子召回源。
通过 NodeName 区分不同层级（影响 recall_source label）。

```go
topFanout := &recall.Fanout{
    NodeName: "recall.top",
    Sources: []recall.Source{
        &recall.Fanout{
            NodeName: "recall.personalized",
            Sources:  []recall.Source{cfRecall, annRecall},
            Dedup:    true,
            MergeStrategy: &recall.RatioMergeStrategy{
                SourceRatios: map[string]float64{"recall.cf": 0.4, "recall.ann": 0.6},
                TotalLimit:   60,
            },
        },
        &recall.Fanout{
            NodeName: "recall.non_personalized",
            Sources:  []recall.Source{hotRecall, trendingRecall},
            Dedup:    true,
            MergeStrategy: &recall.FirstMergeStrategy{},
        },
    },
    Dedup: true,
    MergeStrategy: &recall.QuotaMergeStrategy{
        SourceQuotas: map[string]int{
            "recall.personalized":     60,
            "recall.non_personalized": 30,
        },
    },
}
```

### 6. 自定义策略

```go
// 自定义合并策略
type CustomMergeStrategy struct{}
func (s *CustomMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
    // 自定义逻辑
}

// 使用
fanout := &recall.Fanout{
    MergeStrategy: &CustomMergeStrategy{},
}
```

### 7. 配置驱动 Pipeline + 自定义 Node 注册

reckit 提供 **Node 注册表 + 工厂函数** 模式，支持：
- 用户注册自定义 node type，YAML 中直接引用
- YAML `config` 字段**原样透传**给工厂函数（`map[string]interface{}`）
- 通过**闭包捕获**外部依赖（Redis client、Repository 等），实现依赖注入

#### 核心 API

| API | 说明 |
|---|---|
| `config.Register(typeName, builder)` | 全局注册 Node 构建器（推荐在 `init` 或启动阶段调用） |
| `config.DefaultFactory()` | 返回包含所有已注册 Node 的 `NodeFactory` |
| `config.SupportedTypes()` | 查看已注册类型列表 |
| `config.ValidatePipelineConfig(cfg)` | 校验 YAML 中引用的 type 是否全部已注册 |

#### 注册自定义 Node（含依赖注入）

```go
import "github.com/rushteam/reckit/config"

// 1. 初始化外部依赖
redisClient := redis.NewClient(...)
itemRepo := repository.NewItemRepo(db)

// 2. 注册自定义 Node —— 闭包捕获依赖，config 字段由 YAML 透传
config.Register("custom.boost", func(cfg map[string]interface{}) (pipeline.Node, error) {
    return &BoostNode{
        Repo:      itemRepo,                                 // 闭包注入
        Redis:     redisClient,                              // 闭包注入
        BoostRate: conv.ConfigGet(cfg, "boost_rate", 1.5),   // YAML 透传
        MaxItems:  int(conv.ConfigGetInt64(cfg, "max", 100)),
    }, nil
})
```

#### YAML 配置引用自定义 Node

```yaml
pipeline:
  name: "my_feed"
  nodes:
    - type: "recall.fanout"
      config:
        dedup: true
        sources:
          - type: "hot"
            ids: [1, 2, 3]
    - type: "rank.lr"
      config:
        bias: 0.0
        weights: { ctr: 1.2, cvr: 0.8 }
    # 自定义节点：type 对应 config.Register 的 typeName
    # config 下所有字段原样透传给工厂函数
    - type: "custom.boost"
      config:
        boost_rate: 2.0
        max: 50
```

#### 完整启动流程

```go
import (
    "github.com/rushteam/reckit/config"
    _ "github.com/rushteam/reckit/config/builders" // 注册内置 Node
)

func main() {
    // 1. 初始化依赖
    repo := newMyRepo(db)

    // 2. 注册自定义 Node（闭包捕获依赖）
    config.Register("custom.boost", func(cfg map[string]interface{}) (pipeline.Node, error) {
        return &BoostNode{Repo: repo, BoostRate: conv.ConfigGet(cfg, "boost_rate", 1.5)}, nil
    })

    // 3. 加载 YAML
    cfg, _ := pipeline.LoadFromYAML("pipeline.yaml")

    // 4. 校验配置（可选，提前发现未注册的 type）
    if err := config.ValidatePipelineConfig(cfg); err != nil {
        log.Fatal(err)
    }

    // 5. 构建并运行
    factory := config.DefaultFactory()
    p, _ := cfg.BuildPipeline(factory)
    items, _ := p.Run(ctx, rctx, nil)
}
```

> **注意**：内置 Node（`recall.fanout`、`rank.lr` 等）需要 `import _ "github.com/rushteam/reckit/config/builders"` 触发 `init` 注册。

## 扩展指南

### 添加新的召回算法

1. 实现 `recall.Source` 接口
2. （可选）在工厂中注册

```go
type MyRecall struct {
    Store MyStore
}

func (r *MyRecall) Name() string { return "recall.my" }
func (r *MyRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
    // 实现召回逻辑
    items := []*core.Item{}
    // ...
    return items, nil
}
```

### 添加新的过滤器

实现 `filter.Filter` 接口（逐条过滤）：

```go
type MyFilter struct{}

func (f *MyFilter) Name() string { return "filter.my" }
func (f *MyFilter) ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error) {
    // 返回 true 表示过滤，false 表示保留
    return false, nil
}
```

如果过滤逻辑涉及批量查询外部服务（如批量查 Redis 已曝光），额外实现 `filter.BatchFilter` 接口：

```go
type MyBatchFilter struct{}

func (f *MyBatchFilter) Name() string { return "filter.my_batch" }

// ShouldFilter 作为降级路径（非 Fanout 场景、或 FilterBatch 出错时）
func (f *MyBatchFilter) ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error) {
    // 逐条判断逻辑
    return false, nil
}

// FilterBatch 批量过滤，返回保留的 item 列表
func (f *MyBatchFilter) FilterBatch(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
    // 一次性批量查询外部服务，O(1) 次网络调用
    ids := make([]string, len(items))
    for i, it := range items {
        ids[i] = it.ID
    }
    blocked, _ := myStore.BatchCheck(ctx, rctx.UserID, ids)
    
    out := make([]*core.Item, 0, len(items))
    for _, it := range items {
        if !blocked[it.ID] {
            out = append(out, it)
        }
    }
    return out, nil
}
```

`FilterNode` 执行顺序：先执行所有 `BatchFilter.FilterBatch`（整批过滤），再对剩余 item 逐条执行普通 `Filter.ShouldFilter`。

### 添加新的排序模型

1. 实现 `model.RankModel` 接口
2. 创建对应的 Rank Node

```go
type MyModel struct{}

func (m *MyModel) Name() string { return "my_model" }
func (m *MyModel) Predict(features map[string]float64) (float64, error) {
    // 实现预测逻辑
    return 0.5, nil
}

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

## 设计模式

### 策略模式
- `MergeStrategy` - 合并策略（去重：First/Union/Priority；混排：WeightedScore/Quota/Ratio/RoundRobin/Waterfall；组合：Chain）
- `ErrorHandler` - 错误处理策略
- `SortStrategy` - 排序策略
- `SimilarityCalculator` - 相似度计算策略
- `RankModel` - 排序模型策略

### 工厂模式
- `NodeFactory` - Node 构建工厂（支持动态注册）
- `FeatureServiceFactory` - 特征服务工厂（创建 core.FeatureService 实现）
- `MLServiceFactory` - ML 服务工厂（创建 core.MLService 实现）

### 适配器模式
- `VectorStoreAdapter` - 适配向量服务
- `FeatureServiceAdapter` - 适配 Feast（位于扩展包 `ext/feast/http`）
- `S3Client` - 适配 S3 兼容协议（AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等）

### 装饰器模式
- `FeatureCache` - 特征缓存装饰器
- `FeatureMonitor` - 特征监控装饰器
- `FallbackStrategy` - 降级策略装饰器

## 重要注意事项

1. **ID 类型**：所有 ID 使用 `string` 类型（用户 ID、物品 ID）
2. **必需字段**：
   - `Fanout.MergeStrategy` - 必需（如果为 nil，使用默认 `FirstMergeStrategy`）
   - `UserBasedCF.SimilarityCalculator` - 必需
   - `UserBasedCF.Config` - 必需
   - `ItemBasedCF.SimilarityCalculator` - 必需
   - `ItemBasedCF.Config` - 必需
3. **接口优先**：所有策略都通过接口实现，不使用字符串配置
4. **无硬编码**：所有默认值都从配置接口获取
5. **线程安全**：`NodeFactory` 使用 `sync.RWMutex` 保证线程安全
6. **扩展包设计**：核心包无外部依赖，具体实现位于扩展包中
   - Redis Store: `go get github.com/rushteam/reckit/ext/store/redis`
   - Feast HTTP/gRPC: `go get github.com/rushteam/reckit/ext/feast/http` 或 `/grpc`
   - Milvus Vector: `go get github.com/rushteam/reckit/ext/vector/milvus`
   - 用户按需引入，避免不必要的依赖
   - 也可以参考扩展包实现，自行实现对应接口
7. **领域层接口优先**：推荐使用领域层接口（如 `core.FeatureService`），而非基础设施层接口
   - Feast 应通过适配器适配为 `core.FeatureService` 使用
   - 领域层接口更通用，不绑定具体实现
7. **类型转换工具**：使用 `pkg/conv` 进行类型转换，避免手写 switch-case
   - `conv.ToFloat64`、`conv.ToInt`、`conv.ToString` - 支持多种类型自动转换
   - `conv.MapToFloat64` - map[string]any -> map[string]float64
   - `conv.SliceAnyToString` - []any -> []string（兼容 YAML/JSON）
   - `conv.ConfigGet[T]`、`conv.ConfigGetInt64` - 从配置 map 读取值
8. **UserProfile 扩展属性**：通过 `Extras map[string]any` 存储自定义属性
   - `GetExtraFloat64`、`GetExtraInt`、`GetExtraString` - 带类型转换的获取方法
   - `core.GetExtraAs[T]` - 泛型方法，用于精确类型匹配（不进行数值转换）
9. **模型服务协议**：在需要约定时尽量采用 [KServe v2（Open Inference Protocol）](https://kserve.github.io/website/master/modelserving/data_plane/v2_protocol/) 标准；详见 `python/service/KSERVE_V2_ALIGNMENT.md`。

## 常用操作

### 创建 Item

```go
item := core.NewItem("item_123")
item.Score = 0.85
item.Features = map[string]float64{"ctr": 0.15, "cvr": 0.08}
item.PutLabel("recall_source", utils.Label{Value: "hot", Source: "recall"})
```

### 创建 Context

```go
rctx := &core.RecommendContext{
    UserID: "user_456",
    Scene:  "feed",
    User: &core.UserProfile{
        UserID:    "user_456",
        Age:       25,
        Gender:    "male",
        Interests: map[string]float64{"tech": 0.8, "game": 0.6},
    },
    UserProfile: map[string]any{
        "age":    25.0,
        "gender": "male",
    },
}
```

### 使用 Label DSL

```go
eval := dsl.NewEval(item, rctx)
result, _ := eval.Evaluate(`label.recall_source == "hot"`)
result, _ := eval.Evaluate(`item.score > 0.7`)
result, _ := eval.Evaluate(`label.recall_source.contains("ann")`)
```

### 使用类型转换工具（pkg/conv）

```go
import "github.com/rushteam/reckit/pkg/conv"

// 类型转换（支持多种数值类型自动转换）
v, _ := conv.ToFloat64(anyVal)  // 支持 float64/float32/int/int64/int32/bool
n, _ := conv.ToInt(anyVal)      // 支持 int/int64/int32/float64/float32
s, _ := conv.ToString(anyVal)

// Map/Slice 转换
weights := conv.MapToFloat64(configMap)  // map[string]any -> map[string]float64
ids := conv.SliceAnyToString(yamlSlice)  // []any -> []string（兼容 YAML/JSON）

// 配置读取（泛型）
bias := conv.ConfigGet[float64](config, "bias", 0.0)
timeout := conv.ConfigGetInt64(config, "timeout", 5)  // 兼容 int/float64
labelKey := conv.ConfigGet[string](config, "label_key", "category")

// 泛型类型断言
t, ok := conv.TypeAssert[MyType](v)
```

### 使用 UserProfile 扩展属性

```go
// 设置扩展属性
userProfile.SetExtra("vip_level", 3)
userProfile.SetExtra("preferred_price_range", "100-500")
userProfile.SetExtra("custom_tags", []string{"tech", "gaming"})

// 获取扩展属性（类型转换）
vipLevel, _ := userProfile.GetExtraFloat64("vip_level")
priceRange, _ := userProfile.GetExtraString("preferred_price_range")
purchaseCount, _ := userProfile.GetExtraInt("purchase_history_count")

// 获取扩展属性（泛型，精确类型匹配）
tags, _ := core.GetExtraAs[[]string](userProfile, "custom_tags")
// 注意：GetExtraAs 仅做类型断言，不进行数值转换；数值转换请使用 GetExtraFloat64/GetExtraInt
```

### 使用扩展包（Redis、Feast、Milvus）

核心包无外部依赖，具体实现位于扩展包中，需要单独引入：

#### Redis Store

```go
import (
    "github.com/rushteam/reckit/core"
    redisstore "github.com/rushteam/reckit/ext/store/redis"
)

// 安装：go get github.com/rushteam/reckit/ext/store/redis
store, err := redisstore.NewRedisStore("localhost:6379", 0)
if err != nil {
    log.Fatal(err)
}
defer store.Close(ctx)

// 作为 core.Store 使用
var s core.Store = store
```

#### Feast 特征服务（通过适配器）

Feast 是特征存储工具，应通过适配器适配为 `core.FeatureService` 领域接口使用。

```go
import (
    "github.com/rushteam/reckit/feature"
    feasthttp "github.com/rushteam/reckit/ext/feast/http"
    feastgrpc "github.com/rushteam/reckit/ext/feast/grpc"
)

// 方式 1：使用 HTTP 客户端
// 安装：go get github.com/rushteam/reckit/ext/feast/http
feastClient, _ := feasthttp.NewClient("http://localhost:6566", "my_project")
mapping := &feasthttp.FeatureMapping{
    UserFeatures: []string{"user_stats:age", "user_stats:gender"},
    ItemFeatures: []string{"item_stats:price", "item_stats:category"},
}
featureService := feasthttp.NewFeatureServiceAdapter(feastClient, mapping)

// 方式 2：使用 gRPC 客户端
// 安装：go get github.com/rushteam/reckit/ext/feast/grpc
feastClient, _ := feastgrpc.NewGrpcClient("localhost", 6565, "my_project")
featureService := feasthttp.NewFeatureServiceAdapter(feastClient, mapping)

// 作为 core.FeatureService 使用（领域层接口）
var fs core.FeatureService = featureService
```

**或自行实现**：参考扩展包实现，自行实现 `core.FeatureService` 接口。

#### Milvus 向量数据库

```go
import (
    "github.com/rushteam/reckit/core"
    milvus "github.com/rushteam/reckit/ext/vector/milvus"
)

// 安装：go get github.com/rushteam/reckit/ext/vector/milvus
milvusService := milvus.NewMilvusService("localhost:19530")

// 作为 core.VectorService 使用（召回场景）
var vectorService core.VectorService = milvusService

// 作为 core.VectorDatabaseService 使用（数据管理场景）
var dbService core.VectorDatabaseService = milvusService
```

**或自行实现**：参考扩展包实现，自行实现 `core.VectorService` 或 `core.VectorDatabaseService` 接口。

### 使用 Word2Vec / Item2Vec 模型

```go
import "github.com/rushteam/reckit/model"
import "github.com/rushteam/reckit/recall"

// 1. 创建 Word2Vec 模型（从 JSON 或 map；JSON 可由 python/train/train_item2vec.py 导出）
raw, _ := json.Unmarshal(...) // 或内联 map
w2vModel, _ := model.LoadWord2VecFromMap(raw)

// 2. 文本向量化（Word2Vec）
vector := w2vModel.EncodeText("electronics smartphone tech")

// 3. 序列向量化（Item2Vec：用户行为序列）
userVector := w2vModel.EncodeSequence([]string{"item_1", "item_2", "item_3"})

// 4. 召回：文本模式 vs Item2Vec 序列模式
word2vecRecall := &recall.Word2VecRecall{
    Model: w2vModel, Store: store, TopK: 20,
    Mode: "text", TextField: "title",
}
item2vecRecall := &recall.Word2VecRecall{
    Model: w2vModel, Store: store, TopK: 20,
    Mode: "sequence", // Item2Vec
}
```

**Python 训练**: `python/train/train_item2vec.py --mode item2vec|word2vec`，输出 JSON。详见 `docs/WORD2VEC_ITEM2VEC.md`。

### 使用 Diversity（多样性重排）

```go
import "github.com/rushteam/reckit/rerank"

// 1. 按类别去重（保留每个类别的第一个 item）
categoryDedup := &rerank.Diversity{
    LabelKey: "category", // 从 Labels["category"] / Meta["category"] / Features["category"] 获取
}

// 2. 按作者打散（避免同一作者连续出现）
authorDiversity := &rerank.Diversity{
    AuthorKey:      "author",     // 字段 key
    MaxConsecutive: 1,            // 同一作者最多连续出现 1 次
    WindowSize:     3,            // 滑动窗口大小
}

// 3. 同时使用（先类别去重，再作者打散）
combined := &rerank.Diversity{
    LabelKey:       "category",
    AuthorKey:      "author",
    MaxConsecutive: 2,
}

// 字段查找优先级：Labels > Meta > Features
// - Labels["category"].Value = "tech"     → 使用 "tech"
// - Meta["category"] = "tech" (string)    → 使用 "tech"（Labels 未找到时）
// - Features["category_id"] = 3.0         → 使用 "3"（Labels、Meta 均未找到时）
```

### 使用 DeepFM 模型（PyTorch）

```go
import "github.com/rushteam/reckit/model"
import "github.com/rushteam/reckit/rank"
import "github.com/rushteam/reckit/service"

// 1. 训练 DeepFM 模型（Python）
// cd python
// python train/train_deepfm.py --version v1.0.0

// 2. 启动 DeepFM 服务（Python，需兼容 KServe V2 协议）
// uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080

// 3. 在 Golang 中使用（推荐：通过 KServe V2 协议）
kserveClient := service.NewKServeClient("http://localhost:8080", "deepfm",
    service.WithKServeTimeout(5*time.Second),
)
deepfmModel := model.NewRPCModelFromService("deepfm", kserveClient)

p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        &feature.EnrichNode{...},
        &rank.RPCNode{Model: deepfmModel},
    },
}
```

**Python 训练**: `python/train/train_deepfm.py`，详见 `python/train/DEEPFM_README.md`。

### KServe V2 特征字典模式（BYTES）

默认情况下 `KServeClient` 将 `Features (map[string]float64)` 排序展平为 FP64 tensor 发送。
当服务端需要自行编码特征（如 DeepFM 包含 embedding lookup）时，可启用 `WithKServeV2FeaturesAsJSON`，
将每条特征序列化为 JSON 字符串，以 BYTES 类型发送：

```go
import "github.com/rushteam/reckit/service"

// 启用特征字典模式
kserveClient := service.NewKServeClient("http://localhost:8080", "deepfm",
    service.WithKServeTimeout(5*time.Second),
    service.WithKServeV2FeaturesAsJSON(), // Features 以 JSON BYTES 发送
)

// 发送的 V2 请求体示例：
// {
//   "inputs": [{
//     "name": "features",
//     "shape": [2],
//     "datatype": "BYTES",
//     "data": [
//       "{\"ctr\":0.15,\"cvr\":0.08,\"user_age\":25}",
//       "{\"ctr\":0.22,\"cvr\":0.12,\"user_age\":30}"
//     ]
//   }]
// }
//
// 服务端可直接 json.loads(data[i]) 得到特征字典，无需约定特征顺序。
```

两种模式对比：
- **默认（FP64 tensor）**：客户端排序展平，适合服务端直接接收数值向量的场景
- **`WithKServeV2FeaturesAsJSON`（BYTES）**：客户端保留原始特征字典，适合服务端自行做特征编码（embedding lookup、分箱等）的场景

**标准参考**：
- [KServe V2 Protocol (Open Inference Protocol)](https://kserve.github.io/website/master/modelserving/data_plane/v2_protocol/)
- [Open Inference Protocol REST 规范](https://github.com/kserve/open-inference-protocol/blob/main/specification/protocol/inference_rest.md)
- [Binary Tensor Data Extension（BYTES 等数据类型定义）](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol/binary-tensor-data-extension)

### 使用 BERT 模型

```go
import "github.com/rushteam/reckit/model"
import "github.com/rushteam/reckit/recall"
import "github.com/rushteam/reckit/service"

// 1. 创建 BERT 服务客户端（推荐 KServe V2 / Triton）
kserveClient := service.NewKServeClient(
    "http://localhost:8080", // Triton / KServe 端点
    "bert-base",              // 模型名称
    service.WithKServeTimeout(5*time.Second),
    service.WithKServeV2OutputName("embeddings"),
)

// 2. 创建 BERT 模型
bertModel := model.NewBERTModel(kserveClient, 768).
    WithModelName("bert-base").
    WithMaxLength(512).
    WithPoolingStrategy("cls")

// 3. 文本编码
vector, _ := bertModel.EncodeText(ctx, "electronics smartphone tech")

// 4. 批量编码（提高效率）
vectors, _ := bertModel.EncodeTexts(ctx, []string{"text1", "text2", "text3"})

// 5. 基于 BERT 的召回
bertRecall := &recall.BERTRecall{
    Model:     bertModel,
    Store:     bertStore,
    TopK:      20,
    Mode:      "text",      // text 或 query
    TextField: "title",     // title / description / tags
    BatchSize: 32,          // 批量编码大小
}
```

### MMoE / YouTube DNN / DSSM / GraphRecall

- **MMoE**：多目标重排（CTR + 时长 + GMV）。`python train/train_mmoe.py` → `uvicorn service.mmoe_server:app --port 8081`；Golang `rerank.MMoENode{ Endpoint: "http://localhost:8081/predict", WeightCTR, WeightWatchTime, WeightGMV }`。
- **YouTube DNN**：视频/内容流召回。`python train/train_youtube_dnn.py` → `uvicorn service.youtube_dnn_server:app --port 8082`；Golang `recall.YouTubeDNNRecall{ UserEmbeddingURL, VectorService, TopK, Collection }`。
- **DSSM**：Query-Doc 语义召回。`python train/train_dssm.py` → `uvicorn service.dssm_server:app --port 8083`；Golang `recall.DSSMRecall{ QueryEmbeddingURL, VectorService, TopK, Collection }`；query 特征来自 `rctx.Params["query_features"]`。
- **GraphRecall**：Node2vec 社交/关注页召回。`python train/train_node2vec.py` → `uvicorn service.graph_recall_server:app --port 8084`；Golang `recall.GraphRecall{ Endpoint: "http://localhost:8084/recall", TopK }`。

### 加载特征元数据

```go
import "github.com/rushteam/reckit/feature"

// 方式 1：本地文件加载
fileLoader := feature.NewFileMetadataLoader()
meta, _ := fileLoader.Load(ctx, "python/model/feature_meta.json")

// 方式 2：HTTP 接口加载
httpLoader := feature.NewHTTPMetadataLoader(5 * time.Second)
meta, _ := httpLoader.Load(ctx, "http://api.example.com/models/v1.0.0/feature_meta")

// 方式 3：S3 兼容协议加载（支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等）
s3Client := &MyS3Client{} // 需要实现 feature.S3Client 接口
s3Loader := feature.NewS3MetadataLoader(s3Client, "my-bucket")
meta, _ := s3Loader.Load(ctx, "models/v1.0.0/feature_meta.json")

// 使用特征元数据
validated := meta.ValidateFeatures(features)
missing := meta.GetMissingFeatures(features)
vector := meta.BuildFeatureVector(features)
```

### 加载特征标准化器

```go
// 方式 1：本地文件加载
fileScalerLoader := feature.NewFileScalerLoader()
scaler, _ := fileScalerLoader.Load(ctx, "python/model/feature_scaler.json")

// 方式 2：HTTP 接口加载
httpScalerLoader := feature.NewHTTPScalerLoader(5 * time.Second)
scaler, _ := httpScalerLoader.Load(ctx, "http://api.example.com/models/v1.0.0/feature_scaler")

// 方式 3：S3 兼容协议加载
s3ScalerLoader := feature.NewS3ScalerLoader(s3Client, "my-bucket")
scaler, _ := s3ScalerLoader.Load(ctx, "models/v1.0.0/feature_scaler.json")

// 使用标准化器
normalized := scaler.Normalize(features)
```

## 示例代码位置

- `examples/basic/` - 基础示例
- `examples/all_recall_algorithms/` - 所有召回算法示例
- `examples/extensibility/` - 扩展性示例（自定义策略、Hook）
- `examples/config/` - 配置化 Pipeline 示例（使用 `pkg/conv` 进行配置解析）
- `examples/user_profile/` - 用户画像示例（包含扩展属性 Extras 的使用）
- `examples/feature_service/` - 特征服务示例
- `examples/personalization/` - 个性化推荐示例
- `examples/feature_metadata/` - 特征元数据使用示例
- `examples/feature_metadata_loader/` - 特征元数据加载器示例（本地文件、HTTP、S3 兼容协议）
- `examples/feature_processing/` - 特征处理工具类示例
- `examples/feature_version/` - 特征版本管理示例
- `examples/word2vec/` - Word2Vec / Item2Vec 示例（含 JSON 加载、Python 训练说明）
- `examples/deepfm/` - DeepFM 排序示例（PyTorch 训练 + RPC 调用）
- `examples/bert/` - BERT 模型使用示例

## 相关文档

- `README.md` - 项目主文档
- `docs/ARCHITECTURE.md` - 架构设计文档
- `docs/RECALL_ALGORITHMS.md` - 召回算法文档
- `docs/WORD2VEC_ITEM2VEC.md` - Word2Vec / Item2Vec 使用与 Python 训练
- `docs/RANK_MODELS.md` - 排序模型文档
- `docs/EXTENSIBILITY_ANALYSIS.md` - 可扩展性分析
- `docs/INTERFACES_AND_IMPLEMENTATIONS.md` - 接口与实现完整分析
- `docs/FEATURE_CONSISTENCY.md` - 特征一致性文档（训练与在线一致性）
- `docs/FEATURE_PROCESSING.md` - 特征处理文档（归一化、编码等）
- `docs/ENCODER_INTERFACE_DESIGN.md` - 编码器接口设计说明
- `docs/USER_PROFILE.md` - 用户画像文档（包含扩展属性 Extras 的使用）
- `docs/MODEL_SERVICE_PROTOCOL.md` - 模型服务协议约束（推荐 KServe V2 / Open Inference Protocol）
- `pkg/conv/README.md` - 类型转换与泛型工具文档
- `ext/README.md` - 扩展包使用指南
