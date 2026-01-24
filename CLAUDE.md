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

### 核心接口

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

// 过滤器接口
type Filter interface {
    Name() string
    ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error)
}

// 存储接口
type Store interface {
    Get(ctx context.Context, key string) ([]byte, error)
    Set(ctx context.Context, key string, value []byte, ttl int64) error
    // ...
}
```

### 策略接口（可扩展）

```go
// 合并策略（recall/fanout.go）
type MergeStrategy interface {
    Merge(items []*core.Item, dedup bool) []*core.Item
}
// 内置实现：FirstMergeStrategy, UnionMergeStrategy, PriorityMergeStrategy

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
    
    User        *UserProfile        // 强类型用户画像（推荐使用）
    UserProfile map[string]any     // 向后兼容的 map 形式
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
├── store/             # 存储抽象（Memory, Redis）
├── vector/             # 向量服务（Milvus）
├── service/           # ML 服务（TF Serving, ANN Service）
├── feast/             # Feast 集成
├── config/            # Pipeline 配置工厂
└── pkg/
    ├── utils/         # Label 工具
    ├── dsl/           # Label DSL 表达式引擎
    └── conv/          # 类型转换与泛型工具（ToFloat64、ConfigGet、MapToFloat64 等）
```

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
- `recall/fanout.go` - 多路并发召回和合并策略
- `recall/collaborative_filtering.go` - U2I/I2I 协同过滤
- `recall/ann.go` - Embedding ANN 召回
- `recall/content.go` - 内容推荐
- `recall/matrix_factorization.go` - 矩阵分解召回
- `recall/word2vec_recall.go` - Word2Vec 召回（基于文本/序列相似度）
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

### 特征模块

- `feature/service.go` - FeatureService 接口和实现
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
            FeatureService: featureService,
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

### 3. 自定义策略

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

### 4. 动态注册 Node

```go
factory := pipeline.NewNodeFactory()
factory.Register("my.custom.node", func(config map[string]interface{}) (pipeline.Node, error) {
    return &MyCustomNode{}, nil
})
```

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

实现 `filter.Filter` 接口：

```go
type MyFilter struct{}

func (f *MyFilter) Name() string { return "filter.my" }
func (f *MyFilter) ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error) {
    // 返回 true 表示过滤，false 表示保留
    return false, nil
}
```

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
- `MergeStrategy` - 合并策略
- `ErrorHandler` - 错误处理策略
- `SortStrategy` - 排序策略
- `SimilarityCalculator` - 相似度计算策略
- `RankModel` - 排序模型策略

### 工厂模式
- `NodeFactory` - Node 构建工厂（支持动态注册）
- `FeatureServiceFactory` - 特征服务工厂
- `MLServiceFactory` - ML 服务工厂

### 适配器模式
- `VectorStoreAdapter` - 适配向量服务
- `FeatureServiceAdapter` - 适配 Feast
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
6. **类型转换工具**：使用 `pkg/conv` 进行类型转换，避免手写 switch-case
   - `conv.ToFloat64`、`conv.ToInt`、`conv.ToString` - 支持多种类型自动转换
   - `conv.MapToFloat64` - map[string]any -> map[string]float64
   - `conv.SliceAnyToString` - []any -> []string（兼容 YAML/JSON）
   - `conv.ConfigGet[T]`、`conv.ConfigGetInt64` - 从配置 map 读取值
7. **UserProfile 扩展属性**：通过 `Extras map[string]any` 存储自定义属性
   - `GetExtraFloat64`、`GetExtraInt`、`GetExtraString` - 带类型转换的获取方法
   - `core.GetExtraAs[T]` - 泛型方法，用于精确类型匹配（不进行数值转换）

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

### 使用 Word2Vec 模型

```go
import "github.com/rushteam/reckit/model"
import "github.com/rushteam/reckit/recall"

// 1. 创建 Word2Vec 模型（从预训练的词向量）
wordVectors := map[string][]float64{
    "item_1": []float64{0.1, 0.2, 0.3},
    "item_2": []float64{0.4, 0.5, 0.6},
    // ...
}
w2vModel := model.NewWord2VecModel(wordVectors, 128)

// 2. 文本向量化
text := "electronics smartphone tech"
vector := w2vModel.EncodeText(text)

// 3. 序列向量化（用户行为序列）
sequence := []string{"item_1", "item_2", "item_3"}
userVector := w2vModel.EncodeSequence(sequence)

// 4. 基于 Word2Vec 的召回
word2vecRecall := &recall.Word2VecRecall{
    Model:    w2vModel,
    Store:    word2vecStore,
    TopK:     20,
    Mode:     "sequence", // 或 "text"
    TextField: "title",   // title / description / tags
}
```

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
- `examples/word2vec/` - Word2Vec 模型使用示例

## 相关文档

- `README.md` - 项目主文档
- `docs/ARCHITECTURE.md` - 架构设计文档
- `docs/RECALL_ALGORITHMS.md` - 召回算法文档
- `docs/RANK_MODELS.md` - 排序模型文档
- `docs/EXTENSIBILITY_ANALYSIS.md` - 可扩展性分析
- `docs/INTERFACES_AND_IMPLEMENTATIONS.md` - 接口与实现完整分析
- `docs/FEATURE_CONSISTENCY.md` - 特征一致性文档（训练与在线一致性）
- `docs/FEATURE_PROCESSING.md` - 特征处理文档（归一化、编码等）
- `docs/ENCODER_INTERFACE_DESIGN.md` - 编码器接口设计说明
- `docs/USER_PROFILE.md` - 用户画像文档（包含扩展属性 Extras 的使用）
- `pkg/conv/README.md` - 类型转换与泛型工具文档
