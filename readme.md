# Reckit（推荐系统工具包）

目标：用 **Pipeline + Node** 的方式快速拼装推荐系统（内容流 / 电商 / 广告），并让 **Labels** 成为一等公民（全链路透传、可解释、可观测）。

## 架构图

```
                ┌──────────┐
Request ──────▶ │  Context │  用户画像 / 实时特征
                └────┬─────┘
                     │
        ┌────────────▼─────────────┐
        │          Recall           │  多路召回（并发）
        └────┬─────────┬───────────┘
             │         │
     CF召回   │   热门召回 │   ANN召回 …
             ▼         ▼
        ┌──────────────────────────┐
        │          Filter           │  规则 / 黑白名单
        └─────────────┬────────────┘
                      │
        ┌─────────────▼────────────┐
        │           Rank            │  ML / LR / GBDT / RPC
        └─────────────┬────────────┘
                      │
        ┌─────────────▼────────────┐
        │          ReRank           │  多样性 / 去重 / 策略
        └─────────────┬────────────┘
                      │
        ┌─────────────▼────────────┐
        │       PostProcess         │  截断 / 打散 / 业务规则
        └──────────────────────────┘
```

## 核心思想

| 模块       | 设计模式                                   |
| -------- | -------------------------------------- |
| Pipeline | **Pipeline / Chain of Responsibility** |
| Recall   | **Strategy + Fan-out 并发模式**            |
| Rank     | **Strategy / Template Method**         |
| 特征注入     | **Context Object**                     |
| 可扩展      | **Plugin-like 接口设计**                   |
| 并发       | **errgroup / goroutine pool**          |
| 存储抽象     | **Store 接口（Redis / MySQL / ES）**      |
| 配置化      | **YAML/JSON 配置 + Factory 模式**         |
| 策略驱动     | **Label DSL 表达式引擎**                   |

## 目录结构

```
reckit/
├── core/          # Item + RecommendContext（核心数据结构）
├── pipeline/      # Node 接口 + Pipeline 执行器 + 配置加载
├── recall/        # Recall Source + Fanout Node + ANN 召回
├── filter/        # Filter Node（黑名单、用户拉黑、已曝光等）
├── rank/          # Rank Node（LR / RPC）
├── rerank/        # ReRank Node（多样性等）
├── model/         # RankModel 抽象 + LR / RPC 实现
├── store/         # Store 抽象（Redis / Memory）
├── config/        # Pipeline 配置工厂
├── pkg/
│   ├── utils/     # Label + Merge 规则
│   └── dsl/       # Label DSL 表达式引擎
├── python/            # Python ML 训练与服务
│   ├── data/          # 训练数据
│   ├── train/         # 训练脚本
│   ├── service/       # HTTP 推理服务
│   └── model/         # 训练好的模型
└── examples/
    ├── basic/          # 基础示例
    ├── config/         # 配置化 Pipeline 示例（含配置文件）
    ├── dsl/            # DSL 表达式示例
    ├── personalization/ # 千人千面个性化推荐示例
    └── rpc_xgb/        # Python XGBoost 模型调用示例
```

## 快速开始

### 基础示例

```bash
go run ./examples/basic
```

### 配置化 Pipeline

```bash
go run ./examples/config
```

### DSL 表达式测试

```bash
go run ./examples/dsl
```

### Python XGBoost 模型调用

```bash
# 1. 训练模型（Python）
cd python
pip install -r requirements.txt
python train/train_xgb.py

# 2. 启动服务（Python）
uvicorn service.server:app --host 0.0.0.0 --port 8080

# 3. 运行 Go 示例（新终端）
go run ./examples/rpc_xgb
```

## 功能特性概览

### 核心模块

- **Pipeline + Node 架构** (`pipeline/`): Pipeline 执行器，支持链式 Node 处理，Node 接口统一所有处理单元，支持配置化加载（YAML/JSON）
- **Labels-first 设计** (`core/item.go`, `pkg/utils/label.go`): Labels 全链路透传，自动 Merge 规则，支持策略驱动和可解释性
- **Store 抽象层** (`store/`): 统一的存储接口，支持 Redis、Memory 实现，扩展接口 KeyValueStore（支持有序集合、Hash）

### 召回模块（Recall）

- **多路 Recall 并行 + 合并** (`recall/fanout.go`): 并发执行多个召回源（errgroup），超时控制、限流，合并策略：first / union / priority，自动写入 `recall_source` 和 `recall_priority` labels
- **热门召回** (`recall/hot.go`): 支持从 Store 读取（ZRange 或 JSON），支持内存 fallback，同时实现 Source 和 Node 接口
- **Embedding ANN Recall** (`recall/ann.go`): 向量检索召回，支持余弦相似度和欧氏距离，可配置 TopK，支持从 RecommendContext 获取用户向量实现个性化召回，需要实现 VectorStore 接口
- **用户历史召回** (`recall/user_history.go`): 基于用户历史行为的个性化召回，支持浏览、点击、购买等行为类型，支持时间窗口过滤

### 过滤模块（Filter）

- **过滤模块** (`filter/`): 支持多种过滤策略（黑名单、用户拉黑、已曝光），可组合多个过滤器，支持内存列表和 Store 两种数据源，自动记录过滤原因到 Label
  - `BlacklistFilter`: 黑名单过滤
  - `UserBlockFilter`: 用户拉黑过滤
  - `ExposedFilter`: 已曝光过滤（支持时间窗口）

### 排序模块（Rank）

- **LR 模型排序** (`model/lr.go`, `rank/lr_node.go`): 线性回归模型，可配置权重和偏置，自动写入 `rank_model` label
- **RPC 模型排序** (`model/rpc.go`, `rank/rpc_node.go`): 通过 HTTP/RPC 调用外部模型服务，支持 GBDT、XGBoost、TensorFlow Serving、TorchServe 等，统一使用 RPCModel（通过 name 参数区分），支持超时配置，自动写入 `rank_model` 和 `rank_type` labels
  - 请求格式：`{"features": {"ctr": 0.15, "cvr": 0.08}}`
  - 响应格式：`{"score": 0.85}` 或 `{"prediction": 0.85}`

### 重排模块（ReRank）

- **多样性重排** (`rerank/diversity.go`): 按类别去重（保留首个出现的类别），支持从 Label 或 Meta 读取类别，可配置 label_key

### 特征工程模块（Feature）

- **特征注入** (`feature/enrich.go`): 将用户特征、物品特征、交叉特征组合，支持千人千面个性化推荐，自动添加特征前缀（user_*, item_*, cross_*）

### 配置化

- **配置化 Pipeline** (`pipeline/config.go`, `config/factory.go`): 支持从 YAML/JSON 文件加载配置，NodeFactory 模式构建 Node，支持所有内置 Node 类型的配置化

### 工具模块

- **Label DSL 解释器** (`pkg/dsl/eval.go`): 基于 [CEL (Common Expression Language)](https://github.com/google/cel-go) 实现，支持类型安全、高性能的表达式求值，支持比较、逻辑、字符串、存在性检查等标准 CEL 语法

## 千人千面支持

Reckit 支持千人千面的个性化推荐，通过以下机制实现：

### 1. 用户上下文（RecommendContext）

`RecommendContext` 承载用户信息，贯穿整个 Pipeline：

```go
rctx := &core.RecommendContext{
    UserID: 42,
    Scene:  "feed",
    UserProfile: map[string]any{
        "age": 25,
        "gender": "male",
        "user_vector": []float64{0.1, 0.2, 0.3}, // 用户 embedding
    },
    Realtime: map[string]any{
        "hour": time.Now().Hour(),
        "device": "mobile",
    },
}
```

### 2. 特征注入（Feature Enrichment）

特征注入节点将用户特征、物品特征、交叉特征组合：

```go
enrichNode := &feature.EnrichNode{
    UserFeaturePrefix:  "user_",
    ItemFeaturePrefix:  "item_",
    CrossFeaturePrefix: "cross_",
}
```

**特征组合规则**：
- 用户特征：从 `rctx.UserProfile` 提取，添加 `user_` 前缀
- 物品特征：从 `item.Features` 提取，添加 `item_` 前缀
- 交叉特征：自动生成用户-物品交叉特征（如 `user_age * item_price`），添加 `cross_` 前缀

### 3. 个性化召回

- **ANN 召回**：支持从 `rctx.UserProfile["user_vector"]` 获取用户向量
- **用户历史召回**：基于用户历史行为（浏览、点击、购买）进行个性化召回

### 4. 个性化过滤

- **用户拉黑过滤**：过滤用户拉黑的物品
- **已曝光过滤**：过滤用户已看过的物品

### 5. 个性化排序

排序模型可以使用注入后的特征（包含用户特征、物品特征、交叉特征）进行个性化排序。

## 功能详解

### 1. Pipeline + Node 架构

所有推荐逻辑通过 Pipeline Node 串联，每个 Node 处理 Items、Score、Labels：

```go
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        &rank.LRNode{...},
        &rerank.Diversity{...},
    },
}

items, err := p.Run(ctx, rctx, nil)
```

### 2. Labels-first 设计

Labels 自动透传，节点之间不丢失，支持 Merge 和策略驱动：

```go
item.PutLabel("recall_source", utils.Label{Value: "hot", Source: "recall"})
item.PutLabel("rank_model", utils.Label{Value: "lr", Source: "rank"})
```

### 3. Store 抽象层

统一的存储接口，支持 Redis、MySQL、ES、内存等：

```go
// 内存 Store（测试用）
memStore := store.NewMemoryStore()

// Redis Store（生产用）
redisStore, _ := store.NewRedisStore("localhost:6379", 0)

// 使用 KeyValueStore 扩展功能
if kvStore, ok := redisStore.(store.KeyValueStore); ok {
    kvStore.ZAdd(ctx, "hot:feed", 100.0, "1")
    members, _ := kvStore.ZRange(ctx, "hot:feed", 0, 9) // Top 10
}
```

### 1. 多路 Recall 并行 + 合并

并发执行多个召回源，支持超时、限流、多种合并策略：

```go
fanout := &recall.Fanout{
    Sources: []recall.Source{
        &recall.Hot{IDs: []string{"1", "2", "3"}},
        &recall.ANN{...},
    },
    Dedup:         true,
    Timeout:       2 * time.Second,  // 每个召回源超时
    MaxConcurrent: 5,                // 最大并发数
    MergeStrategy: "priority",        // first / union / priority
}
```

**合并策略**：
- `first`: 按 ID 去重，保留第一个出现的（默认）
- `union`: 合并所有结果，不去重
- `priority`: 按优先级合并（Sources 索引越小优先级越高）

### 2. Embedding ANN Recall

向量检索召回，支持余弦相似度和欧氏距离：

```go
ann := &recall.ANN{
    Store:      vectorStore,              // 实现 VectorStore 接口
    UserVector: []float64{0.1, 0.2, 0.3}, // 用户向量
    TopK:       20,                       // 返回 TopK
    Metric:     "cosine",                 // cosine / euclidean
}
```

**VectorStore 接口**：
```go
type VectorStore interface {
    GetVector(ctx context.Context, itemID int64) ([]float64, error)
    ListVectors(ctx context.Context) (map[int64][]float64, error)
}
```

### 3. RPC Rank（支持 GBDT / XGBoost / TensorFlow Serving 等）

通过 HTTP/RPC 调用外部模型服务，统一使用 RPCModel。支持 Python 训练的 XGBoost 模型：

```go
// XGBoost 模型（Python 服务）
xgbModel := model.NewRPCModel("xgboost", "http://localhost:8080/predict", 5*time.Second)
rpcNode := &rank.RPCNode{Model: xgbModel}

// GBDT 模型
gbdtModel := model.NewRPCModel("gbdt", "http://localhost:8080/predict", 5*time.Second)
rpcNode := &rank.RPCNode{Model: gbdtModel}

// 通用 RPC 模型（TensorFlow Serving、TorchServe 等）
rpcModel := model.NewRPCModel("custom", "http://localhost:8080/predict", 5*time.Second)
```

**Python 模型服务**：
- 使用 `python/train/train_xgb.py` 训练 XGBoost 模型
- 使用 `python/service/server.py` 启动 HTTP 推理服务
- Go 端通过 `RPCModel` 自动调用，特征名自动对齐（去掉前缀）

**请求格式**：
```json
{
  "features": {
    "ctr": 0.15,
    "cvr": 0.08
  }
}
```

**响应格式**：
```json
{
  "score": 0.85
}
```
或
```json
{
  "prediction": 0.85
}
```

### 4. 特征注入（Feature Enrichment）

特征注入节点将用户特征、物品特征、交叉特征组合，支持千人千面：

```go
enrichNode := &feature.EnrichNode{
    UserFeaturePrefix:  "user_",
    ItemFeaturePrefix:  "item_",
    CrossFeaturePrefix: "cross_",
}
```

**使用场景**：
- 在排序前注入特征，让模型使用用户特征进行个性化排序
- 自动生成交叉特征（用户特征 × 物品特征）
- 支持自定义特征提取器

**特征示例**：
```go
// 注入后的 item.Features 包含：
// - user_age: 25.0
// - user_gender: 1.0
// - item_price: 99.0
// - item_category: 3.0
// - cross_age_x_price: 2475.0  // 自动生成的交叉特征
```

### 5. 过滤模块（Filter）

支持多种过滤策略，包括黑名单、用户拉黑、已曝光等：

```go
filterNode := &filter.FilterNode{
    Filters: []filter.Filter{
        // 黑名单过滤
        filter.NewBlacklistFilter(
            []int64{100, 200, 300}, // 内存中的黑名单
            nil,                    // Store 适配器（可选）
            "blacklist:items",      // Store key（可选）
        ),
        // 用户拉黑过滤
        filter.NewUserBlockFilter(
            storeAdapter,           // Store 适配器
            "user:block",          // key 前缀
        ),
        // 已曝光过滤
        filter.NewExposedFilter(
            storeAdapter,          // Store 适配器
            "user:exposed",        // key 前缀
            7*24*3600,             // 时间窗口（7天，秒）
        ),
    },
}
```

**过滤器特性**：
- **BlacklistFilter**：过滤黑名单中的物品，支持内存列表和 Store
- **UserBlockFilter**：过滤用户拉黑的物品，从 Store 读取用户拉黑列表
- **ExposedFilter**：过滤已曝光的物品，支持时间窗口过滤

**Store 适配器**：
```go
storeAdapter := filter.NewStoreAdapter(memStore)
```

### 6. 配置化 Pipeline（YAML / JSON）

从配置文件加载 Pipeline，无需修改代码：

**配置文件** (`examples/config/pipeline.example.yaml`):
```yaml
pipeline:
  name: "demo_recommendation"
  nodes:
    - type: "recall.fanout"
      config:
        dedup: true
        timeout: 2
        max_concurrent: 5
        merge_strategy: "priority"
        sources:
          - type: "hot"
            ids: [1, 2, 3, 4, 5]
    
    - type: "rank.lr"
      config:
        bias: 0.0
        weights:
          ctr: 1.2
          cvr: 0.8
    
    - type: "rerank.diversity"
      config:
        label_key: "category"
```

**使用**：
```go
// 从 YAML 加载
cfg, err := pipeline.LoadFromYAML("examples/config/pipeline.example.yaml")

// 从 JSON 加载
cfg, err := pipeline.LoadFromJSON("examples/config/pipeline.json")

// 构建 Pipeline
factory := config.DefaultFactory()
p, err := cfg.BuildPipeline(factory)

// 运行
items, err := p.Run(ctx, rctx, nil)
```

### 7. Label DSL 解释器（基于 CEL）

使用 [CEL (Common Expression Language)](https://github.com/google/cel-go) 实现的表达式引擎，具有类型安全、高性能、线程安全等特性。

```go
eval := dsl.NewEval(item, rctx)

// 字符串比较
result, _ := eval.Evaluate(`label.recall_source == "hot"`)

// 数值比较
result, _ := eval.Evaluate(`item.score > 0.7`)

// 字符串包含
result, _ := eval.Evaluate(`label.recall_source.contains("hot")`)

// 逻辑组合
result, _ := eval.Evaluate(`label.category == "A" && item.score > 0.8`)

// 存在性检查
result, _ := eval.Evaluate(`label.recall_source != null`)
result, _ := eval.Evaluate(`"nonexist" in label`)

// 复杂表达式
result, _ := eval.Evaluate(`label.recall_source.contains("ann") || label.recall_source.contains("cf")`)
```

**支持的语法（CEL 标准语法）**：
- 比较：`==`, `!=`, `>`, `<`, `>=`, `<=`
- 逻辑：`&&`, `||`, `!`
- 字符串：`.contains()`, `in` 运算符
- 存在性：`!= null`, `"key" in label`

## 完整示例

### 代码方式构建 Pipeline

```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

// 初始化 Store
memStore := store.NewMemoryStore()
defer memStore.Close()

// 构建 Pipeline
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{
            Sources: []recall.Source{
                &recall.Hot{
                    Store: memStore,
                    Key:   "hot:feed",
                },
            },
            Dedup:         true,
            Timeout:       2 * time.Second,
            MaxConcurrent: 5,
        },
        &filter.FilterNode{
            Filters: []filter.Filter{
                filter.NewBlacklistFilter([]int64{100, 200}, nil, ""),
            },
        },
        // 特征注入（千人千面）
        &feature.EnrichNode{
            UserFeaturePrefix:  "user_",
            ItemFeaturePrefix:  "item_",
            CrossFeaturePrefix: "cross_",
        },
        &rank.LRNode{
            Model: &model.LRModel{
                Bias: 0,
                Weights: map[string]float64{
                    "ctr": 1.2,
                    "cvr": 0.8,
                },
            },
        },
        &rerank.Diversity{LabelKey: "category"},
    },
}

rctx := &core.RecommendContext{
    UserID: 42,
    Scene:  "feed",
}

items, err := p.Run(ctx, rctx, nil)
```

### 配置方式构建 Pipeline

```go
// 加载配置
cfg, _ := pipeline.LoadFromYAML("examples/config/pipeline.example.yaml")

// 构建 Pipeline
factory := config.DefaultFactory()
p, _ := cfg.BuildPipeline(factory)

// 运行
items, _ := p.Run(ctx, rctx, nil)
```

## 扩展指南

### 添加新的 Recall Source

1. 实现 `recall.Source` 接口：
```go
type MyRecall struct{}

func (r *MyRecall) Name() string { return "my_recall" }
func (r *MyRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
    // 实现召回逻辑
}
```

2. 在 `config/factory.go` 中注册构建器：
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

2. 在 `config/factory.go` 中注册构建器：
```go
factory.Register("filter", buildFilterNode) // 在 buildFilterNode 中添加新类型
```

### 添加新的 Rank Model

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

3. 在 `config/factory.go` 中注册构建器

### 添加新的 Store 实现

实现 `store.Store` 或 `store.KeyValueStore` 接口：

```go
type MyStore struct{}

func (s *MyStore) Name() string { return "my_store" }
func (s *MyStore) Get(ctx context.Context, key string) ([]byte, error) {
    // 实现 Get
}
// ... 实现其他方法
```

### 扩展 DSL 表达式

在 `pkg/dsl/eval.go` 中添加新的函数或运算符支持。

## 依赖

### Go 依赖

```go
require (
    github.com/google/cel-go v0.26.1
    github.com/redis/go-redis/v9 v9.5.1
    golang.org/x/sync v0.19.0
    gopkg.in/yaml.v3 v3.0.1
)
```

### Python 依赖

见 `python/requirements.txt`：
- xgboost
- fastapi
- uvicorn
- pandas
- numpy
- scikit-learn

## 许可证

MIT
