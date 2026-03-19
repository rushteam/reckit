# Reckit

<div align="center">

**工业级推荐系统工具包 | Production-Ready Recommender System Toolkit**

[![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8?style=flat-square&logo=go)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Go Report Card](https://goreportcard.com/badge/github.com/rushteam/reckit?style=flat-square)](https://goreportcard.com/report/github.com/rushteam/reckit)

</div>

---

## 📖 简介

Reckit 是一个**工业级推荐系统工具包**，采用 **Pipeline + Node** 架构设计，通过接口抽象实现高度可扩展性。支持内容流、电商、广告等多种推荐场景，帮助开发者快速构建生产级推荐系统。

### 🎯 工程目标

场景分工：
- **深度模型、复杂梯度更新** → 适用工具：**PyTorch/Tensorflow**
- **高并发、低延迟、统计计算** → 适用工具：**Golang(Reckit)**

本项目采用 **PyTorch/Tensorflow + Golang** 的分工模式：
- **PyTorch/Tensorflow**：负责深度模型的训练、复杂梯度更新等机器学习任务
- **Golang (Reckit)**：负责高并发推荐服务、低延迟在线推理、统计计算等生产环境任务

### ✨ 核心亮点

- 🎯 **Pipeline-first 架构**：链式 Node 处理，灵活组合推荐流程
- 🏷️ **Labels-first 设计**：全链路透传，支持可解释性和策略驱动
- 🔌 **完全可扩展**：所有策略通过接口实现，无需修改库代码
- ⚡ **高性能并发**：多路召回并发执行，超时控制和限流保护
- 🎨 **策略模式丰富**：合并、排序、错误处理等策略均可自定义
- 🔧 **配置化支持**：YAML/JSON 配置，Pipeline 可配置化加载
- 🚀 **开箱即用**：内置 7+ 召回算法、6+ 排序模型，快速上线

## 🏗️ 系统架构

### 完整架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Request Layer                           │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  RecommendContext                                         │ │
│  │  - UserID, Scene, DeviceID                               │ │
│  │  - UserProfile (用户画像)                                 │ │
│  │  - Realtime Features (实时特征)                           │ │
│  │  - Labels (用户级标签)                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Recall Layer (召回层)                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Fanout (多路并发召回)                                     │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│ │
│  │  │ Hot      │  │ U2I-CF   │  │ I2I-CF   │  │ ANN     ││ │
│  │  │ 热门召回  │  │ 用户协同  │  │ 物品协同  │  │ 向量检索 ││ │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘│ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ MF/ALS   │  │ Content  │  │ History  │              │ │
│  │  │ 矩阵分解  │  │ 内容推荐  │  │ 用户历史  │              │ │
│  │  └──────────┘  └──────────┘  └──────────┘              │ │
│  │                                                         │ │
│  │  Merge Strategy (合并策略)                              │ │
│  │  - First / Union / Priority                            │ │
│  │  - 自定义策略支持                                        │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────────────────┘
                             │ Items[] (候选物品集合)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Filter Layer (过滤层)                        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  FilterNode (组合过滤器)                                  │ │
│  │  - BlacklistFilter (黑名单)                               │ │
│  │  - UserBlockFilter (用户拉黑)                             │ │
│  │  - ExposedFilter (已曝光)                                 │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────────────────┘
                             │ Items[] (过滤后物品)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Feature Layer (特征层)                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  EnrichNode (特征注入)                                    │ │
│  │  - User Features (用户特征)                               │ │
│  │  - Item Features (物品特征)                              │ │
│  │  - Cross Features (交叉特征)                             │ │
│  │  - FeatureService (特征服务)                             │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────────────────┘
                             │ Items[] (带特征物品)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Rank Layer (排序层)                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Rank Models                                             │ │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │ │
│  │  │  LR  │  │ DNN  │  │ DIN  │  │W&D  │  │ RPC  │     │ │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘     │ │
│  │                                                         │ │
│  │  Sort Strategy (排序策略)                                │ │
│  │  - ScoreDesc / ScoreAsc / MultiField                    │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────────────────┘
                             │ Items[] (已排序物品)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ReRank Layer (重排层)                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Diversity (多样性重排)                                    │ │
│  │  - 类别去重                                                │ │
│  │  - 打散策略                                                │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────────────────┘
                             │ Items[] (最终结果)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Layer                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Items[]                                                  │ │
│  │  - ID, Score, Features                                   │ │
│  │  - Labels (全链路标签，可解释性)                           │ │
│  │  - Meta (元信息)                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Store   │  │  Vector  │  │  Service │  │  Feast   │      │
│  │ Redis/   │  │  Milvus  │  │ TF Serve │  │ Feature  │      │
│  │ Memory   │  │          │  │          │  │  Store   │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 设计模式

| 模块 | 设计模式 | 说明 |
|------|---------|------|
| **Pipeline** | Pipeline / Chain of Responsibility | 链式处理，Node 串联 |
| **Recall** | Strategy + Fan-out 并发模式 | 多路召回并发执行 |
| **Rank** | Strategy / Template Method | 多种排序模型策略 |
| **特征注入** | Context Object | 上下文对象传递 |
| **可扩展** | Plugin-like 接口设计 | 接口抽象，插件化扩展 |
| **并发** | errgroup / goroutine pool | 高效并发控制 |
| **存储抽象** | Store 接口 | Redis / MySQL / ES 等 |
| **配置化** | YAML/JSON + Factory 模式 | 配置驱动 |
| **策略驱动** | Label DSL 表达式引擎 | 基于 CEL 的表达式 |

## 🚀 快速开始：构建工业级推荐系统

### 第一步：安装

```bash
go get github.com/rushteam/reckit
```

### 第二步：完整示例（工业级推荐 Pipeline）

```go
package main

import (
    "context"
    "fmt"
    "time"
    
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/feature"
    "github.com/rushteam/reckit/filter"
    "github.com/rushteam/reckit/model"
    "github.com/rushteam/reckit/pipeline"
    "github.com/rushteam/reckit/rank"
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/rerank"
    "github.com/rushteam/reckit/store"
)

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // ========== 1. 初始化基础设施 ==========
    // 生产环境使用 Redis，开发环境可用 Memory
    memStore := store.NewMemoryStore()
    defer memStore.Close()
    
    // 创建协同过滤存储适配器
    cfStore := recall.NewStoreCFAdapter(memStore, "cf")
    
    // ========== 2. 构建召回层 ==========
    config := &core.DefaultRecallConfig{}
    
    // 多路召回：热门 + 协同过滤 + 内容推荐
    fanout := &recall.Fanout{
        Sources: []recall.Source{
            // 热门召回
            &recall.Hot{
                Store: memStore,
                Key:   "hot:feed",
            },
            // 用户协同过滤
            &recall.U2IRecall{
                Store:                cfStore,
                TopKSimilarUsers:     10,
                TopKItems:            20,
                SimilarityCalculator: &recall.CosineSimilarity{},
                Config:                config,
            },
            // 物品协同过滤（工业常青树）
            &recall.I2IRecall{
                Store:                cfStore,
                TopKSimilarItems:     10,
                TopKItems:            20,
                SimilarityCalculator: &recall.CosineSimilarity{},
                Config:                config,
            },
        },
        Dedup:         true,
        Timeout:       2 * time.Second,
        MaxConcurrent: 5,
        MergeStrategy: &recall.PriorityMergeStrategy{},
        ErrorHandler:  &recall.IgnoreErrorHandler{},
    }

    // ========== 3. 构建过滤层 ==========
    storeAdapter := filter.NewStoreAdapter(memStore)
    filterNode := &filter.FilterNode{
        Filters: []filter.Filter{
            // 黑名单过滤
            filter.NewBlacklistFilter([]string{"100", "200"}, storeAdapter, "blacklist:items"),
            // 用户拉黑过滤
            filter.NewUserBlockFilter(storeAdapter, "user:block"),
            // 已曝光过滤（7天窗口）
            filter.NewExposedFilter(storeAdapter, "user:exposed", 7*24*3600, 0),
        },
    }

    // ========== 4. 构建特征层 ==========
    enrichNode := &feature.EnrichNode{
        UserFeaturePrefix:  "user_",
        ItemFeaturePrefix:  "item_",
        CrossFeaturePrefix: "cross_",
        KeyUserFeatures:    []string{"age", "gender", "city"},
        KeyItemFeatures:    []string{"ctr", "cvr", "price"},
    }

    // ========== 5. 构建排序层 ==========
    lrNode := &rank.LRNode{
        Model: &model.LRModel{
            Bias: 0,
            Weights: map[string]float64{
                "item_ctr": 1.2,
                "item_cvr": 0.8,
                "user_age": 0.5,
                "cross_age_x_ctr": 0.2,
            },
        },
        SortStrategy: &rank.ScoreDescSortStrategy{},
    }

    // ========== 6. 构建重排层 ==========
    diversityNode := &rerank.Diversity{
        LabelKey: "category",
    }

    // ========== 7. 组装 Pipeline ==========
    p := &pipeline.Pipeline{
        Nodes: []pipeline.Node{
            fanout,        // 召回
            filterNode,    // 过滤
            enrichNode,    // 特征注入
            lrNode,        // 排序
            diversityNode, // 重排
        },
        Hooks: []pipeline.PipelineHook{
            // 可以添加日志、监控等 Hook
        },
    }

    // ========== 8. 创建用户上下文 ==========
    rctx := &core.RecommendContext{
        UserID: "user_123",
        Scene:  "feed",
        User: &core.UserProfile{
            UserID:    "user_123",
            Age:       25,
            Gender:    "male",
            Location:  "beijing",
            Interests: map[string]float64{"tech": 0.8, "game": 0.6},
        },
        Attributes: map[string]any{
            "age":    25.0,
            "gender": "male",
        },
        Params: map[string]any{
            "hour":   time.Now().Hour(),
            "device": "mobile",
        },
    }

    // ========== 9. 执行 Pipeline ==========
    items, err := p.Run(ctx, rctx, nil)
    if err != nil {
        panic(err)
    }

    // ========== 10. 输出结果 ==========
    fmt.Printf("推荐结果（共 %d 个物品）:\n", len(items))
    for i, item := range items {
        sourceLabel := "unknown"
        if lbl, ok := item.Labels["recall_source"]; ok {
            sourceLabel = lbl.Value
        }
        fmt.Printf("%d. Item: %s, Score: %.4f, Source: %s\n", 
            i+1, item.ID, item.Score, sourceLabel)
    }
}
```

### 第三步：运行

```bash
go run main.go
```

## 💡 核心亮点详解

### 1. 多路并发召回

**工业级特性**：
- ✅ 并发执行多个召回源（errgroup），提升性能
- ✅ 超时控制和限流保护，保证稳定性
- ✅ 自定义合并策略（First / Union / Priority）
- ✅ 自定义错误处理策略（忽略/重试/降级）
- ✅ 自动记录召回来源 Label，支持可解释性

```go
fanout := &recall.Fanout{
    Sources: []recall.Source{
        &recall.Hot{...},
        &recall.U2IRecall{...},
        &recall.I2IRecall{...},
    },
    Dedup:         true,
    Timeout:       2 * time.Second,
    MaxConcurrent: 5,
    MergeStrategy: &recall.PriorityMergeStrategy{},
    ErrorHandler:  &recall.FallbackErrorHandler{
        FallbackSource: &recall.Hot{IDs: []string{"1", "2", "3"}},
    },
}
```

### 2. Labels-first 可解释性

**全链路标签追踪**：
- 每个物品自动记录召回来源、排序模型、过滤原因等
- 支持自定义 Label 合并策略
- 支持 Label DSL 表达式，实现策略驱动

```go
// 自动记录标签
item.PutLabel("recall_source", utils.Label{Value: "hot", Source: "recall"})
item.PutLabel("rank_model", utils.Label{Value: "lr", Source: "rank"})

// 使用 DSL 表达式
eval := dsl.NewEval(item, rctx)
result, _ := eval.Evaluate(`label.recall_source == "hot" && item.score > 0.7`)
```

### 3. 完全可扩展

**所有策略都通过接口实现**：
- 合并策略：实现 `MergeStrategy` 接口
- 排序策略：实现 `SortStrategy` 接口
- 相似度计算：实现 `SimilarityCalculator` 接口
- 错误处理：实现 `ErrorHandler` 接口
- 动态注册：通过 `NodeFactory` 注册自定义 Node

```go
// 自定义合并策略
type CustomMergeStrategy struct{}
func (s *CustomMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
    // 自定义逻辑：例如只保留分数最高的物品
}

// 使用
fanout.MergeStrategy = &CustomMergeStrategy{}
```

### 4. Pipeline Hook 机制

**支持中间件功能**：
- 日志记录
- 性能监控
- 缓存
- 数据统计

```go
type MetricsHook struct {
    startTime time.Time
}

func (h *MetricsHook) BeforeNode(ctx context.Context, rctx *core.RecommendContext, 
    node pipeline.Node, items []*core.Item) ([]*core.Item, error) {
    h.startTime = time.Now()
    return items, nil
}

func (h *MetricsHook) AfterNode(ctx context.Context, rctx *core.RecommendContext, 
    node pipeline.Node, items []*core.Item, err error) ([]*core.Item, error) {
    duration := time.Since(h.startTime)
    metrics.RecordNodeDuration(node.Name(), duration)
    return items, err
}
```

## 📦 支持的算法和模型

### 召回算法

| 算法 | 实现 | 工业地位 | 适用场景 |
|------|------|---------|---------|
| **Item-CF** | `I2IRecall` | ⭐⭐⭐⭐⭐ 工业常青树 | 电商、内容流 |
| **User-CF** | `U2IRecall` | ⭐⭐⭐ 离线分析 | 冷启动补充 |
| **MF/ALS** | `MFRecall` | ⭐⭐⭐⭐ 广泛使用 | 矩阵分解 |
| **Embedding** | `EmbRecall` | ⭐⭐⭐⭐⭐ 主流方案 | 向量检索 |
| **Content** | `ContentRecall` | ⭐⭐⭐⭐ 冷启动首选 | 新物品推荐 |
| **热门** | `Hot` | ⭐⭐⭐⭐⭐ 必备 | 兜底召回 |
| **用户历史** | `UserHistory` | ⭐⭐⭐⭐ 个性化 | 基于行为 |

### 排序模型

| 模型 | 实现 | 说明 |
|------|------|------|
| **LR** | `LRNode` | 线性回归，快速上线 |
| **DNN** | `DNNNode` | 深度神经网络 |
| **Wide&Deep** | `WideDeepNode` | Wide&Deep 模型 |
| **DIN** | `DINNode` | Deep Interest Network |
| **Two Tower** | `TwoTowerNode` | 双塔模型 |
| **RPC** | `RPCNode` | 外部模型服务（XGBoost、TF Serving 等） |

## 🔧 配置化 Pipeline

支持从 YAML/JSON 配置文件加载 Pipeline，无需修改代码：

```yaml
pipeline:
  name: "production_recommendation"
  nodes:
    - type: "recall.fanout"
      config:
        dedup: true
        timeout: 2
        max_concurrent: 5
        merge_strategy: "priority"
        sources:
          - type: "hot"
            ids: ["1", "2", "3", "4", "5"]
    
    - type: "filter"
      config:
        filters:
          - type: "blacklist"
            item_ids: ["100", "200"]
          - type: "exposed"
            key_prefix: "user:exposed"
            time_window: 604800
    
    - type: "feature.enrich"
      config:
        user_feature_prefix: "user_"
        item_feature_prefix: "item_"
    
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

```go
import "github.com/rushteam/reckit/config"

cfg, _ := pipeline.LoadFromYAML("pipeline.yaml")
factory := config.NewFactoryWithBuiltins(config.BuiltinDependencies{
    FilterStore:    myStore,          // 让 blacklist/user_block/exposed 可读 Store
    FeatureService: myFeatureService, // 让 feature.enrich 走 FeatureService
})
p, _ := cfg.BuildPipeline(factory)
items, _ := p.Run(ctx, rctx, nil)
```

## 📚 文档

- [架构设计文档](docs/ARCHITECTURE.md) - 详细架构设计说明
- [召回算法文档](docs/RECALL_ALGORITHMS.md) - 召回算法详解和使用指南
- [排序模型文档](docs/RANK_MODELS.md) - 排序模型详解
- [模型选型指南](docs/MODEL_SELECTION.md) - 推荐系统模型选型参考
- [协同过滤文档](docs/COLLABORATIVE_FILTERING.md) - 协同过滤算法详解
- [可扩展性分析](docs/EXTENSIBILITY_ANALYSIS.md) - 扩展指南和最佳实践
- [用户画像文档](docs/USER_PROFILE.md) - 用户画像使用指南
- [AI Coding 指南](CLAUDE.md) - AI 辅助开发指南

## 🗂️ 目录结构

```
github.com/rushteam/reckit/
├── core/              # 核心数据结构（Item, Context, UserProfile, Config）
├── pipeline/          # Pipeline 和 Node 接口
├── recall/            # 召回模块（Source, Fanout, CF, ANN, Content 等）
├── filter/            # 过滤模块（Blacklist, UserBlock, Exposed）
├── rank/              # 排序模块（LR, DNN, DIN, RPC 等）
├── rerank/            # 重排模块（Diversity）
├── model/             # 排序模型抽象和实现
├── feature/            # 特征服务（Enrich, Service, Provider）
├── store/             # 存储抽象（Memory，Redis 移至扩展包）
├── vector/            # 向量服务接口（Milvus 移至扩展包）
├── service/           # ML 服务（TF Serving, ANN Service）
├── feature/           # 特征服务（领域层接口 FeatureService）
├── config/            # Pipeline 配置工厂
├── ext/                # 扩展包目录（独立 go.mod）
│   ├── store/redis/   # Redis 存储实现
│   ├── feast/
│   │   ├── http/      # Feast HTTP 客户端实现
│   │   └── grpc/      # Feast gRPC 客户端实现
│   └── vector/milvus/ # Milvus 向量数据库实现
├── pkg/
│   ├── utils/         # Label 工具
│   └── dsl/           # Label DSL 表达式引擎
├── python/            # Python ML 训练与服务
└── examples/          # 示例代码
```

## 🔧 依赖

### 核心包依赖

核心包 `github.com/rushteam/reckit` **无外部依赖**，只保留工具库：

```go
require (
    github.com/google/cel-go v0.26.1  // CEL 表达式引擎（Label DSL）
    golang.org/x/sync v0.19.0          // 并发工具
    gopkg.in/yaml.v3 v3.0.1            // YAML 配置解析
)
```

### 扩展包依赖

具体实现位于扩展包中，用户按需引入：

- **Redis Store**: `go get github.com/rushteam/reckit/ext/store/redis`
- **Feast HTTP/gRPC**: `go get github.com/rushteam/reckit/ext/feast/http` 或 `/grpc`（通过适配器适配为 `feature.FeatureService`）
- **Milvus Vector**: `go get github.com/rushteam/reckit/ext/vector/milvus`

**或自行实现**：参考扩展包实现，自行实现对应接口（推荐实现领域层接口，如 `feature.FeatureService`）。

详见 `ext/README.md`。

### Python 依赖

见 `python/requirements.txt`：
- xgboost
- fastapi
- uvicorn
- pandas
- numpy
- scikit-learn

## 🤝 贡献

欢迎贡献代码！请阅读 [贡献指南](CONTRIBUTING.md) 了解详细信息。

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

---

<div align="center">

**Made with ❤️ by [Rush Team](https://github.com/rushteam)**

[文档](README.md) • [示例](examples/) • [问题反馈](https://github.com/rushteam/reckit/issues)

</div>
