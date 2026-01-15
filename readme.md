# Reckit

<div align="center">

**工业级推荐系统工具包 | Production-Ready Recommender System Toolkit**

[![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8?style=flat-square&logo=go)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Go Report Card](https://goreportcard.com/badge/github.com/rushteam/reckit?style=flat-square)](https://goreportcard.com/report/github.com/rushteam/reckit)

</div>

---

## 📖 简介

Reckit 是一个**工业级推荐系统工具包**，采用 **Pipeline + Node** 架构设计，通过接口抽象实现高度可扩展性。支持内容流、电商、广告等多种推荐场景。

### 核心特性

- 🎯 **Pipeline-first 架构**：所有推荐逻辑通过 Node 串联，灵活组合
- 🏷️ **Labels-first 设计**：Labels 全链路透传，支持可解释性和策略驱动
- 🔌 **高度可扩展**：通过接口实现，无需修改库代码即可扩展所有功能
- ⚡ **高性能并发**：多路召回并发执行，支持超时控制和限流
- 🎨 **策略模式丰富**：合并策略、排序策略、错误处理策略等均可自定义
- 🔧 **配置化支持**：支持 YAML/JSON 配置，Pipeline 可配置化加载

## 🏗️ 架构设计

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

## 🚀 快速开始

### 安装

```bash
go get github.com/rushteam/reckit
```

### 基础示例

```go
package main

import (
    "context"
    "time"
    
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/pipeline"
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/rank"
    "github.com/rushteam/reckit/store"
)

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // 初始化存储
    memStore := store.NewMemoryStore()
    defer memStore.Close()

    // 创建召回源
    config := &core.DefaultRecallConfig{}
    u2i := &recall.U2IRecall{
        Store:                cfStore,
        TopKSimilarUsers:     10,
        TopKItems:            20,
        SimilarityCalculator: &recall.CosineSimilarity{},
        Config:                config,
    }

    // 构建 Pipeline
    p := &pipeline.Pipeline{
        Nodes: []pipeline.Node{
            &recall.Fanout{
                Sources: []recall.Source{
                    &recall.Hot{IDs: []string{"1", "2", "3"}},
                    u2i,
                },
                Dedup:         true,
                MergeStrategy: &recall.PriorityMergeStrategy{},
            },
            &rank.LRNode{
                Model:        lrModel,
                SortStrategy: &rank.ScoreDescSortStrategy{},
            },
        },
    }

    // 创建用户上下文
    rctx := &core.RecommendContext{
        UserID: "user_123",
        Scene:  "feed",
        User: &core.UserProfile{
            UserID:    "user_123",
            Age:       25,
            Interests: map[string]float64{"tech": 0.8},
        },
    }

    // 执行 Pipeline
    items, err := p.Run(ctx, rctx, nil)
    if err != nil {
        panic(err)
    }

    // 输出结果
    for _, item := range items {
        fmt.Printf("Item: %s, Score: %.4f\n", item.ID, item.Score)
    }
}
```

### 运行示例

```bash
# 基础示例
go run ./examples/basic

# 配置化 Pipeline
go run ./examples/config

# 所有召回算法示例
go run ./examples/all_recall_algorithms

# 个性化推荐示例
go run ./examples/personalization
```

## 📦 核心模块

### Pipeline + Node 架构

所有推荐逻辑通过 Pipeline Node 串联，每个 Node 处理 Items、Score、Labels：

```go
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},      // 召回
        &filter.FilterNode{...},  // 过滤
        &feature.EnrichNode{...}, // 特征注入
        &rank.LRNode{...},        // 排序
        &rerank.Diversity{...},   // 重排
    },
    Hooks: []pipeline.PipelineHook{
        &LoggingHook{},  // 日志 Hook
    },
}
```

### 召回模块（Recall）

#### 多路并发召回

```go
fanout := &recall.Fanout{
    Sources: []recall.Source{
        &recall.Hot{IDs: []string{"1", "2", "3"}},
        &recall.U2IRecall{...},
        &recall.I2IRecall{...},
    },
    Dedup:         true,
    Timeout:       2 * time.Second,
    MaxConcurrent: 5,
    MergeStrategy: &recall.PriorityMergeStrategy{},
    ErrorHandler:  &recall.IgnoreErrorHandler{},
}
```

**特性**：
- ✅ 并发执行多个召回源（errgroup）
- ✅ 超时控制和限流
- ✅ 自定义合并策略（First / Union / Priority）
- ✅ 自定义错误处理策略
- ✅ 自动记录召回来源 Label

#### 支持的召回算法

| 算法 | 实现 | 说明 |
|------|------|------|
| **User-CF** | `U2IRecall` | 用户协同过滤 |
| **Item-CF** | `I2IRecall` | 物品协同过滤（工业常青树） |
| **MF/ALS** | `MFRecall` | 矩阵分解 |
| **Embedding** | `EmbRecall` | 向量检索召回 |
| **Content** | `ContentRecall` | 内容推荐 |
| **热门** | `Hot` | 热门物品召回 |
| **用户历史** | `UserHistory` | 基于用户历史行为 |

### 排序模块（Rank）

#### 支持的排序模型

| 模型 | 实现 | 说明 |
|------|------|------|
| **LR** | `LRNode` | 线性回归 |
| **DNN** | `DNNNode` | 深度神经网络 |
| **Wide&Deep** | `WideDeepNode` | Wide&Deep 模型 |
| **DIN** | `DINNode` | Deep Interest Network |
| **Two Tower** | `TwoTowerNode` | 双塔模型 |
| **RPC** | `RPCNode` | 外部模型服务（XGBoost、TF Serving 等） |

#### 使用示例

```go
// LR 模型
lrNode := &rank.LRNode{
    Model: &model.LRModel{
        Bias: 0,
        Weights: map[string]float64{
            "ctr": 1.2,
            "cvr": 0.8,
        },
    },
    SortStrategy: &rank.ScoreDescSortStrategy{},
}

// RPC 模型（XGBoost）
xgbModel := model.NewRPCModel("xgboost", "http://localhost:8080/predict", 5*time.Second)
rpcNode := &rank.RPCNode{Model: xgbModel}
```

### 过滤模块（Filter）

```go
filterNode := &filter.FilterNode{
    Filters: []filter.Filter{
        filter.NewBlacklistFilter([]string{"100", "200"}, nil, ""),
        filter.NewUserBlockFilter(storeAdapter, "user:block"),
        filter.NewExposedFilter(storeAdapter, "user:exposed", 7*24*3600),
    },
}
```

### 特征工程模块（Feature）

```go
enrichNode := &feature.EnrichNode{
    FeatureService:     featureService,
    UserFeaturePrefix:  "user_",
    ItemFeaturePrefix:  "item_",
    CrossFeaturePrefix: "cross_",
    KeyUserFeatures:    []string{"age", "gender"},
    KeyItemFeatures:    []string{"ctr", "cvr", "price"},
}
```

### 配置化 Pipeline

支持从 YAML/JSON 配置文件加载 Pipeline：

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
            ids: ["1", "2", "3", "4", "5"]
    
    - type: "rank.lr"
      config:
        bias: 0.0
        weights:
          ctr: 1.2
          cvr: 0.8
```

```go
cfg, _ := pipeline.LoadFromYAML("pipeline.yaml")
factory := config.DefaultFactory()
p, _ := cfg.BuildPipeline(factory)
items, _ := p.Run(ctx, rctx, nil)
```

## 🎯 核心特性详解

### Labels-first 设计

Labels 自动透传，节点之间不丢失，支持 Merge 和策略驱动：

```go
item.PutLabel("recall_source", utils.Label{Value: "hot", Source: "recall"})
item.PutLabel("rank_model", utils.Label{Value: "lr", Source: "rank"})

// 自定义 Label 合并策略
item.LabelMergeStrategy = &utils.PriorityLabelMergeStrategy{
    SourcePriority: map[string]int{
        "recall": 1,
        "rank":   2,
    },
}
```

### 可扩展性设计

所有策略都通过接口实现，支持自定义：

```go
// 自定义合并策略
type CustomMergeStrategy struct{}
func (s *CustomMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
    // 自定义逻辑
}

// 自定义相似度计算器
type JaccardSimilarity struct{}
func (j *JaccardSimilarity) Calculate(x, y []float64) float64 {
    // 实现 Jaccard 相似度
}

// 动态注册 Node
factory := pipeline.NewNodeFactory()
factory.Register("my.custom.node", buildMyCustomNode)
```

### Pipeline Hook 机制

支持在执行前后插入逻辑，用于日志、监控、缓存等：

```go
type LoggingHook struct{}

func (h *LoggingHook) BeforeNode(ctx context.Context, rctx *core.RecommendContext, 
    node pipeline.Node, items []*core.Item) ([]*core.Item, error) {
    fmt.Printf("[Hook] Before %s: %d items\n", node.Name(), len(items))
    return items, nil
}

p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{...},
    Hooks: []pipeline.PipelineHook{&LoggingHook{}},
}
```

## 📚 文档

- [架构设计文档](ARCHITECTURE.md) - 详细架构设计说明
- [召回算法文档](RECALL_ALGORITHMS.md) - 召回算法详解和使用指南
- [排序模型文档](RANK_MODELS.md) - 排序模型详解
- [协同过滤文档](COLLABORATIVE_FILTERING.md) - 协同过滤算法详解
- [可扩展性分析](EXTENSIBILITY_ANALYSIS.md) - 扩展指南和最佳实践
- [用户画像文档](USER_PROFILE.md) - 用户画像使用指南
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
├── feature/           # 特征服务（Enrich, Service, Provider）
├── store/             # 存储抽象（Memory, Redis）
├── vector/            # 向量服务（Milvus）
├── service/           # ML 服务（TF Serving, ANN Service）
├── feast/             # Feast 集成
├── config/            # Pipeline 配置工厂
├── pkg/
│   ├── utils/         # Label 工具
│   └── dsl/           # Label DSL 表达式引擎
├── python/            # Python ML 训练与服务
└── examples/          # 示例代码
```

## 🔧 依赖

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

## 🤝 贡献

欢迎贡献代码！请阅读 [贡献指南](CONTRIBUTING.md) 了解详细信息。

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

---

<div align="center">

**Made with ❤️ by [Rush Team](https://github.com/rushteam)**

[文档](readme.md) • [示例](examples/) • [问题反馈](https://github.com/rushteam/reckit/issues)

</div>
