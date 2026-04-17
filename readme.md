# Reckit

<div align="center">

**工业级推荐系统工具包 | Production-Ready Recommender System Toolkit**

[![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8?style=flat-square&logo=go)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Go Report Card](https://goreportcard.com/badge/github.com/rushteam/reckit?style=flat-square)](https://goreportcard.com/report/github.com/rushteam/reckit)

</div>

---

## 简介

Reckit 是一个**工业级推荐系统工具包**，采用 **Pipeline + Node** 架构设计，通过接口抽象实现高度可扩展性。支持内容流、电商、广告等多种推荐场景，帮助开发者快速构建生产级推荐系统。

### 工程目标

| 场景 | 工具 |
|------|------|
| 深度模型、复杂梯度更新 | PyTorch / TensorFlow |
| 高并发、低延迟、统计计算 | **Golang (Reckit)** |

### 核心亮点

- **Pipeline-first 架构**：链式 Node 处理，灵活组合推荐流程
- **Labels-first 设计**：全链路标签透传，支持可解释性和策略驱动
- **完全可扩展**：所有策略通过接口实现，无需修改库代码
- **高性能并发**：多路召回并发执行，超时控制和限流保护
- **策略模式丰富**：合并、排序、多样性、探索利用等策略均可自定义
- **配置化支持**：YAML 配置 + Node 注册工厂，Pipeline 可配置化加载
- **开箱即用**：内置 15 种召回算法、6 种排序模型、13+ 种重排策略

## 系统架构

```
Request → Context → Recall → Filter → Rank → ReRank → PostProcess → Response
```

```
┌─────────────────────────────────────────────────────────────────┐
│                      Recall Layer (召回层)                       │
│  Fanout (多路并发召回，支持嵌套)                                  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
│  │ Hot  │ │ CF   │ │ ANN  │ │ MF   │ │ BERT │ │ RPC  │ ...  │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘      │
│  MergeStrategy: First/Union/Priority/Weighted/Quota/Ratio/...  │
│  ErrorHandler:  Ignore / Retry / Fallback                      │
└────────────────────────────┬──────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Filter Layer (过滤层)                       │
│  FilterNode (BatchFilter → 逐条 Filter)                         │
│  Blacklist │ UserBlock │ Exposed │ Expr(CEL) │ QualityGate     │
│  DedupField │ TimeDecay │ FrequencyCap │ ConditionalNode       │
└────────────────────────────┬──────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature + Rank Layer (特征+排序)                 │
│  EnrichNode → LR / DNN / DIN / Wide&Deep / TwoTower / RPC      │
│  SortStrategy: ScoreDesc / ScoreAsc / MultiField               │
└────────────────────────────┬──────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ReRank Layer (重排层)                         │
│  Diversity (规则/DPP/SSD) │ TopN │ Sample │ GroupQuota          │
│  FairInterleave │ WeightedInterleave │ RecallChannelMix         │
│  TrafficPlan │ ScoreAdjust │ ScoreWeightBoost │ MMoE            │
│  EpsilonGreedy │ UCB │ ThompsonSampling │ ColdStartBoost       │
└────────────────────────────┬──────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PostProcess Layer (后处理层)                     │
│  PaddingNode (补足) │ TruncateFieldsNode (裁剪字段)               │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
go get github.com/rushteam/reckit
```

### 完整示例

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
    "github.com/rushteam/reckit/postprocess"
    "github.com/rushteam/reckit/rank"
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/rerank"
    "github.com/rushteam/reckit/store"
)

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    memStore := store.NewMemoryStore()
    defer memStore.Close()
    cfStore := recall.NewStoreCFAdapter(memStore, "cf")
    storeAdapter := filter.NewStoreAdapter(memStore)
    config := &core.DefaultRecallConfig{}

    p := &pipeline.Pipeline{
        Nodes: []pipeline.Node{
            // 召回：多路并发
            &recall.Fanout{
                Sources: []recall.Source{
                    recall.NewHotRecall(memStore, "hot:feed", 100),
                    &recall.U2IRecall{
                        Store: cfStore, TopKSimilarUsers: 10, TopKItems: 20,
                        SimilarityCalculator: &recall.CosineSimilarity{}, Config: config,
                    },
                },
                Dedup: true, Timeout: 2 * time.Second, MaxConcurrent: 5,
                MergeStrategy: &recall.PriorityMergeStrategy{},
                ErrorHandler:  &recall.IgnoreErrorHandler{},
            },
            // 过滤
            &filter.FilterNode{Filters: []filter.Filter{
                filter.NewBlacklistFilter([]string{"100"}, storeAdapter, "blacklist"),
                &filter.QualityGateFilter{MinScore: 0.01},
            }},
            // 特征注入
            &feature.EnrichNode{UserFeaturePrefix: "user_", ItemFeaturePrefix: "item_"},
            // 排序
            &rank.LRNode{
                Model:        &model.LRModel{Weights: map[string]float64{"item_ctr": 1.2, "item_cvr": 0.8}},
                SortStrategy: &rank.ScoreDescSortStrategy{},
            },
            // 重排：多样性 + 探索
            &rerank.Diversity{LabelKey: "category"},
            &rerank.EpsilonGreedyNode{Epsilon: 0.1, ExploitSize: 10},
            &rerank.TopNNode{N: 20},
            // 后处理：补足 + 裁剪
            &postprocess.PaddingNode{N: 20, FallbackItems: defaultItems()},
            &postprocess.TruncateFieldsNode{ClearFeatures: true},
        },
        ErrorHooks: []pipeline.ErrorHook{
            &pipeline.KindRecoveryHook{
                RecoverKinds: map[pipeline.Kind]bool{
                    pipeline.KindReRank:      true,
                    pipeline.KindPostProcess: true,
                },
            },
        },
    }

    rctx := &core.RecommendContext{
        UserID: "user_123", Scene: "feed",
        Attributes: map[string]any{"age": 25.0, "gender": "male"},
    }

    items, err := p.Run(ctx, rctx, nil)
    if err != nil {
        panic(err)
    }
    fmt.Printf("推荐结果: %d 条\n", len(items))
}

func defaultItems() []*core.Item {
    ids := []string{"default_1", "default_2", "default_3"}
    out := make([]*core.Item, len(ids))
    for i, id := range ids {
        out[i] = core.NewItem(id)
    }
    return out
}
```

## 支持的算法和模型

### 召回算法（15 种）

| 算法 | 实现 | 说明 |
|------|------|------|
| **Item-CF** | `I2IRecall` | 物品协同过滤，工业常青树 |
| **User-CF** | `U2IRecall` | 用户协同过滤 |
| **MF/ALS** | `MFRecall` | 矩阵分解 |
| **Embedding ANN** | `ANNRecall` | 向量检索（Milvus/Faiss） |
| **Content** | `ContentRecall` | 基于内容的推荐 |
| **SortedSet** | `SortedSetRecall` | 通用有序集合召回（热门/趋势/最新/高分等） |
| **UserHistory** | `UserHistory` | 用户历史召回 |
| **Word2Vec/Item2Vec** | `Word2VecRecall` | 文本模式 + 序列模式 |
| **BERT** | `BERTRecall` | 语义相似度召回 |
| **RPC** | `RPCRecall` | 外部召回服务（通用 HTTP） |
| **Two-Tower** | `TwoTowerRecall` | 双塔模型召回 |
| **YouTube DNN** | `YouTubeDNNRecall` | YouTube DNN 召回 |
| **DSSM** | `DSSMRecall` | Query-Doc 语义召回 |
| **Graph/Node2Vec** | `GraphRecall` | 图嵌入社交/关注页召回 |
| **嵌套 Fanout** | `Fanout` | 树形召回拓扑，Source + Node 双接口 |

### 过滤器（11 种）

| 过滤器 | 实现 | 说明 |
|--------|------|------|
| **黑名单** | `BlacklistFilter` | 物品级黑名单 |
| **用户屏蔽** | `UserBlockFilter` | 用户级屏蔽 |
| **已曝光** | `ExposedFilter` | IDs 列表 + 布隆过滤器双通道 |
| **批量曝光** | `BatchExposedFilter` | 批量曝光判定（BatchFilter），高 QPS 场景优先 |
| **表达式** | `ExprFilter` | CEL/DSL 通用规则过滤（支持 Invert） |
| **分数门槛** | `QualityGateFilter` | Score < MinScore 直接过滤 |
| **字段去重** | `DedupByFieldFilter` | BatchFilter，按字段去重保留首条 |
| **时间衰减** | `TimeDecayFilter` | 按发布时间过滤过期内容 |
| **频次控制** | `FrequencyCapFilter` | user-item 粒度曝光频次上限 |
| **条件节点** | `ConditionalNode` | 条件为 true 时执行内部 Node（AB 实验/场景开关） |
| **组合过滤** | `FilterNode` | 组合多个 Filter，BatchFilter 优先 |

### 排序模型（6 种）

| 模型 | 实现 | 说明 |
|------|------|------|
| **LR** | `LRNode` | 逻辑回归，快速上线 |
| **DNN** | `DNNNode` | 深度神经网络（RPC） |
| **DIN** | `DINNode` | Deep Interest Network |
| **Wide&Deep** | `WideDeepNode` | Wide&Deep（RPC） |
| **Two Tower** | `TwoTowerNode` | 双塔模型（RPC） |
| **RPC** | `RPCNode` | 外部模型服务（KServe V2） |

### 重排策略（17 种）

| 策略 | 实现 | 说明 |
|------|------|------|
| **多样性** | `Diversity` | 类别去重 / 多 key 打散 / 高级 Constraints 多规则 |
| **DPP 多样性** | `DPPDiversityNode` | Determinantal Point Process，embedding 级多样性 |
| **SSD 多样性** | `SSDDiversityNode` | 滑动子空间多样性，比 DPP 更轻量 |
| **TopN** | `TopNNode` | 截断 |
| **采样** | `SampleNode` | N 采样，可选 Fisher-Yates 洗牌 |
| **公平交叉** | `FairInterleaveNode` | 按召回通道等权轮询交叉 |
| **加权交叉** | `WeightedInterleaveNode` | 按通道权重交叉混排 |
| **分组配额** | `GroupQuotaNode` | 按维度/CEL 表达式分组，softmax/avg 配额分配 |
| **流量调控** | `TrafficPlanNode` | 调控 id/位次写入 Label，可选重排 |
| **改分** | `ScoreAdjust` | Filter/CEL 规则多条件改分 |
| **权重提升** | `ScoreWeightBoost` | 按 ID 外部权重批量调分 |
| **通道混排** | `RecallChannelMix` | 精排后按召回通道固定/随机槽位编排 |
| **MMoE** | `MMoENode` | 多目标重排（CTR + 时长 + GMV） |
| **ε-贪心** | `EpsilonGreedyNode` | 以概率 ε 注入随机探索 |
| **UCB** | `UCBNode` | UCB1 置信上界探索（需 BanditStatsProvider） |
| **汤普森采样** | `ThompsonSamplingNode` | Beta-Bernoulli 采样（需 BanditStatsProvider） |
| **冷启动提权** | `ColdStartBoostNode` | 新物品线性衰减提权 |

### 后处理（2 种）

| 节点 | 实现 | 说明 |
|------|------|------|
| **补足** | `PaddingNode` | 结果不足 N 条时补足，去重并打标签 |
| **裁剪字段** | `TruncateFieldsNode` | 清理 Features/Meta/Labels，减少序列化体积 |

### 合并策略（10 种）

| 策略 | 说明 |
|------|------|
| `FirstMergeStrategy` | 保留首次出现（默认） |
| `UnionMergeStrategy` | 合并并去重 |
| `PriorityMergeStrategy` | 按源优先级 |
| `WeightedScoreMergeStrategy` | 按源权重调分后排序 |
| `QuotaMergeStrategy` | 每源固定配额 |
| `RatioMergeStrategy` | 按比例分配 |
| `HybridRatioMergeStrategy` | 未配置源保留 + 显式源按剩余槽位比例分配 |
| `RoundRobinMergeStrategy` | 轮询交叉 |
| `WaterfallMergeStrategy` | 瀑布级联兜底 |
| `ChainMergeStrategy` | 串联组合多策略 |

## 配置化 Pipeline

支持 YAML 配置 + Node 注册工厂，涵盖所有内置节点：

```yaml
pipeline:
  name: "production_feed"
  nodes:
    - type: "recall.fanout"
      config:
        dedup: true
        timeout: 2
        merge_strategy: "hybrid_ratio"
        total_limit: 100
        source_ratios: { "recall.hot": 2.0, "recall.cf": 1.0 }
        sort_by_priority_before_dedup: true
        sources:
          - type: "hot"
            ids: ["1", "2", "3"]

    - type: "filter"
      config:
        filters:
          - type: "blacklist"
            item_ids: ["100"]
          - type: "quality_gate"
            min_score: 0.01
          - type: "exposed_batch"
            key_prefix: "user:exposed"
            time_window: 604800
            bloom_filter_day_window: 28
          - type: "expr"
            expr: 'label.category == "adult"'
          - type: "time_decay"
            time_field: "publish_time"
            max_age_seconds: 604800

    - type: "rank.lr"
      config:
        weights: { ctr: 1.2, cvr: 0.8 }
        explain:
          emit_raw_score: true
          emit_missing_flag: true
          emit_feature_coverage: true

    - type: "rerank.diversity"
      config:
        label_key: "category"
        constraints:
          - dimensions: ["category"]
            max_consecutive: 2
            window_size: 5

    - type: "rerank.epsilon_greedy"
      config:
        epsilon: 0.1
        exploit_size: 20

    - type: "rerank.topn"
      config:
        n: 20

    - type: "postprocess.padding"
      config:
        n: 20

    - type: "postprocess.truncate_fields"
      config:
        clear_features: true
        keep_meta_keys: ["title", "cover"]
```

```go
import (
    "github.com/rushteam/reckit/config"
    "github.com/rushteam/reckit/config/builders"
)

factory := builders.NewFactory(builders.Dependencies{
    FilterStore:         myStore,
    BatchExposureChecker: myBatchExposureChecker, // exposed_batch（可注入 Tair/Redis bloom 批量检查）
    FeatureService:      myFeatureService,
    BanditStatsProvider: myBanditStats,   // UCB / Thompson / ColdStart
    FrequencyCapStore:   myFreqCapStore,  // FrequencyCap Filter
    PaddingFunc:         myPaddingFunc,   // Padding 动态补足
})

cfg, _ := pipeline.LoadFromYAML("pipeline.yaml")
p, _ := cfg.BuildPipeline(factory)
items, _ := p.Run(ctx, rctx, nil)
```

**已注册的 YAML 构建器**：

| 类别 | 支持的 type |
|------|------------|
| **Recall** | `recall.fanout`, `recall.hot` / `recall.sorted_set`, `recall.ann`, `recall.u2i`, `recall.i2i`, `recall.content`, `recall.mf`, `recall.user_history`, `recall.word2vec`, `recall.bert`, `recall.two_tower`, `recall.youtube_dnn`, `recall.dssm`, `recall.rpc`, `recall.graph` |
| **Filter** | `filter`（含 `blacklist` / `user_block` / `exposed` / `exposed_batch` / `expr` / `quality_gate` / `dedup_field` / `time_decay` / `frequency_cap`）, `filter.conditional` |
| **Rank** | `rank.lr`, `rank.rpc`, `rank.wide_deep`, `rank.two_tower`, `rank.dnn`, `rank.din` |
| **ReRank** | `rerank.diversity`, `rerank.dpp_diversity`, `rerank.ssd_diversity`, `rerank.topn`, `rerank.sample`, `rerank.fair_interleave`, `rerank.weighted_interleave`, `rerank.group_quota`, `rerank.traffic_plan`, `rerank.score_adjust`, `rerank.score_weight`, `rerank.recall_channel_mix`, `rerank.mmoe`, `rerank.epsilon_greedy`, `rerank.ucb`, `rerank.thompson_sampling`, `rerank.cold_start_boost` |
| **PostProcess** | `postprocess.padding`, `postprocess.truncate_fields` |
| **Feature** | `feature.enrich` |

## 核心设计

### 策略接口

所有策略通过接口实现，用户可自定义扩展，无需修改库代码：

| 接口 | 说明 | 内置实现 |
|------|------|---------|
| `MergeStrategy` | 召回合并策略 | 10 种 |
| `ErrorHandler` | 召回错误处理 | Ignore / Retry / Fallback |
| `SortStrategy` | 排序策略 | ScoreDesc / ScoreAsc / MultiField |
| `SimilarityCalculator` | 相似度计算 | Cosine / Pearson |
| `RankModel` | 排序模型 | LR / RPC / MLServiceAdapter |
| `Filter` / `BatchFilter` | 过滤器 | 11 种 |
| `PipelineHook` | Pipeline 中间件 | 自定义 |
| `ErrorHook` | Pipeline 错误钩子 | WarnAndSkip / KindRecovery / ErrorCallback |
| `TrafficPlanner` | 流量调控 | NoOp / Static / 自定义 |
| `ScoreWeightProvider` | 外部权重 | 自定义 |
| `BanditStatsProvider` | Bandit 统计 | 自定义（供 UCB/Thompson/ColdStart） |
| `FrequencyCapStore` | 曝光频次 | 自定义 |
| `BatchExposureChecker` | 批量曝光判定 | 自定义（供 `exposed_batch`） |
| `Condition` | 条件判断 | ConditionFunc 适配器 |
| `ABRuntime` | AB 运行时扩展 | 自定义（通过 `core.Extension` 注入） |

### Labels-first 可解释性

全链路标签追踪 + CEL 表达式引擎：

```go
// 全链路标签
item.PutLabel("recall_source", utils.Label{Value: "hot", Source: "recall"})
item.PutLabel("rank_model", utils.Label{Value: "lr", Source: "rank"})

// CEL 表达式过滤/改分
&filter.ExprFilter{Expr: `label.category == "adult"`}
&rerank.ScoreAdjust{Rules: []rerank.ScoreAdjustRule{
    {Expr: `label.recall_source == "hot"`, Mode: "mul", Value: 1.5},
}}
```

`rank.LRNode` 支持可选 explain 标签输出：

```go
&rank.LRNode{
    Model: lrModel,
    Explain: &rank.LRExplainConfig{
        EmitRawScore:        true, // rank_linear_raw
        EmitMissingFlag:     true, // rank_features_missing
        EmitFeatureCoverage: true, // rank_feature_coverage
    },
}
```

### AB Runtime / 诊断 / 统计模板

```go
// 1) AB 决策 helper（未注入 runtime 时自动降级返回零值）
decision, _ := core.GetABDecision(ctx, rctx, "aippy.abtest", "feed_new_rank")

// 2) 过滤诊断（定位 item 被哪个过滤器拦截）
diag := filter.DiagnoseItem(ctx, rctx, item, filters)

// 3) 通用统计 hook（记录每个 node 的输入/输出条数到 rctx.Params）
p := &pipeline.Pipeline{
    Nodes: nodes,
    Hooks: []pipeline.PipelineHook{
        &pipeline.StatsHook{},
    },
}
_ = decision
_ = diag
```

### Explore / Exploit（探索与利用）

```go
// ε-贪心：10% 概率随机替换 top 结果
&rerank.EpsilonGreedyNode{Epsilon: 0.1, ExploitSize: 20}

// UCB1：曝光少的物品获得探索加成
&rerank.UCBNode{Provider: banditStats, C: 1.0}

// Thompson Sampling：Beta 分布采样
&rerank.ThompsonSamplingNode{Provider: banditStats}

// 冷启动提权：曝光 < 100 时线性加成
&rerank.ColdStartBoostNode{Provider: banditStats, Threshold: 100, BoostValue: 2.0}
```

### Pipeline 错误处理与降级

两层机制：Pipeline 全局 + 召回层独立。

```go
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{recall, filter, rank, rerank},
    ErrorHooks: []pipeline.ErrorHook{
        &pipeline.ErrorCallbackHook{Callback: metricsReporter},
        &pipeline.KindRecoveryHook{
            RecoverKinds: map[pipeline.Kind]bool{
                pipeline.KindReRank:      true, // 重排失败 → 跳过
                pipeline.KindPostProcess: true, // 后处理失败 → 跳过
            },
        },
    },
}
```

## 目录结构

```
github.com/rushteam/reckit/
├── core/              # 核心数据结构（Item, Context, UserProfile, Config, Extension）
├── pipeline/          # Pipeline、Node 接口、Hook、ErrorHook
├── recall/            # 召回（15 种 Source + Fanout + 10 种 MergeStrategy + ErrorHandler）
├── filter/            # 过滤（11 种 Filter + FilterNode + ConditionalNode）
├── rank/              # 排序（6 种 RankNode + SortStrategy）
├── rerank/            # 重排（17 种 ReRankNode + Explore/Exploit）
├── postprocess/       # 后处理（Padding、TruncateFields）
├── model/             # 排序模型（LR、RPC、DNN、DIN、WideDeep、TwoTower、Word2Vec、BERT）
├── feature/           # 特征（Enrich、Service、Provider、Cache、Monitor、Metadata、Encoder、Processing）
├── store/             # 存储抽象（Memory，Redis → ext/）
├── vector/            # 向量服务接口（Milvus → ext/）
├── service/           # ML 服务（KServe V2、TF Serving、TorchServe）
├── config/            # Pipeline 配置工厂 + YAML Builder
├── ext/               # 扩展包（独立 go.mod，按需引入）
│   ├── store/redis/   # Redis 存储
│   ├── feast/         # Feast HTTP / gRPC
│   └── vector/milvus/ # Milvus 向量数据库
├── pkg/
│   ├── utils/         # Label 工具
│   ├── dsl/           # CEL 表达式引擎
│   └── conv/          # 类型转换与泛型工具
├── python/            # Python ML 训练与服务
├── k8s/               # Kubernetes 部署配置
└── examples/          # 示例代码（30+ 个）
```

## 依赖

### 核心包

核心包 `github.com/rushteam/reckit` **无外部依赖**，仅保留：

```go
require (
    github.com/google/cel-go v0.26.1  // CEL 表达式引擎
    golang.org/x/sync v0.19.0          // 并发工具
    gopkg.in/yaml.v3 v3.0.1            // YAML 配置解析
)
```

### 扩展包

按需引入，核心包不强制依赖：

- **Redis Store**: `go get github.com/rushteam/reckit/ext/store/redis`
- **Feast HTTP/gRPC**: `go get github.com/rushteam/reckit/ext/feast/http` 或 `/grpc`
- **Milvus Vector**: `go get github.com/rushteam/reckit/ext/vector/milvus`

## 文档

- [架构设计](docs/ARCHITECTURE.md)
- [召回算法](docs/RECALL_ALGORITHMS.md)
- [排序模型](docs/RANK_MODELS.md)
- [可扩展性分析](docs/EXTENSIBILITY_ANALYSIS.md)
- [用户画像](docs/USER_PROFILE.md)
- [特征一致性](docs/FEATURE_CONSISTENCY.md)
- [特征处理](docs/FEATURE_PROCESSING.md)
- [模型服务协议](docs/MODEL_SERVICE_PROTOCOL.md) (KServe V2)
- [Word2Vec / Item2Vec](docs/WORD2VEC_ITEM2VEC.md)
- [AI Coding 指南](AGENTS.md)

## 许可证

[Apache License 2.0](LICENSE)

---

<div align="center">

**Made with ❤️ by [Rush Team](https://github.com/rushteam)**

[文档](README.md) · [示例](examples/) · [问题反馈](https://github.com/rushteam/reckit/issues)

</div>
