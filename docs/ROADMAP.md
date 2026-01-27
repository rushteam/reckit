# Reckit 开发路线图

本文档描述 Reckit 项目的未来发展规划和待实现功能。

---

## 🎯 短期规划（1-3 个月）

### 1. Swing 召回算法

**优先级**：🟡 中

**描述**：实现 Swing 召回算法，改进版 Item-CF，通过惩罚热门物品挖掘更有说服力的相似物品。

**适用场景**：电商场景，需要挖掘长尾物品相似关系

**实现计划**：
- 在 `ItemBasedCF` 基础上扩展
- 实现热门物品惩罚机制
- 支持 Swing 相似度计算
- 创建 `SwingRecall` 或扩展 `I2IRecall`

**接口设计**：
```go
// recall/swing_recall.go
type SwingRecall struct {
    Store CFStore
    TopK  int
    // Swing 特有参数
    Alpha float64 // 热门物品惩罚系数
}
```

---

### 2. 复购召回

**优先级**：🟡 中

**描述**：实现复购召回，针对快消品、外卖等场景，直接召回用户历史购买过的物品。

**适用场景**：快消品、外卖、高频复购场景

**实现计划**：
- 扩展 `UserHistory` 支持购买历史
- 或创建新的 `RepurchaseRecall`
- 支持购买频率和时间权重

**接口设计**：
```go
// recall/repurchase_recall.go
type RepurchaseRecall struct {
    Store RepurchaseStore
    TopK  int
    // 复购特有参数
    MinPurchaseCount int // 最小购买次数
    TimeWindow       int64 // 时间窗口
}
```

---

### 3. 自定义服务客户端

**优先级**：🟡 中

**描述**：实现 `ServiceTypeCustom` 支持，允许用户接入自定义的 ML 服务。

**位置**：`service/factory.go:47`

**实现计划**：
- 定义自定义服务配置接口
- 实现通用的 HTTP/gRPC 客户端
- 支持自定义请求/响应格式转换

---

### 4. 监控和 Metrics 系统

**优先级**：🟡 中

**描述**：实现 Pipeline 级别的监控和 Metrics 收集系统。

**功能**：
- Pipeline 执行时间统计（各 Node 耗时）
- QPS、延迟、错误率等指标
- Prometheus/StatsD 集成
- Dashboard 支持

**接口设计**：
```go
// pkg/metrics/metrics.go
type MetricsCollector interface {
    RecordPipelineDuration(duration time.Duration, pipelineName string)
    RecordNodeDuration(duration time.Duration, nodeName string, nodeKind string)
    RecordRecallCount(source string, count int)
    RecordRankCount(model string, count int)
    RecordError(nodeName string, err error)
}
```

---

### 5. 结构化日志系统

**优先级**：🟡 中

**描述**：实现结构化日志系统，支持日志级别、采样、追踪等功能。

**功能**：
- 结构化日志接口（JSON 格式）
- 日志级别管理
- 日志采样和聚合
- 分布式追踪（Trace ID）
- 日志查询和分析支持

**接口设计**：
```go
// pkg/log/logger.go
type Logger interface {
    Info(ctx context.Context, msg string, fields ...Field)
    Error(ctx context.Context, msg string, fields ...Field)
    Debug(ctx context.Context, msg string, fields ...Field)
}
```

---

## 🚀 中期规划（3-6 个月）

### 6. 多兴趣模型 (MIND/SDM)

**优先级**：🟡 中

**描述**：实现多兴趣模型召回，一个用户生成多个向量表示不同维度的兴趣，匹配更精准。

**适用场景**：用户兴趣多样化的场景，需要捕捉用户多个兴趣维度

**实现计划**：
- 扩展 `TwoTowerRecall` 支持多兴趣向量
- 或创建新的 `MultiInterestRecall`
- 支持兴趣向量聚合和匹配
- Python 训练脚本（MIND/SDM）

**接口设计**：
```go
// recall/multi_interest_recall.go
type MultiInterestRecall struct {
    FeatureService   feature.FeatureService
    UserTowerService  core.MLService
    VectorService     core.VectorService
    TopK              int
    NumInterests      int // 兴趣向量数量
    InterestAggMethod string // 聚合方法：max / mean / weighted
}
```

---

### 7. 地理位置召回 (LBS)

**优先级**：🟢 低

**描述**：实现地理位置召回，基于用户地理位置召回附近的物品。

**适用场景**：本地生活服务、地理位置相关的推荐、O2O 场景

**实现计划**：
- 定义地理位置存储接口
- 实现距离计算（Haversine 公式）
- 支持地理围栏和距离排序

**接口设计**：
```go
// recall/lbs_recall.go
type LBSRecall struct {
    Store LBSStore
    TopK  int
    MaxDistance float64 // 最大距离（公里）
    UserLocation *Location // 用户位置
}

type Location struct {
    Latitude  float64 // 纬度
    Longitude float64 // 经度
}
```

---

### 8. 数据反馈和收集系统

**优先级**：🟡 中

**描述**：实现完整的反馈数据收集和处理系统。

**功能**：
- 反馈数据收集（点击、曝光、购买等）
- 反馈数据存储（Kafka、Redis、数据库）
- 反馈数据处理 Pipeline
- 用户行为回放
- 模型训练数据生成

**接口设计**：
```go
// feedback/collector.go
type FeedbackCollector interface {
    RecordImpression(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) error
    RecordClick(ctx context.Context, rctx *core.RecommendContext, itemID string) error
    RecordConversion(ctx context.Context, rctx *core.RecommendContext, itemID string) error
}
```

---

### 9. 完整的 A/B 测试框架

**优先级**：🟡 中

**描述**：实现完整的 A/B 测试框架，支持实验配置、统计分析等功能。

**功能**：
- 实验配置管理（实验分组、流量分配）
- 实验指标统计（CTR、CVR、时长等）
- 实验报告生成
- 实验自动停止和切换
- 实验效果显著性检验

**接口设计**：
```go
// experiment/manager.go
type ExperimentManager interface {
    GetExperiment(ctx context.Context, userID string, scene string) (*Experiment, error)
    RecordMetric(ctx context.Context, experimentID string, metric string, value float64) error
    GenerateReport(ctx context.Context, experimentID string) (*ExperimentReport, error)
}
```

---

## 🔮 长期规划（6-12 个月）

### 10. 知识图谱召回

**优先级**：🟢 低

**描述**：实现知识图谱召回，利用实体关系（如：演员A -> 出演 -> 电影B）来挖掘关联。

**适用场景**：内容推荐（电影、音乐、书籍）、知识密集型推荐

**实现计划**：
- 定义知识图谱存储接口
- 实现图遍历算法
- 支持多跳关系召回

**接口设计**：
```go
// recall/kg_recall.go
type KnowledgeGraphRecall struct {
    Store KGStore
    TopK  int
    MaxHops int // 最大跳数
}

type KGStore interface {
    GetRelatedItems(ctx context.Context, itemID string, relation string, maxHops int) ([]string, error)
    GetEntityRelations(ctx context.Context, entityID string) ([]Relation, error)
}
```

---

### 11. 实时特征计算引擎

**优先级**：🟢 低

**描述**：实现实时特征计算引擎，支持复杂特征计算和流式处理。

**功能**：
- 实时特征计算 DSL
- 流式特征处理
- 特征依赖管理
- 特征版本控制

---

### 12. 模型热更新系统

**优先级**：🟢 低

**描述**：实现模型热更新系统，支持模型版本管理和无缝切换。

**功能**：
- 模型版本管理
- 模型热加载
- 模型 A/B 测试
- 模型回滚机制

---

### 13. 分布式推荐服务

**优先级**：🟢 低

**描述**：实现分布式推荐服务，支持水平扩展和负载均衡。

**功能**：
- 服务注册与发现
- 负载均衡
- 分布式缓存
- 服务监控和告警

---

## 📋 已完成的功能

以下功能已经实现，不再列入规划：

### 架构与基础设施

- ✅ DDD 架构设计（所有接口定义在 `core` 包）
- ✅ 接口统一（`core.Store`、`core.MLService`、`core.VectorService`）
- ✅ 错误处理统一（`core.DomainError`）
- ✅ Milvus 原生 string ID 支持
- ✅ Pipeline Hook 机制
- ✅ 用户画像系统
- ✅ Python ML 服务集成

### 召回算法（13个已实现，72.2% 覆盖率）

#### 基于行为的召回（3/4）
- ✅ User-CF（`U2IRecall`）- 用户协同过滤
- ✅ Item-CF（`I2IRecall`）- 物品协同过滤
- ✅ 矩阵分解 MF/ALS（`MFRecall`）- 隐向量召回

#### 向量化/深度学习召回（3/4）
- ✅ 双塔模型（`TwoTowerRecall`）- 工业标准召回
- ✅ DSSM（`DSSMRecall`）- 深度语义匹配
- ✅ YouTube DNN（`YouTubeDNNRecall`）- 视频流召回

#### 基于内容的召回（2/3）
- ✅ Content（`ContentRecall`）- 基于物品特征
- ✅ Word2Vec/BERT（`Word2VecRecall`, `BERTRecall`）- 文本语义召回

#### 基于图的召回（1/1）
- ✅ GraphSAGE/Node2vec（`GraphRecall`）- 社交推荐

#### 其他策略召回（1/3）
- ✅ 热门/Trending（`Hot`）- 兜底策略

#### 其他已实现（3/3）
- ✅ Embedding 向量检索（`EmbRecall`/`ANN`）- 通用向量检索
- ✅ 用户历史召回（`UserHistory`）- 基于行为历史
- ✅ RPC 召回（`RPCRecall`）- 外部服务召回

### 其他核心功能

- ✅ 过滤系统（黑名单、用户拉黑、已曝光）
- ✅ 特征服务（特征注入、缓存、监控、Feast 集成）
- ✅ 排序模型（LR、DNN、DIN、Wide&Deep、TwoTower、RPC）
- ✅ 重排系统（多样性重排）

---

## 🔗 相关文档

- [TODO 清单](./TODO.md) - 详细的问题清单
- [架构设计文档](./ARCHITECTURE.md) - 架构设计说明
- [接口与实现完整分析](./INTERFACES_AND_IMPLEMENTATIONS.md) - 接口使用手册
