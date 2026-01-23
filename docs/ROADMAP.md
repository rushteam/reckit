# Reckit 开发路线图

本文档描述 Reckit 项目的未来发展规划和待实现功能。

---

## 🎯 短期规划（1-3 个月）

### 1. 自定义服务客户端

**优先级**：🟡 中

**描述**：实现 `ServiceTypeCustom` 支持，允许用户接入自定义的 ML 服务。

**位置**：`service/factory.go:47`

**实现计划**：
- 定义自定义服务配置接口
- 实现通用的 HTTP/gRPC 客户端
- 支持自定义请求/响应格式转换

---

### 2. 监控和 Metrics 系统

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

### 3. 结构化日志系统

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

### 4. 数据反馈和收集系统

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

### 5. 完整的 A/B 测试框架

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

### 6. 实时特征计算引擎

**优先级**：🟢 低

**描述**：实现实时特征计算引擎，支持复杂特征计算和流式处理。

**功能**：
- 实时特征计算 DSL
- 流式特征处理
- 特征依赖管理
- 特征版本控制

---

### 7. 模型热更新系统

**优先级**：🟢 低

**描述**：实现模型热更新系统，支持模型版本管理和无缝切换。

**功能**：
- 模型版本管理
- 模型热加载
- 模型 A/B 测试
- 模型回滚机制

---

### 8. 分布式推荐服务

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

- ✅ DDD 架构设计（所有接口定义在 `core` 包）
- ✅ 接口统一（`core.Store`、`core.MLService`、`core.VectorService`）
- ✅ 错误处理统一（`core.DomainError`）
- ✅ Milvus 原生 string ID 支持
- ✅ 召回算法（User-CF、Item-CF、MF、ANN、Content、热门等）
- ✅ 过滤系统（黑名单、用户拉黑、已曝光）
- ✅ 特征服务（特征注入、缓存、监控、Feast 集成）
- ✅ 排序模型（LR、DNN、DIN、Wide&Deep、TwoTower、RPC）
- ✅ 重排系统（多样性重排）
- ✅ Pipeline Hook 机制
- ✅ 用户画像系统
- ✅ Python ML 服务集成

---

## 🔗 相关文档

- [TODO 清单](./TODO.md) - 详细的问题清单
- [架构设计文档](./ARCHITECTURE.md) - 架构设计说明
- [接口与实现完整分析](./INTERFACES_AND_IMPLEMENTATIONS.md) - 接口使用手册
