# TODO - 待解决问题清单

本文档列出项目中待解决的问题和未实现的功能。

---

## 🔴 高优先级

### 1. 自定义服务客户端未实现

**位置**：`service/factory.go:47`

**问题**：
```go
case ServiceTypeCustom:
    // TODO: 实现自定义服务客户端
    return nil, fmt.Errorf("custom service not implemented")
```

**影响**：
- 功能不完整
- 用户可能期望支持自定义服务

**建议**：
- 实现自定义服务客户端
- 或明确标记为"暂不支持"，并提供替代方案

**优先级**：🟡 中

---

## 🟡 中优先级

### 2. 监控和 Metrics 系统

**现状**：只有特征监控（MemoryFeatureMonitor），缺少 Pipeline 级别的监控

**缺失内容**：
- Pipeline 执行时间统计（各 Node 耗时）
- QPS、延迟、错误率等指标
- Prometheus/StatsD 集成
- 告警系统
- Dashboard 支持

**建议实现**：
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

**优先级**：🟡 中

---

### 3. 结构化日志系统

**现状**：只有示例代码中的简单 `fmt.Printf`，缺少结构化日志

**缺失内容**：
- 结构化日志接口（JSON 格式）
- 日志级别管理
- 日志采样和聚合
- 分布式追踪（Trace ID）
- 日志查询和分析支持

**建议实现**：
```go
// pkg/log/logger.go
type Logger interface {
    Info(ctx context.Context, msg string, fields ...Field)
    Error(ctx context.Context, msg string, fields ...Field)
    Debug(ctx context.Context, msg string, fields ...Field)
}
```

**优先级**：🟡 中

---

### 4. 数据反馈和收集系统

**现状**：有 Label 支持，但缺少完整的反馈收集和处理系统

**缺失内容**：
- 反馈数据收集（点击、曝光、购买等）
- 反馈数据存储（Kafka、Redis、数据库）
- 反馈数据处理 Pipeline
- 用户行为回放
- 模型训练数据生成

**建议实现**：
```go
// feedback/collector.go
type FeedbackCollector interface {
    RecordImpression(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) error
    RecordClick(ctx context.Context, rctx *core.RecommendContext, itemID string) error
    RecordConversion(ctx context.Context, rctx *core.RecommendContext, itemID string) error
}
```

**优先级**：🟡 中

---

### 5. 完整的 A/B 测试框架

**现状**：有实验桶（Bucket）支持，但缺少实验配置、统计分析等功能

**缺失内容**：
- 实验配置管理（实验分组、流量分配）
- 实验指标统计（CTR、CVR、时长等）
- 实验报告生成
- 实验自动停止和切换
- 实验效果显著性检验

**优先级**：🟡 中

---

## 🟢 低优先级（可选优化）

### 6. 文档更新

**问题**：
- 部分文档可能有过时信息
- 示例代码风格可以统一

**建议**：
- 定期检查文档是否与代码一致
- 统一示例代码风格

**优先级**：🟢 低

---

## 📋 已解决的问题

以下问题已经解决，不再需要处理：

- ✅ **FeatureStore 接口未实现** - 已删除未使用的接口
- ✅ **Milvus ID 转换冲突** - 已改为使用 Milvus 原生 string ID 支持
- ✅ **错误处理不一致** - 已统一使用 `core.DomainError`
- ✅ **接口定义位置不符合 DDD** - 已将所有接口提升到 `core` 包
- ✅ **类型别名和向后兼容** - 已移除所有类型别名，统一使用 `core` 包接口

---

## 📝 说明

- **优先级**：🔴 高、🟡 中、🟢 低
- **状态**：待解决、进行中、已完成
- 本文档会定期更新，已解决的问题会移到"已解决的问题"部分
