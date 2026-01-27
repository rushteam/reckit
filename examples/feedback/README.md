# 反馈收集系统示例

生产级反馈收集系统示例，使用 Kafka（franzgo）进行异步非阻塞的数据采集。

**注意**：这是一个示例实现，位于 `examples/feedback` 目录，使用独立的 `go.mod` 文件，不侵入主项目。

## 架构设计

```
推荐服务 (Golang)
    ↓ (异步非阻塞)
Kafka Collector (轻量级 Producer)
    ↓ (批量发送)
Kafka (消息队列)
    ↓ (流处理)
Flink (实时处理)
    ├─→ 用户画像更新 (Redis/数据库)
    ├─→ 训练数据生成 (HDFS/对象存储)
    └─→ 实时指标统计 (ClickHouse/时序数据库)
```

## 快速开始

### 1. 安装依赖

在 `examples` 目录下运行：

```bash
cd examples
go mod tidy
```

这会自动安装所有依赖，包括：
- `github.com/rushteam/reckit`（主项目，通过 replace 指向本地）
- `github.com/twmb/franz-go`（Kafka 客户端）

### 2. 启动 Kafka

确保 Kafka 服务已启动，默认地址为 `localhost:9092`。

### 3. 运行示例

在 `examples` 目录下运行：

```bash
cd examples
go run feedback/main.go
```

或者从项目根目录：

```bash
cd examples/feedback
go run main.go
```

## 使用说明

### 创建 Kafka Collector

```go
import "github.com/rushteam/reckit/examples/feedback"

collector, err := feedback.NewKafkaCollector(feedback.KafkaCollectorConfig{
    Brokers:       []string{"localhost:9092"},
    Topic:         "feedback-topic",
    BatchSize:     100,
    FlushInterval: 1 * time.Second,
    ClientID:      "reckit-feedback",
    RequiredAcks:  1,
    Compression:   "gzip",
    Idempotent:    false,
    MaxRetries:    3,
})
```

### 集成到 Pipeline

```go
import (
    "github.com/rushteam/reckit/examples/feedback"
    "github.com/rushteam/reckit/pipeline"
)

feedbackHook := feedback.NewFeedbackHook(collector)

p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{...},
    Hooks: []pipeline.PipelineHook{
        feedbackHook, // 自动记录曝光
    },
}
```

### 手动记录反馈

```go
// 记录点击
collector.RecordClick(ctx, rctx, itemID, position)

// 记录转化
collector.RecordConversion(ctx, rctx, itemID, map[string]any{
    "amount":   99.0,
    "order_id": "order_123",
})
```

## 配置说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `Brokers` | `[]string` | Kafka Broker 地址列表 | 必填 |
| `Topic` | `string` | Kafka Topic | 必填 |
| `BatchSize` | `int` | 批量大小（建议 100-1000） | 100 |
| `FlushInterval` | `time.Duration` | 刷新间隔（建议 1-5 秒） | 1 秒 |
| `ClientID` | `string` | 客户端 ID | "reckit-feedback-collector" |
| `RequiredAcks` | `int16` | ACK 数量（1=leader, -1=all） | 1 |
| `Compression` | `string` | 压缩类型（gzip/snappy/lz4/zstd） | "" |
| `Idempotent` | `bool` | 是否启用幂等性 | false |
| `MaxRetries` | `int` | 最大重试次数 | 3 |

## 反馈事件格式

```json
{
  "user_id": "user_123",
  "item_id": "item_456",
  "scene": "feed",
  "type": "impression",
  "timestamp": 1704067200,
  "position": 0,
  "score": 0.85,
  "labels": {
    "recall_source": "recall.hot",
    "recall_type": "hot"
  },
  "extras": {}
}
```

## 独立模块说明

本示例使用独立的 `go.mod` 文件（位于 `examples/go.mod`），具有以下优势：

1. **依赖隔离**：示例项目的依赖（如 `franz-go`）不会影响主项目
2. **独立版本管理**：示例可以独立管理依赖版本
3. **易于测试**：可以独立运行和测试示例代码
4. **不侵入主项目**：主项目的 `go.mod` 保持简洁

### 集成到主项目

如果需要将反馈模块集成到主项目中：

1. 将 `examples/feedback` 目录复制到项目根目录，重命名为 `feedback`
2. 更新导入路径：`github.com/rushteam/reckit/feedback`
3. 在主项目的 `go.mod` 中添加 `github.com/twmb/franz-go` 依赖
4. 在 `core/errors.go` 中添加 `ModuleFeedback` 常量

## 性能建议

1. **批量大小**：根据 QPS 调整，高 QPS 场景建议 500-1000
2. **刷新间隔**：根据实时性要求调整，建议 1-5 秒
3. **压缩**：高吞吐场景建议启用 gzip 或 zstd
4. **幂等性**：需要精确一次语义时启用，但会降低性能
5. **ACK 级别**：1=性能优先，-1=可靠性优先

## 生产环境注意事项

1. **监控**：监控发送失败率、缓冲大小、延迟等指标
2. **降级**：Kafka 不可用时考虑本地缓存或降级策略
3. **重试**：生产环境应该实现更完善的重试机制
4. **本地缓存**：可以考虑实现本地缓存，防止数据丢失
