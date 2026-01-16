# Feast Feature Store 封装

Feast 是一个开源的 Feature Store，用于机器学习的特征管理。本包提供了对 Feast 的封装，遵循 DDD 设计原则，保持高内聚低耦合。

## 设计原则

### 1. 接口抽象（Domain Layer）
- `Client` 接口：定义 Feast 客户端抽象，不直接依赖 Python SDK
- `ClientFactory` 接口：支持依赖注入

### 2. 实现层（Infrastructure Layer）
- `HTTPClient`：HTTP 客户端实现（使用 Feast Feature Server HTTP API，自定义实现）
- `GrpcClient`：gRPC 客户端实现（使用官方 SDK `github.com/feast-dev/feast/sdk/go`）
- `FeatureServiceAdapter`：将 Feast Client 适配为 `feature.FeatureService` 接口

### 3. 高内聚低耦合
- 通过接口抽象避免直接依赖 Python SDK
- 支持依赖注入，可以替换实现
- 使用标准 HTTP/gRPC 协议，不依赖特定 SDK
- 支持官方 SDK 和自定义实现，用户可根据需求选择

## 核心接口

### Client

Feast 客户端接口，提供特征获取、物化等功能。

```go
type Client interface {
    // 获取在线特征（用于实时预测）
    GetOnlineFeatures(ctx context.Context, req *GetOnlineFeaturesRequest) (*GetOnlineFeaturesResponse, error)
    
    // 获取历史特征（用于训练数据）
    GetHistoricalFeatures(ctx context.Context, req *GetHistoricalFeaturesRequest) (*GetHistoricalFeaturesResponse, error)
    
    // 将特征物化到在线存储
    Materialize(ctx context.Context, req *MaterializeRequest) error
    
    // 列出所有可用的特征
    ListFeatures(ctx context.Context) ([]Feature, error)
    
    // 获取特征服务信息
    GetFeatureService(ctx context.Context) (*FeatureServiceInfo, error)
    
    // 关闭客户端连接
    Close() error
}
```

## 使用方式

### 1. 创建 Feast 客户端

本包提供两种客户端实现方式：

#### 方式 A：使用 HTTP 客户端（自定义实现，推荐用于简单场景）

```go
import "github.com/rushteam/reckit/feast"

// 创建 HTTP 客户端
client, err := feast.NewHTTPClient(
    "http://localhost:6566", // Feast Feature Server 端点
    "my_project",            // 项目名称
)
if err != nil {
    log.Fatal(err)
}
defer client.Close()
```

#### 方式 B：使用 gRPC 客户端（官方 SDK，推荐用于生产环境）

```go
import "github.com/rushteam/reckit/feast"

// 创建 gRPC 客户端（使用官方 SDK）
client, err := feast.NewGrpcClient(
    "localhost",  // 主机地址
    6565,         // gRPC 端口（默认 6565）
    "my_project", // 项目名称
)
if err != nil {
    log.Fatal(err)
}
defer client.Close()
```

#### 方式 C：使用工厂模式（自动选择实现）

```go
import "github.com/rushteam/reckit/feast"

factory := &feast.DefaultClientFactory{}

// 使用 HTTP 客户端
httpClient, err := factory.NewClient(ctx, "http://localhost:6566", "my_project")

// 使用 gRPC 客户端（官方 SDK）
grpcClient, err := factory.NewClient(
    ctx, 
    "localhost:6565", 
    "my_project",
    feast.WithGRPC(), // 指定使用 gRPC
)
```

**选择建议：**
- **HTTP 客户端**：简单易用，支持完整功能（包括历史特征、物化等）
- **gRPC 客户端**：性能更好，使用官方 SDK，但仅支持在线特征获取

### 2. 获取在线特征

```go
// 构建请求
req := &feast.GetOnlineFeaturesRequest{
    Features: []string{
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips",
    },
    EntityRows: []map[string]interface{}{
        {"driver_id": 1001},
        {"driver_id": 1002},
    },
}

// 获取特征
resp, err := client.GetOnlineFeatures(ctx, req)
if err != nil {
    log.Fatal(err)
}

// 使用特征
for _, fv := range resp.FeatureVectors {
    fmt.Printf("Features: %+v\n", fv.Values)
}
```

### 3. 获取历史特征（用于训练）

```go
// 构建请求
req := &feast.GetHistoricalFeaturesRequest{
    EntityDF: []map[string]interface{}{
        {
            "driver_id":       1001,
            "event_timestamp": "2021-04-12T10:59:42Z",
        },
        {
            "driver_id":       1002,
            "event_timestamp": "2021-04-12T08:12:10Z",
        },
    },
    Features: []string{
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
    },
}

// 获取历史特征
resp, err := client.GetHistoricalFeatures(ctx, req)
if err != nil {
    log.Fatal(err)
}

// 使用历史特征（训练数据）
for _, row := range resp.DataFrame {
    fmt.Printf("Row: %+v\n", row)
}
```

### 4. 物化特征（离线到在线）

```go
// 构建请求
req := &feast.MaterializeRequest{
    StartTime: time.Now().Add(-24 * time.Hour),
    EndTime:   time.Now(),
    FeatureViews: []string{
        "driver_hourly_stats",
    },
}

// 物化特征
err := client.Materialize(ctx, req)
if err != nil {
    log.Fatal(err)
}
```

### 5. 适配到 FeatureService

```go
import (
    "github.com/rushteam/reckit/feast"
    "github.com/rushteam/reckit/feature"
)

// 创建 Feast 客户端
feastClient, _ := feast.NewHTTPClient("http://localhost:6566", "my_project")

// 创建适配器
adapter := feast.NewFeatureServiceAdapter(feastClient, &feast.FeatureMapping{
    UserFeatures: []string{
        "user_stats:age",
        "user_stats:gender",
        "user_stats:city",
    },
    ItemFeatures: []string{
        "item_stats:price",
        "item_stats:category",
        "item_stats:rating",
    },
    RealtimeFeatures: []string{
        "interaction:click_count",
        "interaction:view_count",
    },
    UserEntityKey: "user_id",
    ItemEntityKey: "item_id",
})

// 使用适配器作为 FeatureService
var featureService feature.FeatureService = adapter

// 获取用户特征
features, err := featureService.GetUserFeatures(ctx, 1001)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("User features: %+v\n", features)
```

## 集成到 Pipeline

```go
import (
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/feature"
    "github.com/rushteam/reckit/feast"
    "github.com/rushteam/reckit/pipeline"
    "github.com/rushteam/reckit/recall"
)

func main() {
    // 1. 创建 Feast 客户端
    feastClient, _ := feast.NewHTTPClient("http://localhost:6566", "my_project")
    
    // 2. 创建适配器
    adapter := feast.NewFeatureServiceAdapter(feastClient, &feast.FeatureMapping{
        UserFeatures: []string{"user_stats:age", "user_stats:gender"},
        ItemFeatures: []string{"item_stats:price", "item_stats:category"},
    })
    
    // 3. 创建特征注入节点
    enrichNode := &feature.EnrichNode{
        FeatureService:     adapter,
        UserFeaturePrefix:  "user_",
        ItemFeaturePrefix:  "item_",
        CrossFeaturePrefix: "cross_",
    }
    
    // 4. 构建 Pipeline
    p := &pipeline.Pipeline{
        Nodes: []pipeline.Node{
            &recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}},
            enrichNode,
            // ... 其他节点
        },
    }
    
    // 5. 运行 Pipeline
    rctx := &core.RecommendContext{
        UserID: "1001",
        Scene:  "feed",
    }
    items, err := p.Run(ctx, rctx, nil)
    // ...
}
```

## 配置选项

### HTTP 客户端选项

```go
client, err := feast.NewHTTPClient(
    "http://localhost:6566",
    "my_project",
    feast.WithFeastTimeout(10 * time.Second), // 设置超时
    feast.WithFeastAuth(&feast.AuthConfig{    // 设置认证
        Type:   "bearer",
        Token:  "your-token",
    }),
)
```

### gRPC 客户端选项（官方 SDK）

```go
client, err := feast.NewGrpcClient(
    "localhost",
    6565,
    "my_project",
    feast.WithFeastTimeout(10 * time.Second), // 设置超时
    feast.WithFeastAuth(&feast.AuthConfig{    // 设置认证
        Type:   "static", // gRPC 支持 static token
        Token:  "your-token",
    }),
)
```

### 认证配置

支持的认证类型：
- `basic`：HTTP Basic 认证（仅 HTTP 客户端）
- `bearer`：Bearer Token 认证（仅 HTTP 客户端）
- `api_key`：API Key 认证（仅 HTTP 客户端）
- `static`：静态 Token 认证（仅 gRPC 客户端，使用官方 SDK）

```go
// HTTP 客户端认证
httpAuth := &feast.AuthConfig{
    Type:     "bearer",
    Token:    "your-token",
}

// gRPC 客户端认证（官方 SDK）
grpcAuth := &feast.AuthConfig{
    Type:     "static",
    Token:    "your-token",
}
```

## 架构设计

```
┌─────────────────────────────────────────┐
│         FeatureService (接口)            │
│  (github.com/rushteam/reckit/feature 包)                    │
└─────────────────┬───────────────────────┘
                  │
                  │ 适配
                  ▼
┌─────────────────────────────────────────┐
│    FeatureServiceAdapter                │
│    (feast/adapter.go)                   │
└─────────────────┬───────────────────────┘
                  │
                  │ 调用
                  ▼
┌─────────────────────────────────────────┐
│         Client (接口)                    │
│    (feast/client.go) - 领域层抽象         │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                    │
        │ 实现（基础设施层）    │
        ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│   HTTPClient     │  │   GrpcClient     │
│ (自定义实现)      │  │ (官方 SDK)       │
│ http_client.go   │  │ grpc_client.go   │
└────────┬─────────┘  └────────┬─────────┘
         │                      │
         │ HTTP                 │ gRPC
         ▼                      ▼
┌──────────────────┐  ┌──────────────────┐
│  Feast Server    │  │  Feast Server    │
│  (HTTP API)      │  │  (gRPC API)      │
│  :6566           │  │  :6565           │
└──────────────────┘  └──────────────────┘
```

### DDD 分层说明

1. **领域层（Domain Layer）**：
   - `Client` 接口：定义 Feast 客户端的领域抽象
   - `FeatureServiceAdapter`：适配器，连接领域和基础设施

2. **基础设施层（Infrastructure Layer）**：
   - `HTTPClient`：自定义 HTTP 实现
   - `GrpcClient`：基于官方 SDK 的 gRPC 实现

3. **应用层（Application Layer）**：
   - `DefaultClientFactory`：工厂模式，统一创建客户端

## 依赖说明

### 官方 SDK 依赖

使用 gRPC 客户端（`GrpcClient`）需要添加官方 SDK 依赖：

```bash
go get github.com/feast-dev/feast/sdk/go@latest
```

### 可选依赖

- `google.golang.org/grpc`：gRPC 客户端库（官方 SDK 已包含）

## 完整示例

### 使用官方 SDK gRPC 客户端

```go
package main

import (
    "context"
    "log"
    "time"

    "github.com/rushteam/reckit/feast"
    "github.com/rushteam/reckit/feature"
)

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()

    // 方式 1：直接创建 gRPC 客户端（官方 SDK）
    client, err := feast.NewGrpcClient("localhost", 6565, "my_project")
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // 方式 2：使用统一入口（推荐）
    // client, err := feast.NewClient("localhost:6565", "my_project", feast.WithGRPC())

    // 创建特征映射
    mapping := &feast.FeatureMapping{
        UserFeatures: []string{
            "user_stats:age",
            "user_stats:gender",
        },
        ItemFeatures: []string{
            "item_stats:price",
            "item_stats:category",
        },
        UserEntityKey: "user_id",
        ItemEntityKey: "item_id",
    }

    // 创建适配器
    adapter := feast.NewFeatureServiceAdapter(client, mapping)

    // 使用适配器作为 FeatureService
    var featureService feature.FeatureService = adapter

    // 获取用户特征
    features, err := featureService.GetUserFeatures(ctx, "user_123")
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("用户特征: %+v", features)
}
```

### 选择建议

| 特性 | HTTP 客户端 | gRPC 客户端（官方 SDK） |
|------|------------|----------------------|
| 性能 | 良好 | 优秀 |
| 延迟 | 较高 | 低 |
| 功能完整性 | 完整（支持所有操作） | 主要支持在线特征 |
| 依赖 | 无外部依赖 | 需要官方 SDK |
| 适用场景 | 开发/测试/简单场景 | 生产环境/高性能需求 |

## 参考

- [Feast 官方文档](https://docs.feast.dev/)
- [Feast GitHub](https://github.com/feast-dev/feast)
- [Feast Go SDK](https://pkg.go.dev/github.com/feast-dev/feast/sdk/go)
- [Feast Feature Server API](https://docs.feast.dev/reference/api/feast-online-serving-api)
