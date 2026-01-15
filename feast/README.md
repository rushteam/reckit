# Feast Feature Store 封装

Feast 是一个开源的 Feature Store，用于机器学习的特征管理。本包提供了对 Feast 的封装，遵循 DDD 设计原则，保持高内聚低耦合。

## 设计原则

### 1. 接口抽象（Domain Layer）
- `Client` 接口：定义 Feast 客户端抽象，不直接依赖 Python SDK
- `ClientFactory` 接口：支持依赖注入

### 2. 实现层（Infrastructure Layer）
- `HTTPClient`：HTTP 客户端实现（使用 Feast Feature Server HTTP API）
- `FeatureServiceAdapter`：将 Feast Client 适配为 `feature.FeatureService` 接口

### 3. 高内聚低耦合
- 通过接口抽象避免直接依赖 Python SDK
- 支持依赖注入，可以替换实现
- 使用标准 HTTP/gRPC 协议，不依赖特定 SDK

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

```go
import "reckit/feast"

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
    "reckit/feast"
    "reckit/feature"
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
    "reckit/core"
    "reckit/feature"
    "reckit/feast"
    "reckit/pipeline"
    "reckit/recall"
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
    // 可以添加更多配置选项（待实现）
)
```

### 认证配置

```go
auth := &feast.AuthConfig{
    Type:     "bearer",
    Token:    "your-token",
}

// 在创建客户端时传入认证信息（待实现）
```

## 架构设计

```
┌─────────────────────────────────────────┐
│         FeatureService (接口)            │
│  (reckit/feature 包)                    │
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
│    (feast/client.go)                    │
└─────────────────┬───────────────────────┘
                  │
                  │ 实现
                  ▼
┌─────────────────────────────────────────┐
│         HTTPClient                      │
│    (feast/http_client.go)               │
└─────────────────┬───────────────────────┘
                  │
                  │ HTTP/gRPC
                  ▼
┌─────────────────────────────────────────┐
│      Feast Feature Server               │
│    (http://localhost:6566)              │
└─────────────────────────────────────────┘
```

## 参考

- [Feast 官方文档](https://docs.feast.dev/)
- [Feast GitHub](https://github.com/feast-dev/feast)
- [Feast Feature Server API](https://docs.feast.dev/reference/api/feast-online-serving-api)
