# Feast 扩展包

Feast 特征存储客户端实现，所有代码在同一目录下，一个 go.mod。

## 安装

```bash
go get github.com/rushteam/reckit/ext/feast
```

## 使用

**HTTP 客户端：**

```go
import (
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/ext/feast"
)

client, _ := feast.NewHTTPClient("http://localhost:6566", "my_project")
mapping := &feast.FeatureMapping{
    UserFeatures: []string{"user_stats:age", "user_stats:gender"},
    ItemFeatures: []string{"item_stats:price", "item_stats:category"},
}
fs := feast.NewFeatureServiceAdapter(client, mapping)
var _ core.FeatureService = fs
```

**gRPC 客户端（推荐生产环境）：**

```go
client, _ := feast.NewGrpcClient("localhost", 6565, "my_project")
mapping := &feast.FeatureMapping{...}
fs := feast.NewFeatureServiceAdapter(client, mapping)
var _ core.FeatureService = fs
```

## 依赖

- `github.com/rushteam/reckit` - 核心包
- `github.com/feast-dev/feast/sdk/go` - Feast 官方 SDK（gRPC 使用）
