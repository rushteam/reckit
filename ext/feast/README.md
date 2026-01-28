# Feast 扩展包

Feast 特征存储客户端实现，位于扩展包中，**一个 go.mod 管理整个扩展**。子目录 `common`、`http`、`grpc` 为子包，无需单独 go.mod。

## 安装

```bash
go get github.com/rushteam/reckit/ext/feast
```

安装后可按需引用子包：`ext/feast/http`、`ext/feast/grpc`、`ext/feast/common`。

## 子包说明

| 子包 | 说明 |
|------|------|
| `common` | 共享类型与接口（`Client`、`FeatureMapping` 等） |
| `http` | HTTP 客户端 + 适配器（实现 `core.FeatureService`） |
| `grpc` | gRPC 客户端 + 适配器（基于官方 SDK，推荐生产环境） |

## HTTP 使用示例

```go
import (
    "github.com/rushteam/reckit/core"
    feasthttp "github.com/rushteam/reckit/ext/feast/http"
    feastcommon "github.com/rushteam/reckit/ext/feast/common"
)

feastClient, _ := feasthttp.NewClient("http://localhost:6566", "my_project")
mapping := &feastcommon.FeatureMapping{
    UserFeatures: []string{"user_stats:age", "user_stats:gender"},
    ItemFeatures: []string{"item_stats:price", "item_stats:category"},
}
featureService := feasthttp.NewFeatureServiceAdapter(feastClient, mapping)
var fs core.FeatureService = featureService
```

## gRPC 使用示例

```go
import (
    "github.com/rushteam/reckit/core"
    feastgrpc "github.com/rushteam/reckit/ext/feast/grpc"
    feastcommon "github.com/rushteam/reckit/ext/feast/common"
)

feastClient, _ := feastgrpc.NewClient("localhost", 6565, "my_project")
mapping := &feastcommon.FeatureMapping{...}
featureService := feastgrpc.NewFeatureServiceAdapter(feastClient, mapping)
var fs core.FeatureService = featureService
```

## 依赖

- `github.com/rushteam/reckit` - 核心包（仅接口）
- `github.com/feast-dev/feast/sdk/go` - Feast 官方 SDK（grpc 子包使用）
