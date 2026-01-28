# Feast gRPC 子包

Feast gRPC 客户端实现，为 `ext/feast` 扩展的子包（无单独 go.mod）。

## 安装

安装整个 Feast 扩展即可使用本子包：

```bash
go get github.com/rushteam/reckit/ext/feast
```

## 使用

```go
import (
    "github.com/rushteam/reckit/core"
    feastgrpc "github.com/rushteam/reckit/ext/feast/grpc"
    feastcommon "github.com/rushteam/reckit/ext/feast/common"
)

// 1. 创建 Feast gRPC 客户端
feastClient, err := feastgrpc.NewClient("localhost", 6565, "my_project")
if err != nil {
    log.Fatal(err)
}
defer feastClient.Close()

// 2. 创建特征映射配置
mapping := &feastcommon.FeatureMapping{
    UserFeatures: []string{"user_stats:age", "user_stats:gender"},
    ItemFeatures: []string{"item_stats:price", "item_stats:category"},
    UserEntityKey: "user_id",
    ItemEntityKey: "item_id",
}

// 3. 创建适配器（将 Feast 适配为 core.FeatureService）
featureService := feastgrpc.NewFeatureServiceAdapter(feastClient, mapping)

// 4. 作为 core.FeatureService 使用
var fs core.FeatureService = featureService
```

**注意**：本子包使用 Feast 官方 Go SDK，性能优于 HTTP 客户端，推荐在生产环境使用。

## 自行实现

你也可以参考此实现，自行实现 `core.FeatureService` 接口，满足你的特定需求。
