# Feast gRPC 扩展包

Feast gRPC 客户端实现，位于扩展包中，独立管理依赖。

## 安装

```bash
go get github.com/rushteam/reckit/ext/feast/grpc
```

## 使用

```go
import (
    "github.com/rushteam/reckit/feast"
    feastgrpc "github.com/rushteam/reckit/ext/feast/grpc"
)

// 创建 gRPC 客户端
client, err := feastgrpc.NewGrpcClient("localhost", 6565, "my_project")
if err != nil {
    log.Fatal(err)
}
defer client.Close()

// 作为 feast.Client 使用
var c feast.Client = client
```

## 依赖

- `github.com/rushteam/reckit` - 核心包（仅接口定义）
- `github.com/feast-dev/feast/sdk/go` - Feast 官方 SDK

## 自行实现

你也可以参考此实现，自行实现 `feast.Client` 接口，满足你的特定需求。