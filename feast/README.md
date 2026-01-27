# Feast Feature Store 接口

Feast Feature Store 的接口定义，位于核心包中。

## 接口定义

核心包只包含接口定义（`feast.Client`），具体实现已移至扩展包：

- **HTTP 客户端**：`ext/feast/http` - 使用 HTTP API
- **gRPC 客户端**：`ext/feast/grpc` - 使用官方 SDK

## 使用扩展包

### HTTP 客户端

```bash
go get github.com/rushteam/reckit/ext/feast/http
```

```go
import (
    "github.com/rushteam/reckit/feast"
    feasthttp "github.com/rushteam/reckit/ext/feast/http"
)

client, err := feasthttp.NewClient("http://localhost:6566", "my_project")
var c feast.Client = client
```

### gRPC 客户端

```bash
go get github.com/rushteam/reckit/ext/feast/grpc
```

```go
import (
    "github.com/rushteam/reckit/feast"
    feastgrpc "github.com/rushteam/reckit/ext/feast/grpc"
)

client, err := feastgrpc.NewGrpcClient("localhost", 6565, "my_project")
var c feast.Client = client
```

## 自行实现

你也可以参考扩展包实现，自行实现 `feast.Client` 接口，满足你的特定需求。

参考实现：
- `ext/feast/http/http_client.go` - HTTP 实现示例
- `ext/feast/grpc/grpc_client.go` - gRPC 实现示例