# 扩展包目录

此目录包含所有具体实现的扩展包，每个扩展包都有独立的 `go.mod` 文件，独立管理依赖。

## 设计原则

- **核心包无外部依赖**：核心包 `github.com/rushteam/reckit` 只保留工具库（CEL、YAML、sync），不依赖具体实现
- **扩展包独立管理**：每个扩展包有独立的 `go.mod`，可以管理自己的依赖版本
- **用户按需引入**：用户只引入需要的扩展包，避免不必要的依赖
- **可自行实现**：用户可以使用扩展包，也可以参考其实现自行实现对应接口

## 扩展包列表

### 1. Redis Store (`ext/store/redis`)

Redis 存储实现。

**使用扩展包**：
```bash
go get github.com/rushteam/reckit/ext/store/redis
```

```go
import (
    "github.com/rushteam/reckit/core"
    redisstore "github.com/rushteam/reckit/ext/store/redis"
)

store, err := redisstore.NewRedisStore("localhost:6379", 0)
var s core.Store = store
```

**或自行实现**：参考 `ext/store/redis/redis.go`，实现 `core.Store` 或 `core.KeyValueStore` 接口。

### 2. Feast HTTP (`ext/feast/http`)

Feast HTTP 客户端实现。

**使用扩展包**：
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

**或自行实现**：参考 `ext/feast/http/http_client.go`，实现 `feast.Client` 接口。

### 3. Feast gRPC (`ext/feast/grpc`)

Feast gRPC 客户端实现（基于官方 SDK）。

**使用扩展包**：
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

**或自行实现**：参考 `ext/feast/grpc/grpc_client.go`，实现 `feast.Client` 接口。

### 4. Milvus Vector (`ext/vector/milvus`)

Milvus 向量数据库实现。

**使用扩展包**：
```bash
go get github.com/rushteam/reckit/ext/vector/milvus
```

```go
import (
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/vector"
    milvus "github.com/rushteam/reckit/ext/vector/milvus"
)

milvusService := milvus.NewMilvusService("localhost:19530")
var vectorService core.VectorService = milvusService
var annService vector.ANNService = milvusService
```

**或自行实现**：参考 `ext/vector/milvus/milvus.go`，实现 `core.VectorService` 或 `vector.ANNService` 接口。

## 迁移说明

以下具体实现已从核心包迁移到扩展包：

- `store/redis.go` → `ext/store/redis/redis.go`
- `feast/http_client.go` → `ext/feast/http/http_client.go`
- `feast/factory.go` → `ext/feast/http/factory.go`
- `feast/adapter.go` → `ext/feast/http/adapter.go`
- `feast/grpc_client.go` → `ext/feast/grpc/grpc_client.go`
- `vector/milvus.go` → `ext/vector/milvus/milvus.go`
- `vector/milvus_client.go` → `ext/vector/milvus/milvus_client.go`

核心包现在只包含：
- 接口定义（`core.Store`、`feast.Client`、`vector.ANNService`）
- 无外部依赖的实现（`store.MemoryStore`）

## 优势

1. **核心包轻量**：无外部依赖，只保留工具库
2. **版本独立**：每个扩展包可以管理自己的依赖版本
3. **按需引入**：用户只引入需要的扩展包
4. **易于扩展**：用户可以创建自己的扩展包实现接口
5. **灵活选择**：可以使用扩展包，也可以参考其实现自行实现接口