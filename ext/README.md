# 扩展包目录

此目录包含所有具体实现的扩展包。**每个扩展程序（ext 第一层下的一个“外部依赖”）只有一个 `go.mod`**，其子目录为子包，无需单独 go.mod。

## 设计原则

- **核心包无外部依赖**：核心包 `github.com/rushteam/reckit` 只保留工具库（CEL、YAML、sync），不依赖具体实现
- **扩展程序一个 go.mod**：ext 第一层即每个扩展程序（如 `ext/feast`、`ext/store/redis`、`ext/vector/milvus`）单独一个 go.mod，子目录为子包，不单独 go.mod
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

### 2. Feast (`ext/feast`)

Feast 特征存储客户端，**一个 go.mod**，子包 `common`、`http`、`grpc` 无单独 go.mod。

**使用扩展包**（安装整个 Feast 扩展）：
```bash
go get github.com/rushteam/reckit/ext/feast
```

**HTTP 子包**：
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

**gRPC 子包**（推荐生产环境）：
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

**或自行实现**：参考 `ext/feast/http/feast.go` 或 `ext/feast/grpc/feast.go`，自行实现 `core.FeatureService` 接口。

### 4. Milvus Vector (`ext/vector/milvus`)

Milvus 向量数据库实现。

**使用扩展包**：
```bash
go get github.com/rushteam/reckit/ext/vector/milvus
```

```go
import (
    "github.com/rushteam/reckit/core"
    milvus "github.com/rushteam/reckit/ext/vector/milvus"
)

milvusService := milvus.NewMilvusService("localhost:19530")
var vectorService core.VectorService = milvusService
var dbService core.VectorDatabaseService = milvusService
```

**或自行实现**：参考 `ext/vector/milvus/milvus.go`，实现 `core.VectorService` 或 `core.VectorDatabaseService` 接口。

## 迁移说明

以下具体实现已从核心包迁移到扩展包：

- `store/redis.go` → `ext/store/redis/redis.go`
- `feast/` → 整个包移至扩展包（`ext/feast/http` 和 `ext/feast/grpc`）
- `vector/milvus.go` → `ext/vector/milvus/milvus.go`
- `vector/milvus_client.go` → `ext/vector/milvus/milvus_client.go`

核心包现在只包含：
- 领域层接口（`core.Store`、`feature.FeatureService`、`core.VectorDatabaseService`）
- 无外部依赖的实现（`store.MemoryStore`）

**架构说明**：
- `core.FeatureService` 是领域层接口，推荐使用
- Feast 是基础设施层实现，应通过适配器适配为 `core.FeatureService` 使用
- `ext/feast/common` 包含共享的类型和接口（`Client`、`FeatureMapping` 等）

## 优势

1. **核心包轻量**：无外部依赖，只保留工具库
2. **扩展一个 go.mod**：每个扩展程序（一种外部依赖）一个 go.mod，子目录为子包，结构清晰
3. **按需引入**：用户只引入需要的扩展包
4. **易于扩展**：用户可以创建自己的扩展包实现接口
5. **灵活选择**：可以使用扩展包，也可以参考其实现自行实现接口