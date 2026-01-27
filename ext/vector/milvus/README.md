# Milvus 向量数据库扩展包

Milvus 向量数据库实现，位于扩展包中，独立管理依赖。

## 安装

```bash
go get github.com/rushteam/reckit/ext/vector/milvus
```

## 使用

```go
import (
    "github.com/rushteam/reckit/core"
    milvus "github.com/rushteam/reckit/ext/vector/milvus"
)

// 创建 Milvus 服务
milvusService := milvus.NewMilvusService("localhost:19530")

// 作为 core.VectorService 使用
var vectorService core.VectorService = milvusService

// 作为 core.VectorDatabaseService 使用
var dbService core.VectorDatabaseService = milvusService
```

## 依赖

- `github.com/rushteam/reckit` - 核心包（仅接口定义）
- `github.com/milvus-io/milvus/client/v2` - Milvus 官方 SDK v2

**注意**：此扩展包使用 Milvus Go SDK v2 (`github.com/milvus-io/milvus/client/v2`)，这是官方维护的最新 Go SDK。

## 自行实现

你也可以参考此实现，自行实现 `core.VectorService` 或 `core.VectorDatabaseService` 接口，满足你的特定需求。