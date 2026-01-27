# 架构重构总结

## 重构时间
2026-01-26

## 重构目标
消除过度抽象，符合 DDD 原则，实现高内聚低耦合，结构清晰简单。

## 重构内容

### 1. ✅ 删除 `service.ANNServiceClient`（高优先级）

**问题**：
- 概念混淆：ANN（向量检索）不应该实现 ML 预测接口
- 名字和职责不匹配：`ANNServiceClient` 实现的是 `core.MLService`，但名字暗示它是向量服务

**解决方案**：
- 删除 `service/ann_service.go`
- 更新 `service/factory.go`，移除 `ServiceTypeANN` 支持
- 更新示例代码和文档

**影响**：
- 向量检索应使用 `core.VectorService` 接口
- 参考 `ext/vector/milvus` 或 `store.MemoryVectorService` 实现

---

### 2. ✅ 将 `feature.FeatureService` 移到 `core` 包（高优先级）

**问题**：
- 不符合 DDD：领域接口应该在 `core` 包中定义

**解决方案**：
- 创建 `core/feature_service.go`，定义 `FeatureService` 接口
- 更新 `feature/service.go`，移除接口定义，改为使用 `core.FeatureService`
- 更新所有使用 `feature.FeatureService` 的地方改为 `core.FeatureService`

**影响**：
- `feature.BaseFeatureService` 现在实现 `core.FeatureService`
- 所有使用特征服务的地方需要使用 `core.FeatureService`

---

### 3. ✅ 统一 recall 包的 Store 接口（高优先级）

**问题**：
- 过度抽象：每个召回算法都有自己的 Store 接口（`CFStore`、`ContentStore`、`MFStore` 等）
- 不符合 DDD：领域接口应该在 `core` 包中定义

**解决方案**：
- 创建 `core/recall_store.go`，定义统一的 `RecallDataStore` 接口
- 将 `CFStore`、`ContentStore`、`MFStore` 改为类型别名指向 `core.RecallDataStore`
- 更新适配器（`StoreCFAdapter`、`StoreContentAdapter`、`StoreMFAdapter`）实现 `core.RecallDataStore`

**影响**：
- 统一的召回数据存储接口，避免接口爆炸
- 适配器模式统一管理，符合 DDD 原则

---

### 4. ✅ 将 `core.VectorDatabaseService` 的请求类型移到 `core` 包（已完成）

**问题**：
- 向量数据库的 CRUD 操作是领域概念，应该在 `core` 中定义

**解决方案**：
- 创建 `core/vector_database_service.go`，定义 `VectorDatabaseService` 接口
- 将所有请求类型移到 `core` 包：
  - `core.VectorInsertRequest`
  - `core.VectorUpdateRequest`
  - `core.VectorDeleteRequest`
  - `core.VectorCreateCollectionRequest`
- 将 `vector.ANNService` 改为类型别名：`type ANNService = core.VectorDatabaseService` ✅ 已完成
- 更新所有实现（`ext/vector/milvus`、`store.MemoryVectorService`）

**影响**：
- 更清晰的接口定义位置
- 统一命名：`VectorService`（召回）+ `VectorDatabaseService`（数据管理）

---

## 重构后的架构

### core 包（领域层）
```
core/
  ├── vector_service.go          // VectorService (召回场景)
  ├── vector_database_service.go // VectorDatabaseService (数据管理)
  ├── store.go                   // Store, KeyValueStore
  ├── recall_store.go            // RecallDataStore (统一召回数据存储)
  ├── ml_service.go              // MLService
  └── feature_service.go         // FeatureService
```

### vector 包（基础设施层）
```
vector/
  └── ann_service.go  // 类型别名：type ANNService = core.VectorDatabaseService
```

### service 包（基础设施层）
```
service/
  ├── ml_service.go        // MLService 实现
  ├── tf_serving.go        // TFServingClient
  └── torchserve.go        // TorchServeClient
  // 已删除：ann_service.go
```

### recall 包（应用层）
```
recall/
  ├── source.go
  ├── cf_store_adapter.go      // 实现 core.RecallDataStore
  ├── content_store_adapter.go // 实现 core.RecallDataStore
  ├── mf_store_adapter.go      // 实现 core.RecallDataStore
  └── ... (其他召回实现)
```

### feature 包（基础设施层）
```
feature/
  ├── provider.go      // FeatureProvider 接口
  ├── cache.go         // FeatureCache 接口
  ├── monitor.go       // FeatureMonitor 接口
  ├── service.go       // 实现 core.FeatureService
  └── base_service.go  // BaseFeatureService 实现 core.FeatureService
```

---

## 接口使用指南

### FeatureService

**领域接口**：`core.FeatureService`

```go
import "github.com/rushteam/reckit/core"
import "github.com/rushteam/reckit/feature"

// 创建特征服务
featureService := feature.NewBaseFeatureService(provider)

// 使用领域接口
var fs core.FeatureService = featureService
features, err := fs.GetUserFeatures(ctx, "user_123")
```

### VectorService

**召回场景**：`core.VectorService`

```go
import "github.com/rushteam/reckit/core"
import milvus "github.com/rushteam/reckit/ext/vector/milvus"

milvusService, _ := milvus.NewMilvusService("localhost:19530")

// 作为 core.VectorService 使用（召回场景）
var vectorService core.VectorService = milvusService
result, err := vectorService.Search(ctx, &core.VectorSearchRequest{
    Collection: "items",
    Vector:     userVector,
    TopK:       20,
    Metric:     "cosine",
})
```

### VectorDatabaseService

**数据管理场景**：`core.VectorDatabaseService`

```go
import "github.com/rushteam/reckit/core"
import milvus "github.com/rushteam/reckit/ext/vector/milvus"

milvusService, _ := milvus.NewMilvusService("localhost:19530")

// 作为 core.VectorDatabaseService 使用（数据管理场景）
var dbService core.VectorDatabaseService = milvusService

// 创建集合
err := dbService.CreateCollection(ctx, &core.VectorCreateCollectionRequest{
    Name:      "items",
    Dimension: 128,
    Metric:    "cosine",
})

// 插入向量
err = dbService.Insert(ctx, &core.VectorInsertRequest{
    Collection: "items",
    Vectors:    itemVectors,
    IDs:        itemIDs,
})
```

**向后兼容**：可以使用类型别名

```go
import milvus "github.com/rushteam/reckit/ext/vector/milvus"

// core.VectorDatabaseService 领域接口
var dbService core.VectorDatabaseService = milvusService
```

### RecallDataStore

**统一召回数据存储**：`core.RecallDataStore`

```go
import "github.com/rushteam/reckit/core"
import "github.com/rushteam/reckit/recall"

// 创建适配器（基于 core.Store）
store := store.NewMemoryStore()
cfAdapter := recall.NewStoreCFAdapter(store, "cf")

// 使用领域接口
var recallStore core.RecallDataStore = cfAdapter

// 协同过滤数据
userItems, _ := recallStore.GetUserItems(ctx, "user_123")

// 内容推荐数据
itemFeatures, _ := recallStore.GetItemFeatures(ctx, "item_456")

// 矩阵分解数据
userVector, _ := recallStore.GetUserVector(ctx, "user_123")
```

---

## 重构原则总结

1. **DDD 分层**：领域接口在 `core`，基础设施层只实现接口
2. **高内聚**：同一类概念统一管理（如 `RecallDataStore` 统一所有召回数据访问）
3. **低耦合**：通过接口依赖，避免具体实现依赖
4. **清晰简单**：消除不必要的抽象层（如删除 `service.ANNServiceClient`）

---

## 迁移指南

### 从 `feature.FeatureService` 迁移

**之前**：
```go
import "github.com/rushteam/reckit/feature"
var fs feature.FeatureService = featureService
```

**现在**：
```go
import "github.com/rushteam/reckit/core"
import "github.com/rushteam/reckit/feature"
var fs core.FeatureService = featureService
```

### 从 `vector.ANNService` 迁移 ✅ 已完成

**之前**：
```go
import milvus "github.com/rushteam/reckit/ext/vector/milvus"
var dbService core.VectorDatabaseService = milvusService
err := annService.CreateCollection(ctx, &vector.CreateCollectionRequest{...})
```

**现在（推荐）**：
```go
import "github.com/rushteam/reckit/core"
var dbService core.VectorDatabaseService = milvusService
err := dbService.CreateCollection(ctx, &core.VectorCreateCollectionRequest{...})
```

**向后兼容**：
```go
import milvus "github.com/rushteam/reckit/ext/vector/milvus"
var dbService core.VectorDatabaseService = milvusService // 类型别名，仍可使用
```

### 从 `service.ANNServiceClient` 迁移

**之前**：
```go
import "github.com/rushteam/reckit/service"
annService := service.NewANNServiceClient("http://localhost:19530", "items")
```

**现在**：
```go
import "github.com/rushteam/reckit/core"
import milvus "github.com/rushteam/reckit/ext/vector/milvus"

// 使用向量服务
vectorService, _ := milvus.NewMilvusService("localhost:19530")
var vs core.VectorService = vectorService
```

---

## 注意事项

1. **接口定义位置**：所有领域接口都在 `core` 包
2. **类型别名**：`vector.ANNService` 已移除，使用 `core.VectorDatabaseService`
3. **请求类型**：所有请求类型都在 `core` 包（`core.VectorInsertRequest` 等）
4. **实现位置**：基础设施层实现领域接口，不定义新接口

---

## 相关文档

- `docs/ARCHITECTURE_ANALYSIS.md` - 详细的架构分析和重构建议
- `CLAUDE.md` - 完整的使用指南
