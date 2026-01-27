# VectorService 与 ANNService 统一方案（符合 DDD 原则）

## 问题分析

当前存在两个功能重叠的接口：
- `core.VectorService` - 领域层接口，只提供 `Search` 方法
- `vector.ANNService` - 基础设施层接口，提供完整的 CRUD + 集合管理

两者在 `Search` 功能上完全重复，但数据结构不同（`VectorSearchRequest` vs `SearchRequest`）。

## DDD 依赖倒置原则

**核心原则**：
- ✅ 领域层（core）定义接口
- ✅ 基础设施层（vector）实现接口
- ❌ 领域层不能依赖基础设施层

**当前项目的模式**：
- `core.Store` → `store.MemoryStore`、`store.RedisStore` 实现
- `core.MLService` → `service.TFServingClient`、`service.TorchServeClient` 实现
- `core.VectorService` → `vector.MilvusService`（通过包装器）实现

## 合理抽象方案

### 方案 1：接口组合（Interface Composition）⭐ 推荐

**核心思想**：让 `vector.ANNService` 嵌入 `core.VectorService`，通过接口组合实现统一。

```go
// core/vector_service.go
package core

type VectorService interface {
    Search(ctx context.Context, req *VectorSearchRequest) (*VectorSearchResult, error)
    Close() error
}

// vector/ann_service.go
package vector

import "github.com/rushteam/reckit/core"

// ANNService 是完整的向量数据库服务接口
// 它组合了 core.VectorService（领域层接口）和额外的数据管理功能
type ANNService interface {
    // 嵌入领域层接口（符合 DDD 原则）
    core.VectorService
    
    // 额外的数据管理功能（基础设施层特有）
    Insert(ctx context.Context, req *InsertRequest) error
    Update(ctx context.Context, req *UpdateRequest) error
    Delete(ctx context.Context, req *DeleteRequest) error
    CreateCollection(ctx context.Context, req *CreateCollectionRequest) error
    DropCollection(ctx context.Context, collection string) error
    HasCollection(ctx context.Context, collection string) (bool, error)
}
```

**实现方式**：

```go
// vector/milvus.go
package vector

import "github.com/rushteam/reckit/core"

type MilvusService struct {
    // ...
}

// 实现 core.VectorService.Search
func (s *MilvusService) Search(ctx context.Context, req *core.VectorSearchRequest) (*core.VectorSearchResult, error) {
    // 转换为内部 SearchRequest
    internalReq := &SearchRequest{
        Collection: req.Collection,
        Vector:     req.Vector,
        TopK:       req.TopK,
        Metric:     req.Metric,
        Filter:     req.Filter,
        Params:     req.Params,
    }
    
    // 调用内部实现
    result, err := s.searchInternal(ctx, internalReq)
    if err != nil {
        return nil, err
    }
    
    // 转换为 core.VectorSearchResult
    return &core.VectorSearchResult{
        IDs:       result.IDs,
        Scores:    result.Scores,
        Distances: result.Distances,
    }, nil
}

// 内部实现（接受 vector.SearchRequest）
func (s *MilvusService) searchInternal(ctx context.Context, req *SearchRequest) (*SearchResult, error) {
    // Milvus 实际实现
}

// 实现 vector.ANNService 的其他方法
func (s *MilvusService) Insert(ctx context.Context, req *InsertRequest) error { /* ... */ }
func (s *MilvusService) Update(ctx context.Context, req *UpdateRequest) error { /* ... */ }
// ...
```

**优点**：
- ✅ 符合 DDD 原则：领域层接口在 `core`，基础设施层接口在 `vector`
- ✅ 统一接口：`vector.ANNService` 自动包含 `core.VectorService` 的所有方法
- ✅ 无需包装器：`MilvusService` 直接实现两个接口
- ✅ 类型安全：编译时检查接口实现

**使用示例**：

```go
// 作为 core.VectorService 使用（召回场景）
var vectorService core.VectorService = milvusService
result, err := vectorService.Search(ctx, &core.VectorSearchRequest{...})

// 作为 vector.ANNService 使用（数据管理场景）
var annService vector.ANNService = milvusService
err := annService.Insert(ctx, &vector.InsertRequest{...})
result, err := annService.Search(ctx, &core.VectorSearchRequest{...}) // 也可以使用
```

### 方案 2：类型别名 + 适配器（当前方案改进）

保持当前结构，但简化适配逻辑：

```go
// vector/ann_service.go
type ANNService interface {
    // 使用与 core.VectorService 相同的数据结构
    Search(ctx context.Context, req *core.VectorSearchRequest) (*core.VectorSearchResult, error)
    
    // 其他方法...
}

// MilvusService 直接实现两个接口
func (s *MilvusService) Search(ctx context.Context, req *core.VectorSearchRequest) (*core.VectorSearchResult, error) {
    // 直接实现，无需转换
}
```

**优点**：
- ✅ 统一数据结构，无需转换
- ✅ 直接实现，无需包装器

**缺点**：
- ⚠️ `vector.ANNService` 依赖 `core` 包（违反分层原则）

### 方案 3：接口转换器（Adapter Pattern）

保持两个接口独立，提供便捷的转换函数：

```go
// vector/adapter.go
package vector

import "github.com/rushteam/reckit/core"

// ToVectorService 将 ANNService 转换为 core.VectorService
func ToVectorService(ann ANNService) core.VectorService {
    return &annToVectorAdapter{ann: ann}
}

type annToVectorAdapter struct {
    ann ANNService
}

func (a *annToVectorAdapter) Search(ctx context.Context, req *core.VectorSearchRequest) (*core.VectorSearchResult, error) {
    // 转换 SearchRequest
    internalReq := &SearchRequest{
        Collection: req.Collection,
        Vector:     req.Vector,
        TopK:       req.TopK,
        Metric:     req.Metric,
        Filter:     req.Filter,
        Params:     req.Params,
    }
    
    result, err := a.ann.Search(ctx, internalReq)
    if err != nil {
        return nil, err
    }
    
    return &core.VectorSearchResult{
        IDs:       result.IDs,
        Scores:    result.Scores,
        Distances: result.Distances,
    }, nil
}

func (a *annToVectorAdapter) Close() error {
    return a.ann.Close()
}
```

**优点**：
- ✅ 保持接口独立
- ✅ 符合 DDD 分层原则

**缺点**：
- ⚠️ 需要转换器，有性能开销
- ⚠️ 代码复杂度增加

## 推荐方案：方案 1（接口组合）

### 实现步骤

1. **修改 `vector.ANNService` 接口**：
   ```go
   type ANNService interface {
       core.VectorService  // 嵌入领域层接口
       // 其他方法...
   }
   ```

2. **统一数据结构**：
   - 保留 `core.VectorSearchRequest` 和 `core.VectorSearchResult`（领域层）
   - `vector.ANNService.Search` 直接使用 `core.VectorSearchRequest`
   - 内部实现可以有自己的 `SearchRequest` 类型（如果需要）

3. **修改 `MilvusService` 实现**：
   ```go
   // 直接实现 core.VectorService.Search
   func (s *MilvusService) Search(ctx context.Context, req *core.VectorSearchRequest) (*core.VectorSearchResult, error)
   
   // 实现其他 vector.ANNService 方法
   func (s *MilvusService) Insert(...) error
   // ...
   ```

4. **移除包装器**：
   - 删除 `milvusVectorServiceWrapper`
   - 删除 `NewMilvusVectorService` 函数
   - 直接使用 `MilvusService` 作为 `core.VectorService`

### 代码示例

```go
// 使用方式
milvusService := vector.NewMilvusService("localhost:19530")

// 作为 core.VectorService 使用
var vectorService core.VectorService = milvusService
result, err := vectorService.Search(ctx, &core.VectorSearchRequest{
    Collection: "items",
    Vector:     userVector,
    TopK:       20,
    Metric:     "cosine",
})

// 作为 vector.ANNService 使用（数据管理）
var annService vector.ANNService = milvusService
err := annService.Insert(ctx, &vector.InsertRequest{
    Collection: "items",
    Vectors:    itemVectors,
    IDs:        itemIDs,
})
```

## 总结

**方案 1（接口组合）是最佳选择**，因为：

1. ✅ **符合 DDD 原则**：
   - 领域层接口在 `core`
   - 基础设施层接口在 `vector`
   - 基础设施层实现领域层接口

2. ✅ **统一接口**：
   - `vector.ANNService` 自动包含 `core.VectorService`
   - 无需包装器或适配器

3. ✅ **类型安全**：
   - 编译时检查接口实现
   - 无需运行时转换

4. ✅ **代码简洁**：
   - 移除包装器代码
   - 统一数据结构

5. ✅ **向后兼容**：
   - 现有代码使用 `core.VectorService` 无需修改
   - 新增功能使用 `vector.ANNService`
