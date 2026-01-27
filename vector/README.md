# Vector ANN Service

抽象向量 ANN（Approximate Nearest Neighbor）服务接口，支持 Milvus、Faiss、Pinecone 等向量数据库。

## 设计原则

- **DDD 分层架构**：`vector.ANNService` 嵌入 `core.VectorService`（领域层接口），符合依赖倒置原则
- **高内聚低耦合**：基础设施层（vector）实现领域层接口（core），领域层不依赖基础设施层
- **接口组合**：通过接口组合实现统一，无需包装器或适配器

## 接口使用场景对比

### core.VectorService（召回场景专用）

**适用场景**：推荐系统的召回阶段，只需要向量搜索功能

```go
import (
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/vector"
)

// 创建服务
milvusService := vector.NewMilvusService("localhost:19530")

// 作为 core.VectorService 使用（召回场景）
var vectorService core.VectorService = milvusService

// 向量搜索
result, err := vectorService.Search(ctx, &core.VectorSearchRequest{
    Collection: "items",
    Vector:     userVector,
    TopK:       20,
    Metric:     "cosine",
})
```

**特点**：
- ✅ 接口简洁，只包含 Search 方法
- ✅ 专注于召回场景，符合单一职责原则
- ✅ 领域层接口，不依赖基础设施层

### vector.ANNService（完整向量数据库接口）

**适用场景**：向量数据管理、离线导入、在线检索等完整操作

```go
import (
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/vector"
)

// 创建服务
milvusService := vector.NewMilvusService("localhost:19530")

// 作为 vector.ANNService 使用（数据管理场景）
var annService vector.ANNService = milvusService

// 1. 创建集合
err := annService.CreateCollection(ctx, &vector.CreateCollectionRequest{
    Name:      "items",
    Dimension: 128,
    Metric:    "cosine",
})

// 2. 插入向量
err = annService.Insert(ctx, &vector.InsertRequest{
    Collection: "items",
    Vectors:    itemVectors,
    IDs:        itemIDs,
})

// 3. 向量搜索（也可以使用，因为嵌入了 core.VectorService）
result, err := annService.Search(ctx, &core.VectorSearchRequest{
    Collection: "items",
    Vector:     userVector,
    TopK:       20,
    Metric:     "cosine",
})

// 4. 更新向量
err = annService.Update(ctx, &vector.UpdateRequest{
    Collection: "items",
    ID:         "item_1",
    Vector:     newVector,
})

// 5. 删除向量
err = annService.Delete(ctx, &vector.DeleteRequest{
    Collection: "items",
    IDs:        []string{"item_1", "item_2"},
})
```

**特点**：
- ✅ 完整的向量数据库操作（CRUD + 集合管理）
- ✅ 嵌入 `core.VectorService`，可以使用 Search 方法
- ✅ 基础设施层接口，提供技术实现细节

### 选择建议

| 场景 | 推荐接口 | 原因 |
|------|---------|------|
| **召回场景** | `core.VectorService` | 只需要 Search，接口简洁，符合领域层抽象 |
| **数据管理** | `vector.ANNService` | 需要完整的 CRUD 和集合管理功能 |
| **混合场景** | `vector.ANNService` | 既需要召回也需要数据管理，一个接口满足所有需求 |

## 设计目标

- **统一接口**：提供统一的向量数据库接口，便于替换实现
- **高性能搜索**：支持大规模向量检索
- **多种距离度量**：支持余弦相似度、欧氏距离、内积
- **完整 CRUD**：支持向量的插入、更新、删除
- **集合管理**：支持集合的创建、删除、查询

## 核心接口

### ANNService

```go
type ANNService interface {
    // 嵌入 core.VectorService（领域层接口）
    core.VectorService
    
    // 额外的数据管理功能
    Insert(ctx context.Context, req *InsertRequest) error
    Update(ctx context.Context, req *UpdateRequest) error
    Delete(ctx context.Context, req *DeleteRequest) error
    CreateCollection(ctx context.Context, req *CreateCollectionRequest) error
    DropCollection(ctx context.Context, collection string) error
    HasCollection(ctx context.Context, collection string) (bool, error)
}
```

## 实现状态

### MilvusService

当前为**占位实现**，接口已定义但未完整实现。实际实现需要：

1. **安装 Milvus SDK**：
   ```bash
   go get github.com/milvus-io/milvus-sdk-go/v2/client
   ```

2. **实现核心方法**：
   - Search：向量搜索
   - Insert：向量插入
   - Delete：向量删除
   - CreateCollection：创建集合

3. **类型转换**：
   - Milvus 使用 `float32`，需要与 `float64` 转换
   - Milvus 使用不同的距离度量名称（L2、IP、COSINE）
   - **使用 VARCHAR 主键**：直接使用 string IDs，无需转换为 int64（Milvus 2.0+ 支持）

## 使用示例

### 1. 创建 Milvus 服务

```go
import "github.com/rushteam/reckit/vector"

// 创建 Milvus 服务
milvusService := vector.NewMilvusService(
    "localhost:19530",
    vector.WithMilvusAuth("root", "Milvus"),
    vector.WithMilvusDatabase("recommend"),
    vector.WithMilvusTimeout(30),
)
defer milvusService.Close()
```

### 2. 创建集合

```go
err := milvusService.CreateCollection(ctx, &vector.CreateCollectionRequest{
    Name:      "items",
    Dimension: 128,
    Metric:    "cosine",
})
```

### 3. 插入向量

```go
err := milvusService.Insert(ctx, &vector.InsertRequest{
    Collection: "items",
    Vectors: [][]float64{
        {0.1, 0.2, 0.3, ...}, // 物品 1 的向量
        {0.4, 0.5, 0.6, ...}, // 物品 2 的向量
    },
    IDs: []string{"1", "2"},
})
```

### 4. 向量搜索

```go
result, err := milvusService.Search(ctx, &core.VectorSearchRequest{
    Collection: "items",
    Vector:     userVector,
    TopK:       20,
    Metric:     "cosine",
})

// result.IDs: ["item_3", "item_7", "item_12", ...]  // 最相似的物品 ID（string 类型）
// result.Scores: [0.95, 0.92, 0.88, ...]  // 相似度分数
```

### 5. 检查集合是否存在

```go
exists, err := milvusService.HasCollection(ctx, "items")
if err != nil {
    // 处理错误
} else if !exists {
    // 创建集合
    err = milvusService.CreateCollection(ctx, &vector.CreateCollectionRequest{
        Name:      "items",
        Dimension: 128,
        Metric:    "cosine",
    })
}
```

### 6. 更新向量

```go
err := milvusService.Update(ctx, &vector.UpdateRequest{
    Collection: "items",
    Vector:     newVector,
    ID:         itemID,
    Metadata: map[string]interface{}{
        "category": "tech",
        "updated_at": time.Now(),
    },
})
```

### 7. 删除向量

```go
err := milvusService.Delete(ctx, &vector.DeleteRequest{
    Collection: "items",
    IDs:        []string{"item_1", "item_2", "item_3"}, // 使用 string IDs
})
```

### 8. 删除集合

```go
err := milvusService.DropCollection(ctx, "items")
```

### 9. 与 recall.ANN 集成

```go
import (
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/vector"
)

// 创建适配器
adapter := vector.NewVectorStoreAdapter(milvusService, "items")

// 使用 ANN 召回
ann := &recall.ANN{
    Store:      adapter,
    TopK:       20,
    Metric:     "cosine",
    UserVector: userVector,
}

items, err := ann.Recall(ctx, rctx)
```

**注意**：由于 `VectorStoreAdapter` 的 `GetVector` 和 `ListVectors` 返回 `ErrNotSupported`，
当前 `recall.ANN` 的实现会失败。建议：
- 直接使用 `ANNService.Search` 方法
- 或者修改 `recall.ANN` 以支持 `ANNService` 接口

### 10. Pipeline 集成

```go
import (
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/pipeline"
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/vector"
)

// 创建 Milvus 服务
milvusService := vector.NewMilvusService("localhost:19530")
defer milvusService.Close()

// 创建适配器
adapter := vector.NewVectorStoreAdapter(milvusService, "items")

// 创建推荐上下文
rctx := &core.RecommendContext{
    UserID: 1,
    Scene:  "feed",
    UserProfile: map[string]any{
        "user_vector": userVector,
    },
}

// 创建 Pipeline
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{
            Sources: []recall.Source{
                &recall.ANN{
                    Store:      adapter,
                    TopK:       20,
                    Metric:     "cosine",
                    UserVector: userVector,
                },
            },
            Dedup: true,
        },
        // 其他节点...
    },
}

// 运行 Pipeline
items, err := p.Run(ctx, rctx, nil)
```

### 11. 完整示例

完整示例代码请参考：`examples/milvus_ann/main.go`

运行示例：
```bash
go run ./examples/milvus_ann
```

示例包含：
- 创建 Milvus 服务
- 检查并创建集合
- 插入向量
- 向量搜索
- 与 recall.ANN 集成
- Pipeline 集成

## 距离度量

支持的距离度量类型：

- **cosine**：余弦相似度（默认）
- **euclidean**：欧氏距离
- **inner_product**：内积

## 适配器设计

`VectorStoreAdapter` 将 `ANNService` 适配为 `recall.VectorStore` 接口，但需要注意：

### 接口限制

- **GetVector**：返回 `ErrNotSupported`
  - 原因：Milvus 等向量数据库通常不支持直接通过 ID 获取向量
  - 替代方案：使用 `Search` 方法（带 ID 过滤条件）或维护 ID -> Vector 缓存

- **ListVectors**：返回 `ErrNotSupported`
  - 原因：向量数据库通常不支持全量查询（性能问题）
  - 替代方案：使用 `Search` 方法进行向量搜索

### 推荐使用方式

1. **直接使用 ANNService**（推荐）：
   ```go
   result, err := milvusService.Search(ctx, &core.VectorSearchRequest{
       Collection: "items",
       Vector:     userVector,
       TopK:       20,
       Metric:     "cosine",
   })
   ```
   
   或者作为 `core.VectorService` 使用（召回场景）：
   ```go
   var vectorService core.VectorService = milvusService
   result, err := vectorService.Search(ctx, &core.VectorSearchRequest{
       Collection: "items",
       Vector:     userVector,
       TopK:       20,
       Metric:     "cosine",
   })
   ```

2. **使用适配器的 Search 方法**：
   ```go
   adapter := vector.NewVectorStoreAdapter(milvusService, "items")
   ids, scores, err := adapter.Search(ctx, userVector, 20, "cosine")
   ```

3. **修改 recall.ANN 以支持 ANNService**（未来改进）：
   - 在 `recall.ANN` 中添加 `ANNService` 字段
   - 优先使用 `ANNService.Search`，回退到 `VectorStore.ListVectors`

## 后续实现计划

### 1. 完整实现 MilvusService

```go
import (
    "github.com/milvus-io/milvus-sdk-go/v2/client"
    "github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type MilvusService struct {
    client *client.Client
    // ...
}

// 注意：使用 VARCHAR 主键，直接使用 string IDs
// 创建集合时指定 VARCHAR 主键类型
schema := &entity.Schema{
    CollectionName: "items",
    Fields: []*entity.Field{
        {
            Name:       "id",
            DataType:   entity.FieldTypeVarChar,
            MaxLength:  255,
            IsPrimary:  true,
        },
        {
            Name:     "vector",
            DataType: entity.FieldTypeFloatVector,
            TypeParams: map[string]string{
                "dim": "128",
            },
        },
    },
}
```

### 2. 支持其他向量数据库

- **FaissService**：Facebook 的向量相似度搜索库
- **PineconeService**：云原生向量数据库
- **WeaviateService**：开源向量搜索引擎

### 3. 性能优化

- 连接池管理
- 批量操作优化
- 异步操作支持

## 参考

- [Milvus 官方文档](https://milvus.io/docs)
- [Milvus Go SDK](https://github.com/milvus-io/milvus-sdk-go)
