package core

import "context"

// VectorDatabaseService 是完整的向量数据库服务接口。
//
// 设计原则：
//   - 定义在领域层（core），由基础设施层（vector）实现
//   - 嵌入 VectorService（召回场景接口），符合接口组合原则
//   - 提供完整的向量数据库操作（CRUD + 集合管理）
//
// 使用场景对比：
//
//  1. 召回场景（推荐使用 VectorService）：
//     ```go
//     var vectorService VectorService = milvusService
//     result, err := vectorService.Search(ctx, &VectorSearchRequest{
//         Collection: "items",
//         Vector:     userVector,
//         TopK:       20,
//         Metric:     "cosine",
//     })
//     ```
//
//  2. 数据管理场景（使用 VectorDatabaseService）：
//     ```go
//     var dbService VectorDatabaseService = milvusService
//     // 创建集合
//     err := dbService.CreateCollection(ctx, &VectorCreateCollectionRequest{
//         Name:      "items",
//         Dimension: 128,
//         Metric:    "cosine",
//     })
//     // 插入向量
//     err = dbService.Insert(ctx, &VectorInsertRequest{
//         Collection: "items",
//         Vectors:    itemVectors,
//         IDs:        itemIDs,
//     })
//     // 也可以使用 Search（因为嵌入了 VectorService）
//     result, err := dbService.Search(ctx, &VectorSearchRequest{...})
//     ```
//
// 实现：
//   - ext/vector/milvus.MilvusService 实现此接口（扩展包）
//   - store.MemoryVectorService 实现此接口（内存实现）
//   - 其他向量数据库（Faiss、Elasticsearch 等）也可以实现此接口
type VectorDatabaseService interface {
	// 嵌入召回场景接口（符合 DDD 原则）
	// 基础设施层接口扩展领域层接口，而不是相反
	VectorService

	// Insert 插入向量
	Insert(ctx context.Context, req *VectorInsertRequest) error

	// Update 更新向量
	Update(ctx context.Context, req *VectorUpdateRequest) error

	// Delete 删除向量
	Delete(ctx context.Context, req *VectorDeleteRequest) error

	// CreateCollection 创建集合
	CreateCollection(ctx context.Context, req *VectorCreateCollectionRequest) error

	// DropCollection 删除集合
	DropCollection(ctx context.Context, collection string) error

	// HasCollection 检查集合是否存在
	HasCollection(ctx context.Context, collection string) (bool, error)
}

// VectorInsertRequest 向量插入请求
type VectorInsertRequest struct {
	// Collection 集合名称
	Collection string

	// Vectors 向量列表
	Vectors [][]float64

	// IDs 对应的物品 ID 列表
	IDs []string

	// Metadata 元数据
	Metadata []map[string]interface{}
}

// VectorUpdateRequest 向量更新请求
type VectorUpdateRequest struct {
	// Collection 集合名称
	Collection string

	// Vector 新向量
	Vector []float64

	// ID 物品 ID
	ID string

	// Metadata 元数据
	Metadata map[string]interface{}
}

// VectorDeleteRequest 向量删除请求
type VectorDeleteRequest struct {
	// Collection 集合名称
	Collection string

	// IDs 要删除的物品 ID 列表
	IDs []string
}

// VectorCreateCollectionRequest 创建集合请求
type VectorCreateCollectionRequest struct {
	// Name 集合名称
	Name string

	// Dimension 向量维度
	Dimension int

	// Metric 距离度量方式
	Metric string

	// Params 额外参数
	Params map[string]interface{}
}
