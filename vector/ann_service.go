package vector

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// ANNService 是完整的向量数据库服务接口。
//
// 设计原则：
//   - 嵌入 core.VectorService（领域层接口），符合 DDD 依赖倒置原则
//   - 基础设施层（vector）实现领域层接口（core）
//   - 提供完整的向量数据库操作（CRUD + 集合管理）
//
// 使用场景对比：
//
//  1. 召回场景（推荐使用 core.VectorService）：
//     ```go
//     var vectorService core.VectorService = milvusService
//     result, err := vectorService.Search(ctx, &core.VectorSearchRequest{
//         Collection: "items",
//         Vector:     userVector,
//         TopK:       20,
//         Metric:     "cosine",
//     })
//     ```
//
//  2. 数据管理场景（使用 vector.ANNService）：
//     ```go
//     var annService vector.ANNService = milvusService
//     // 创建集合
//     err := annService.CreateCollection(ctx, &vector.CreateCollectionRequest{
//         Name:      "items",
//         Dimension: 128,
//         Metric:    "cosine",
//     })
//     // 插入向量
//     err = annService.Insert(ctx, &vector.InsertRequest{
//         Collection: "items",
//         Vectors:    itemVectors,
//         IDs:        itemIDs,
//     })
//     // 也可以使用 Search（因为嵌入了 core.VectorService）
//     result, err := annService.Search(ctx, &core.VectorSearchRequest{...})
//     ```
//
// 实现：
//   - vector.MilvusService 实现此接口
//   - 其他向量数据库（Faiss、Elasticsearch 等）也可以实现此接口
type ANNService interface {
	// 嵌入领域层接口（符合 DDD 原则）
	// 基础设施层接口扩展领域层接口，而不是相反
	core.VectorService

	// Insert 插入向量
	Insert(ctx context.Context, req *InsertRequest) error

	// Update 更新向量
	Update(ctx context.Context, req *UpdateRequest) error

	// Delete 删除向量
	Delete(ctx context.Context, req *DeleteRequest) error

	// CreateCollection 创建集合
	CreateCollection(ctx context.Context, req *CreateCollectionRequest) error

	// DropCollection 删除集合
	DropCollection(ctx context.Context, collection string) error

	// HasCollection 检查集合是否存在
	HasCollection(ctx context.Context, collection string) (bool, error)
}

// InsertRequest 向量插入请求
type InsertRequest struct {
	// Collection 集合名称
	Collection string

	// Vectors 向量列表
	Vectors [][]float64

	// IDs 对应的物品 ID 列表
	IDs []string

	// Metadata 元数据
	Metadata []map[string]interface{}
}

// UpdateRequest 向量更新请求
type UpdateRequest struct {
	// Collection 集合名称
	Collection string

	// Vector 新向量
	Vector []float64

	// ID 物品 ID
	ID string

	// Metadata 元数据
	Metadata map[string]interface{}
}

// DeleteRequest 向量删除请求
type DeleteRequest struct {
	// Collection 集合名称
	Collection string

	// IDs 要删除的物品 ID 列表
	IDs []string
}

// CreateCollectionRequest 创建集合请求
type CreateCollectionRequest struct {
	// Name 集合名称
	Name string

	// Dimension 向量维度
	Dimension int

	// Metric 距离度量方式
	Metric string

	// Params 额外参数
	Params map[string]interface{}
}

// MetricType 距离度量类型（用于类型安全）
// 注意：验证函数统一使用 core.ValidateVectorMetric
type MetricType string

const (
	MetricCosine      MetricType = "cosine"
	MetricEuclidean   MetricType = "euclidean"
	MetricInnerProduct MetricType = "inner_product"
)
