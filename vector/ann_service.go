package vector

import (
	"context"
)

// ANNService 是抽象向量 ANN（Approximate Nearest Neighbor）服务接口。
//
// 设计目标：
//   - 统一向量数据库接口（Milvus、Faiss、Pinecone、Weaviate 等）
//   - 支持向量搜索、插入、更新、删除
//   - 支持多种距离度量（余弦、欧氏、内积等）
//   - 支持集合（Collection）管理
//
// 使用示例：
//
//	service := vector.NewMilvusService(...)
//	results, err := service.Search(ctx, &vector.SearchRequest{
//		Collection: "items",
//		Vector:     userVector,
//		TopK:       20,
//		Metric:     "cosine",
//	})
type ANNService interface {
	// Search 向量搜索（核心功能）
	Search(ctx context.Context, req *SearchRequest) (*SearchResult, error)

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

	// Close 关闭连接
	Close() error
}

// SearchRequest 向量搜索请求
type SearchRequest struct {
	// Collection 集合名称
	Collection string

	// Vector 查询向量
	Vector []float64

	// TopK 返回 TopK 个最相似的结果
	TopK int

	// Metric 距离度量方式：cosine / euclidean / inner_product
	Metric string

	// Filter 过滤条件（可选，格式由具体实现决定）
	Filter map[string]interface{}

	// Params 额外参数（可选，用于调优）
	Params map[string]interface{}
}

// SearchResult 向量搜索结果
type SearchResult struct {
	// IDs 物品 ID 列表（按相似度排序）
	IDs []int64

	// Scores 相似度分数列表（与 IDs 一一对应）
	Scores []float64

	// Distances 距离列表（与 IDs 一一对应，如果支持）
	Distances []float64
}

// InsertRequest 向量插入请求
type InsertRequest struct {
	// Collection 集合名称
	Collection string

	// Vectors 向量列表
	Vectors [][]float64

	// IDs 对应的物品 ID 列表（与 Vectors 一一对应）
	IDs []int64

	// Metadata 元数据（可选，格式由具体实现决定）
	Metadata []map[string]interface{}
}

// UpdateRequest 向量更新请求
type UpdateRequest struct {
	// Collection 集合名称
	Collection string

	// Vector 新向量
	Vector []float64

	// ID 物品 ID
	ID int64

	// Metadata 元数据（可选）
	Metadata map[string]interface{}
}

// DeleteRequest 向量删除请求
type DeleteRequest struct {
	// Collection 集合名称
	Collection string

	// IDs 要删除的物品 ID 列表
	IDs []int64
}

// CreateCollectionRequest 创建集合请求
type CreateCollectionRequest struct {
	// Name 集合名称
	Name string

	// Dimension 向量维度
	Dimension int

	// Metric 距离度量方式：cosine / euclidean / inner_product
	Metric string

	// Params 额外参数（可选）
	Params map[string]interface{}
}

// MetricType 距离度量类型
type MetricType string

const (
	MetricCosine      MetricType = "cosine"       // 余弦相似度
	MetricEuclidean   MetricType = "euclidean"    // 欧氏距离
	MetricInnerProduct MetricType = "inner_product" // 内积
)

// ValidateMetric 验证距离度量类型
func ValidateMetric(metric string) bool {
	switch metric {
	case string(MetricCosine), string(MetricEuclidean), string(MetricInnerProduct):
		return true
	default:
		return false
	}
}
