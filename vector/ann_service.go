package vector

import (
	"context"
)

// ANNService 是抽象向量 ANN（Approximate Nearest Neighbor）服务接口。
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

	// Filter 过滤条件
	Filter map[string]interface{}

	// Params 额外参数
	Params map[string]interface{}
}

// SearchResult 向量搜索结果
type SearchResult struct {
	// IDs 物品 ID 列表（按相似度排序）
	IDs []string

	// Scores 相似度分数列表
	Scores []float64

	// Distances 距离列表
	Distances []float64
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

// MetricType 距离度量类型
type MetricType string

const (
	MetricCosine      MetricType = "cosine"
	MetricEuclidean   MetricType = "euclidean"
	MetricInnerProduct MetricType = "inner_product"
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
