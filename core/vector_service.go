package core

import "context"

// VectorService 是向量检索服务的领域接口。
//
// 设计原则：
//   - 定义在领域层（core），由基础设施层（vector）实现
//   - 遵循依赖倒置原则：领域层定义接口，基础设施层实现接口
//   - 避免循环依赖：领域层不依赖基础设施层
//
// 使用场景：
//   - 双塔模型召回：根据 User Embedding 检索 Item Embeddings
//   - ANN 召回：向量相似度搜索
//   - 其他需要向量检索的场景
//
// 实现：
//   - vector.MilvusService 实现此接口
//   - 其他向量数据库（Faiss、Elasticsearch 等）也可以实现此接口
type VectorService interface {
	// Search 向量搜索
	Search(ctx context.Context, req *VectorSearchRequest) (*VectorSearchResult, error)

	// Close 关闭连接
	Close() error
}

// VectorSearchRequest 向量搜索请求
type VectorSearchRequest struct {
	// Collection 集合名称
	Collection string

	// Vector 查询向量
	Vector []float64

	// TopK 返回 TopK 个最相似的结果
	TopK int

	// Metric 距离度量方式：cosine / euclidean / inner_product
	Metric string

	// Filter 过滤条件（可选）
	Filter map[string]interface{}

	// Params 额外参数（可选）
	Params map[string]interface{}
}

// VectorSearchResult 向量搜索结果
type VectorSearchResult struct {
	// IDs 物品 ID 列表（按相似度排序）
	IDs []string

	// Scores 相似度分数列表
	Scores []float64

	// Distances 距离列表
	Distances []float64
}

// ValidateMetric 验证距离度量类型
func ValidateVectorMetric(metric string) bool {
	switch metric {
	case "cosine", "euclidean", "inner_product":
		return true
	default:
		return false
	}
}
