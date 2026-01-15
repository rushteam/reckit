package vector

import (
	"context"

	"reckit/recall"
)

// VectorStoreAdapter 将 ANNService 适配为 recall.VectorStore 接口。
//
// 这样可以让现有的 recall.ANN 使用 Milvus 等向量数据库服务。
//
// 使用示例：
//
//	milvusService := vector.NewMilvusService("localhost:19530")
//	adapter := vector.NewVectorStoreAdapter(milvusService, "items")
//	ann := &recall.ANN{
//	    Store: adapter,
//	    TopK:  20,
//	}
type VectorStoreAdapter struct {
	service    ANNService
	collection string
}

// NewVectorStoreAdapter 创建一个新的 VectorStore 适配器。
func NewVectorStoreAdapter(service ANNService, collection string) *VectorStoreAdapter {
	return &VectorStoreAdapter{
		service:    service,
		collection: collection,
	}
}

// GetVector 获取单个物品的向量（实现 recall.VectorStore 接口）
//
// 注意：Milvus 等向量数据库通常不支持直接通过 ID 获取向量。
// 此方法返回错误，建议使用 Search 方法或直接使用 ANNService 接口。
func (a *VectorStoreAdapter) GetVector(ctx context.Context, itemID int64) ([]float64, error) {
	// TODO: 实际实现
	// 如果向量数据库支持通过 ID 查询，可以实现：
	// 1. 使用 ID 作为过滤条件进行查询
	// 2. 或者维护一个 ID -> Vector 的缓存映射

	return nil, ErrNotSupported
}

// ListVectors 获取所有物品的向量（实现 recall.VectorStore 接口）
//
// 注意：Milvus 等向量数据库通常不支持直接列出所有向量（性能问题）。
// 此方法返回错误，建议使用 Search 方法进行向量搜索。
func (a *VectorStoreAdapter) ListVectors(ctx context.Context) (map[int64][]float64, error) {
	// TODO: 实际实现
	// 如果向量数据库支持全量查询，可以实现：
	// 1. 使用游标或分页方式获取所有向量
	// 2. 或者维护一个内存缓存

	return nil, ErrNotSupported
}

// Search 使用向量搜索（推荐使用此方法）
func (a *VectorStoreAdapter) Search(ctx context.Context, vector []float64, topK int, metric string) ([]int64, []float64, error) {
	req := &SearchRequest{
		Collection: a.collection,
		Vector:     vector,
		TopK:       topK,
		Metric:     metric,
	}

	result, err := a.service.Search(ctx, req)
	if err != nil {
		return nil, nil, err
	}

	return result.IDs, result.Scores, nil
}

// ErrNotSupported 表示操作不支持的错误
var ErrNotSupported = &NotSupportedError{}

// NotSupportedError 操作不支持错误
type NotSupportedError struct{}

func (e *NotSupportedError) Error() string {
	return "operation not supported: use Search method instead"
}

// 确保 VectorStoreAdapter 实现了 recall.VectorStore 接口
var _ recall.VectorStore = (*VectorStoreAdapter)(nil)
