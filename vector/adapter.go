package vector

import (
	"context"

	"reckit/recall"
)

// VectorStoreAdapter 将 ANNService 适配为 recall.VectorStore 接口。
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

func (a *VectorStoreAdapter) GetVector(ctx context.Context, itemID string) ([]float64, error) {
	return nil, ErrNotSupported
}

func (a *VectorStoreAdapter) ListVectors(ctx context.Context) (map[string][]float64, error) {
	return nil, ErrNotSupported
}

func (a *VectorStoreAdapter) Search(ctx context.Context, vector []float64, topK int, metric string) ([]string, []float64, error) {
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

type NotSupportedError struct{}

func (e *NotSupportedError) Error() string {
	return "operation not supported: use Search method instead"
}

var _ recall.VectorStore = (*VectorStoreAdapter)(nil)
