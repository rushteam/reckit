package vector

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/recall"
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

func (a *VectorStoreAdapter) Search(ctx context.Context, vector []float64, topK int, metric string) ([]core.VectorSearchItem, error) {
	req := &core.VectorSearchRequest{
		Collection: a.collection,
		Vector:     vector,
		TopK:       topK,
		Metric:     metric,
	}

	result, err := a.service.Search(ctx, req)
	if err != nil {
		return nil, err
	}

	return result.Items, nil
}

// ErrNotSupported 表示操作不支持的错误（使用统一的 DomainError）
var ErrNotSupported = core.NewDomainError(core.ModuleVector, core.ErrorCodeNotSupported, "vector: operation not supported: use Search method instead")

var _ recall.VectorStore = (*VectorStoreAdapter)(nil)
