package milvus

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus/client/v2"
	"github.com/milvus-io/milvus/client/v2/entity"
)

// MilvusClient 是 Milvus SDK 客户端的接口抽象。
type MilvusClient interface {
	Search(ctx context.Context, collection string, vectors [][]float32, topK int64, metricType string, searchParams map[string]interface{}, filter string) ([]string, []float64, []float64, error)
	Insert(ctx context.Context, collection string, data []map[string]interface{}) error
	Delete(ctx context.Context, collection string, expr string) error
	CreateCollection(ctx context.Context, schema interface{}) error
	DropCollection(ctx context.Context, collection string) error
	HasCollection(ctx context.Context, collection string) (bool, error)
	Close() error
}

// MilvusClientFactory 是 Milvus 客户端工厂接口。
type MilvusClientFactory interface {
	NewClient(ctx context.Context, address string, username, password, database string, timeout time.Duration) (MilvusClient, error)
}

// DefaultMilvusClientFactory 是默认的 Milvus 客户端工厂。
type DefaultMilvusClientFactory struct{}

// NewClient 创建 Milvus SDK 客户端
func (f *DefaultMilvusClientFactory) NewClient(ctx context.Context, address, username, password, database string, timeout time.Duration) (MilvusClient, error) {
	config := &client.ClientConfig{
		Address:  address,
		Username: username,
		Password: password,
		DBName:   database,
	}
	milvusClient, err := client.New(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("create milvus client: %w", err)
	}
	return &MilvusSDKClientAdapter{client: milvusClient}, nil
}

// MilvusSDKClientAdapter 是 Milvus SDK 客户端的适配器。
type MilvusSDKClientAdapter struct {
	client *client.Client
}

func (a *MilvusSDKClientAdapter) Search(ctx context.Context, collection string, vectors [][]float32, topK int64, metricType string, searchParams map[string]interface{}, filter string) ([]string, []float64, []float64, error) {
	// 转换 metricType
	var metric entity.MetricType
	switch metricType {
	case "COSINE":
		metric = entity.COSINE
	case "L2":
		metric = entity.L2
	case "IP":
		metric = entity.IP
	default:
		metric = entity.COSINE
	}

	// 执行搜索
	searchResults, err := a.client.Search(ctx, collection, []string{}, filter, []string{"id"}, vectors, metric, topK, searchParams)
	if err != nil {
		return nil, nil, nil, err
	}

	ids := make([]string, 0)
	scores := make([]float64, 0)
	distances := make([]float64, 0)

	for _, result := range searchResults {
		for i, id := range result.IDs {
			var strID string
			switch v := id.(type) {
			case string:
				strID = v
			default:
				strID = fmt.Sprintf("%v", v)
			}
			ids = append(ids, strID)
			if i < len(result.Scores) {
				scores = append(scores, result.Scores[i])
			}
			if i < len(result.Distances) {
				distances = append(distances, result.Distances[i])
			}
		}
	}

	return ids, scores, distances, nil
}

func (a *MilvusSDKClientAdapter) Insert(ctx context.Context, collection string, data []map[string]interface{}) error {
	// 实现插入逻辑
	return fmt.Errorf("not implemented")
}

func (a *MilvusSDKClientAdapter) Delete(ctx context.Context, collection string, expr string) error {
	_, err := a.client.Delete(ctx, collection, expr, "")
	return err
}

func (a *MilvusSDKClientAdapter) CreateCollection(ctx context.Context, schema interface{}) error {
	// 实现创建集合逻辑
	return fmt.Errorf("not implemented")
}

func (a *MilvusSDKClientAdapter) DropCollection(ctx context.Context, collection string) error {
	return a.client.DropCollection(ctx, collection)
}

func (a *MilvusSDKClientAdapter) HasCollection(ctx context.Context, collection string) (bool, error) {
	return a.client.HasCollection(ctx, collection)
}

func (a *MilvusSDKClientAdapter) Close() error {
	return a.client.Close()
}