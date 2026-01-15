package vector

import (
	"context"
	"fmt"
	"time"
)

// MilvusClient 是 Milvus SDK 客户端的接口抽象（遵循 DDD 原则，高内聚低耦合）。
//
// 这个接口定义了 Milvus 客户端需要实现的方法，不直接依赖具体 SDK。
// 实际实现可以通过依赖注入的方式提供。
//
// 使用方式：
//   - 方式1：直接使用 SDK（需要安装依赖）
//   - 方式2：通过依赖注入（推荐，保持低耦合）
type MilvusClient interface {
	// Search 向量搜索
	Search(ctx context.Context, collection string, vectors [][]float32, topK int64, metricType string, searchParams map[string]interface{}, filter string) ([]int64, []float64, []float64, error)

	// Insert 插入向量
	Insert(ctx context.Context, collection string, data []map[string]interface{}) error

	// Delete 删除向量
	Delete(ctx context.Context, collection string, expr string) error

	// CreateCollection 创建集合
	CreateCollection(ctx context.Context, schema interface{}) error

	// DropCollection 删除集合
	DropCollection(ctx context.Context, collection string) error

	// HasCollection 检查集合是否存在
	HasCollection(ctx context.Context, collection string) (bool, error)

	// Close 关闭连接
	Close() error
}

// MilvusClientFactory 是 Milvus 客户端工厂接口（用于依赖注入）。
type MilvusClientFactory interface {
	NewClient(ctx context.Context, address string, username, password, database string, timeout time.Duration) (MilvusClient, error)
}

// DefaultMilvusClientFactory 是默认的 Milvus 客户端工厂（使用 SDK）。
// 实际实现需要安装：go get github.com/milvus-io/milvus-sdk-go/v2/client
type DefaultMilvusClientFactory struct{}

// NewClient 创建 Milvus SDK 客户端
func (f *DefaultMilvusClientFactory) NewClient(ctx context.Context, address, username, password, database string, timeout time.Duration) (MilvusClient, error) {
	// 实际实现（需要安装 SDK）：
	// import "github.com/milvus-io/milvus-sdk-go/v2/client"
	// import "github.com/milvus-io/milvus-sdk-go/v2/entity"
	//
	// client, err := client.NewClient(ctx, client.Config{
	//     Address:  address,
	//     Username: username,
	//     Password: password,
	//     DBName:   database,
	// })
	// if err != nil {
	//     return nil, fmt.Errorf("create milvus client: %w", err)
	// }
	// return &MilvusSDKClientAdapter{client: client}, nil

	// 占位实现：返回错误提示安装 SDK
	return nil, fmt.Errorf("milvus SDK not available, please install: go get github.com/milvus-io/milvus-sdk-go/v2/client")
}

// MilvusSDKClientAdapter 是 Milvus SDK 客户端的适配器（实现 MilvusClient 接口）。
// 这个适配器将 SDK 的方法适配到我们的接口。
//
// 实际类型：*client.Client (github.com/milvus-io/milvus-sdk-go/v2/client)
type MilvusSDKClientAdapter struct {
	// client *client.Client
	client interface{} // 使用 interface{} 避免编译时依赖
}

// Search 向量搜索
func (a *MilvusSDKClientAdapter) Search(ctx context.Context, collection string, vectors [][]float32, topK int64, metricType string, searchParams map[string]interface{}, filter string) ([]int64, []float64, []float64, error) {
	// 实际实现（需要安装 SDK）：
	// import "github.com/milvus-io/milvus-sdk-go/v2/client"
	//
	// client := a.client.(*client.Client)
	// searchResult, err := client.Search(ctx, &client.SearchRequest{
	//     CollectionName: collection,
	//     Vectors:        vectors,
	//     TopK:           topK,
	//     MetricType:     metricType,
	//     SearchParams:   searchParams,
	//     Filter:         filter,
	// })
	// if err != nil {
	//     return nil, nil, nil, err
	// }
	//
	// // 提取结果
	// ids := make([]int64, 0)
	// scores := make([]float64, 0)
	// distances := make([]float64, 0)
	// for _, result := range searchResult {
	//     for i, id := range result.IDs {
	//         ids = append(ids, id)
	//         scores = append(scores, result.Scores[i])
	//         if len(result.Distances) > i {
	//             distances = append(distances, result.Distances[i])
	//         }
	//     }
	// }
	// return ids, scores, distances, nil

	return nil, nil, nil, fmt.Errorf("milvus SDK not available")
}

// Insert 插入向量
func (a *MilvusSDKClientAdapter) Insert(ctx context.Context, collection string, data []map[string]interface{}) error {
	// 实际实现（需要安装 SDK）：
	// import "github.com/milvus-io/milvus-sdk-go/v2/client"
	//
	// client := a.client.(*client.Client)
	// _, err := client.Insert(ctx, &client.InsertRequest{
	//     CollectionName: collection,
	//     Data:           data,
	// })
	// return err

	return fmt.Errorf("milvus SDK not available")
}

// Delete 删除向量
func (a *MilvusSDKClientAdapter) Delete(ctx context.Context, collection string, expr string) error {
	// 实际实现（需要安装 SDK）：
	// import "github.com/milvus-io/milvus-sdk-go/v2/client"
	//
	// client := a.client.(*client.Client)
	// _, err := client.Delete(ctx, &client.DeleteRequest{
	//     CollectionName: collection,
	//     Expr:           expr,
	// })
	// return err

	return fmt.Errorf("milvus SDK not available")
}

// CreateCollection 创建集合
func (a *MilvusSDKClientAdapter) CreateCollection(ctx context.Context, schema interface{}) error {
	// 实际实现（需要安装 SDK）：
	// import "github.com/milvus-io/milvus-sdk-go/v2/client"
	// import "github.com/milvus-io/milvus-sdk-go/v2/entity"
	//
	// client := a.client.(*client.Client)
	// err := client.CreateCollection(ctx, schema.(*entity.Schema), int32(2)) // shardNum
	// return err

	return fmt.Errorf("milvus SDK not available")
}

// DropCollection 删除集合
func (a *MilvusSDKClientAdapter) DropCollection(ctx context.Context, collection string) error {
	// 实际实现（需要安装 SDK）：
	// import "github.com/milvus-io/milvus-sdk-go/v2/client"
	//
	// client := a.client.(*client.Client)
	// err := client.DropCollection(ctx, collection)
	// return err

	return fmt.Errorf("milvus SDK not available")
}

// HasCollection 检查集合是否存在
func (a *MilvusSDKClientAdapter) HasCollection(ctx context.Context, collection string) (bool, error) {
	// 实际实现（需要安装 SDK）：
	// import "github.com/milvus-io/milvus-sdk-go/v2/client"
	//
	// client := a.client.(*client.Client)
	// exists, err := client.HasCollection(ctx, collection)
	// return exists, err

	return false, fmt.Errorf("milvus SDK not available")
}

// Close 关闭连接
func (a *MilvusSDKClientAdapter) Close() error {
	// 实际实现（需要安装 SDK）：
	// import "github.com/milvus-io/milvus-sdk-go/v2/client"
	//
	// client := a.client.(*client.Client)
	// return client.Close()

	return nil
}
