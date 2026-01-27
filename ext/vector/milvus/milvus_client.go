package milvus

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
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
	config := client.Config{
		Address:  address,
		Username: username,
		Password: password,
		DBName:   database,
	}
	milvusClient, err := client.NewClient(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("create milvus client: %w", err)
	}
	return &MilvusSDKClientAdapter{client: milvusClient}, nil
}

// MilvusSDKClientAdapter 是 Milvus SDK 客户端的适配器。
type MilvusSDKClientAdapter struct {
	client client.Client
}

// Search 执行向量搜索
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

	// 转换向量为 entity.Vector 类型
	entityVectors := make([]entity.Vector, len(vectors))
	for i, v := range vectors {
		entityVectors[i] = entity.FloatVector(v)
	}

	// 构建搜索参数
	searchParam := entity.NewSearchParam().
		WithTopK(int(topK)).
		WithMetricType(metric)

	// 添加自定义搜索参数
	if searchParams != nil {
		if nprobe, ok := searchParams["nprobe"].(int); ok {
			searchParam = searchParam.WithNProbe(nprobe)
		}
		if ef, ok := searchParams["ef"].(int); ok {
			searchParam = searchParam.WithEF(ef)
		}
		if radius, ok := searchParams["radius"].(float64); ok {
			searchParam = searchParam.WithRadius(radius)
		}
		if rangeFilter, ok := searchParams["range_filter"].(float64); ok {
			searchParam = searchParam.WithRangeFilter(rangeFilter)
		}
	}

	// 执行搜索
	// Milvus SDK v2 的 Search 方法签名: Search(ctx, collection, partitions, expr, outputFields, vectors, fieldName, metricType, topK, searchParam)
	searchResults, err := a.client.Search(
		ctx,
		collection,
		[]string{},     // partitions
		filter,         // expr (filter expression)
		[]string{"id"}, // outputFields
		entityVectors,
		"vector", // fieldName
		metric,
		int(topK),
		searchParam,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("milvus search failed: %w", err)
	}

	// 提取结果
	ids := make([]string, 0)
	scores := make([]float64, 0)
	distances := make([]float64, 0)

	for _, result := range searchResults {
		for i, id := range result.IDs {
			var strID string
			switch v := id.(type) {
			case string:
				strID = v
			case int64:
				strID = fmt.Sprintf("%d", v)
			default:
				strID = fmt.Sprintf("%v", v)
			}
			ids = append(ids, strID)

			// 提取分数和距离
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

// Insert 插入向量数据
func (a *MilvusSDKClientAdapter) Insert(ctx context.Context, collection string, data []map[string]interface{}) error {
	if len(data) == 0 {
		return fmt.Errorf("data is empty")
	}

	// 提取数据
	var ids []string
	var vectors [][]float32
	var metadata []map[string]interface{}

	// 从第一个数据项中提取结构
	firstItem := data[0]

	// 提取 IDs
	if idValue, ok := firstItem["id"]; ok {
		if idSlice, ok := idValue.([]string); ok {
			ids = idSlice
		} else if idSlice, ok := idValue.([]interface{}); ok {
			ids = make([]string, len(idSlice))
			for i, v := range idSlice {
				ids[i] = fmt.Sprintf("%v", v)
			}
		}
	}

	// 提取向量
	if vectorValue, ok := firstItem["vector"]; ok {
		if vectorSlice, ok := vectorValue.([][]float32); ok {
			vectors = vectorSlice
		} else if vectorSlice, ok := vectorValue.([][]float64); ok {
			vectors = make([][]float32, len(vectorSlice))
			for i, v := range vectorSlice {
				vectors[i] = make([]float32, len(v))
				for j, f := range v {
					vectors[i][j] = float32(f)
				}
			}
		}
	}

	// 提取元数据
	if metaValue, ok := firstItem["metadata"]; ok {
		if metaSlice, ok := metaValue.([]map[string]interface{}); ok {
			metadata = metaSlice
		}
	}

	// 构建插入数据（新 SDK 使用 []map[string]interface{} 格式）
	insertData := make([]map[string]interface{}, len(ids))
	for i := range ids {
		insertData[i] = make(map[string]interface{})
		insertData[i]["id"] = ids[i]
		insertData[i]["vector"] = vectors[i]
		if i < len(metadata) {
			for k, v := range metadata[i] {
				insertData[i][k] = v
			}
		}
	}

	// 执行插入
	// 新 SDK 的 Insert 方法签名: Insert(ctx, collection, data, ...options)
	_, err := a.client.Insert(ctx, collection, insertData)
	if err != nil {
		return fmt.Errorf("milvus insert failed: %w", err)
	}

	return nil
}

// Delete 删除向量数据
func (a *MilvusSDKClientAdapter) Delete(ctx context.Context, collection string, expr string) error {
	if expr == "" {
		return fmt.Errorf("delete expression is required")
	}

	// 新 SDK 的 Delete 方法签名: Delete(ctx, collection, expr, partitionName)
	_, err := a.client.Delete(ctx, collection, expr, "")
	if err != nil {
		return fmt.Errorf("milvus delete failed: %w", err)
	}

	return nil
}

// CreateCollection 创建集合
func (a *MilvusSDKClientAdapter) CreateCollection(ctx context.Context, schema interface{}) error {
	// 将 schema 转换为 entity.Schema
	schemaMap, ok := schema.(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid schema type, expected map[string]interface{}")
	}

	collectionName, ok := schemaMap["collection_name"].(string)
	if !ok {
		return fmt.Errorf("collection_name is required")
	}

	dimension, ok := schemaMap["dimension"].(int)
	if !ok {
		return fmt.Errorf("dimension is required")
	}

	metricStr, ok := schemaMap["metric"].(string)
	if !ok {
		metricStr = "cosine"
	}

	// 转换 metric
	var metric entity.MetricType
	switch metricStr {
	case "cosine", "COSINE":
		metric = entity.COSINE
	case "euclidean", "L2":
		metric = entity.L2
	case "inner_product", "IP":
		metric = entity.IP
	default:
		metric = entity.COSINE
	}

	// 构建 Schema
	schemaDef := entity.NewSchema().
		WithName(collectionName).
		WithField(entity.NewField().
			WithName("id").
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(255).
			WithIsPrimaryKey(true)).
		WithField(entity.NewField().
			WithName("vector").
			WithDataType(entity.FieldTypeFloatVector).
			WithDim(dimension))

	// 创建集合
	// Milvus SDK v2 的 CreateCollection 方法签名: CreateCollection(ctx, schema, shardNum)
	err := a.client.CreateCollection(ctx, schemaDef, entity.DefaultShardNumber)
	if err != nil {
		return fmt.Errorf("milvus create collection failed: %w", err)
	}

	// 创建索引（使用 AUTOINDEX，自动选择最优索引）
	index, err := entity.NewIndexAUTOINDEX(entity.MetricType(metric))
	if err != nil {
		return fmt.Errorf("create index failed: %w", err)
	}

	// Milvus SDK v2 的 CreateIndex 方法签名: CreateIndex(ctx, collection, fieldName, index, async)
	err = a.client.CreateIndex(ctx, collectionName, "vector", index, false)
	if err != nil {
		return fmt.Errorf("milvus create index failed: %w", err)
	}

	return nil
}

// DropCollection 删除集合
func (a *MilvusSDKClientAdapter) DropCollection(ctx context.Context, collection string) error {
	err := a.client.DropCollection(ctx, collection)
	if err != nil {
		return fmt.Errorf("milvus drop collection failed: %w", err)
	}
	return nil
}

// HasCollection 检查集合是否存在
func (a *MilvusSDKClientAdapter) HasCollection(ctx context.Context, collection string) (bool, error) {
	exists, err := a.client.HasCollection(ctx, collection)
	if err != nil {
		return false, fmt.Errorf("milvus has collection failed: %w", err)
	}
	return exists, nil
}

// Close 关闭客户端连接
func (a *MilvusSDKClientAdapter) Close() error {
	return a.client.Close()
}
