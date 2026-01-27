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
	// 使用新 SDK 的客户端创建方式
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

	// 构建搜索选项
	// 新 SDK 使用 SearchOption 模式
	searchOption := client.NewSearchOption(collection, int(topK), entityVectors).
		WithMetricType(metric).
		WithOutputFields([]string{"id"})

	// 添加过滤表达式
	if filter != "" {
		searchOption = searchOption.WithFilter(filter)
	}

	// 添加自定义搜索参数
	if searchParams != nil {
		if nprobe, ok := searchParams["nprobe"].(int); ok {
			searchOption = searchOption.WithNProbe(nprobe)
		}
		if ef, ok := searchParams["ef"].(int); ok {
			searchOption = searchOption.WithEF(ef)
		}
		if radius, ok := searchParams["radius"].(float64); ok {
			searchOption = searchOption.WithRadius(radius)
		}
		if rangeFilter, ok := searchParams["range_filter"].(float64); ok {
			searchOption = searchOption.WithRangeFilter(rangeFilter)
		}
	}

	// 执行搜索
	// 新 SDK 的 Search 方法签名: Search(ctx, option SearchOption, callOptions ...grpc.CallOption)
	searchResults, err := a.client.Search(ctx, searchOption)
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

	// 构建 entity.Column 格式的插入数据
	// 新 SDK 的 Insert 方法使用 columns
	idColumn := entity.NewColumnVarChar("id", ids)
	vectorColumn := entity.NewColumnFloatVector("vector", int32(len(vectors[0])), vectors)

	columns := []entity.Column{idColumn, vectorColumn}

	// 添加元数据列（如果有）
	if len(metadata) > 0 {
		for key, values := range extractMetadataColumns(metadata) {
			if col := buildMetadataColumn(key, values); col != nil {
				columns = append(columns, col)
			}
		}
	}

	// 执行插入
	// 新 SDK 的 Insert 方法签名: Insert(ctx, option InsertOption, callOptions ...grpc.CallOption)
	insertOption := client.NewInsertOption(collection, columns...)
	_, err := a.client.Insert(ctx, insertOption)
	if err != nil {
		return fmt.Errorf("milvus insert failed: %w", err)
	}

	return nil
}

// extractMetadataColumns 从元数据中提取列数据
func extractMetadataColumns(metadata []map[string]interface{}) map[string][]interface{} {
	if len(metadata) == 0 {
		return nil
	}

	// 获取所有键
	keys := make(map[string]bool)
	for _, m := range metadata {
		for k := range m {
			keys[k] = true
		}
	}

	// 构建列数据
	result := make(map[string][]interface{})
	for k := range keys {
		values := make([]interface{}, len(metadata))
		for i, m := range metadata {
			values[i] = m[k]
		}
		result[k] = values
	}

	return result
}

// buildMetadataColumn 根据值类型构建元数据列
func buildMetadataColumn(name string, values []interface{}) entity.Column {
	if len(values) == 0 {
		return nil
	}

	// 根据第一个值的类型判断列类型
	switch values[0].(type) {
	case string:
		strValues := make([]string, len(values))
		for i, v := range values {
			strValues[i] = fmt.Sprintf("%v", v)
		}
		return entity.NewColumnVarChar(name, strValues)
	case int, int32, int64:
		intValues := make([]int64, len(values))
		for i, v := range values {
			switch val := v.(type) {
			case int:
				intValues[i] = int64(val)
			case int32:
				intValues[i] = int64(val)
			case int64:
				intValues[i] = val
			}
		}
		return entity.NewColumnInt64(name, intValues)
	case float32, float64:
		floatValues := make([]float32, len(values))
		for i, v := range values {
			switch val := v.(type) {
			case float32:
				floatValues[i] = val
			case float64:
				floatValues[i] = float32(val)
			}
		}
		return entity.NewColumnFloat(name, floatValues)
	case bool:
		boolValues := make([]bool, len(values))
		for i, v := range values {
			boolValues[i] = v.(bool)
		}
		return entity.NewColumnBool(name, boolValues)
	default:
		// 默认转为字符串
		strValues := make([]string, len(values))
		for i, v := range values {
			strValues[i] = fmt.Sprintf("%v", v)
		}
		return entity.NewColumnVarChar(name, strValues)
	}
}

// Delete 删除向量数据
func (a *MilvusSDKClientAdapter) Delete(ctx context.Context, collection string, expr string) error {
	if expr == "" {
		return fmt.Errorf("delete expression is required")
	}

	// 新 SDK 的 Delete 方法使用 DeleteOption
	deleteOption := client.NewDeleteOption(collection, expr)
	_, err := a.client.Delete(ctx, deleteOption)
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
	// 新 SDK 的 CreateCollection 方法使用 CreateCollectionOption
	createOption := client.NewCreateCollectionOption(schemaDef)
	err := a.client.CreateCollection(ctx, createOption)
	if err != nil {
		return fmt.Errorf("milvus create collection failed: %w", err)
	}

	// 创建索引（使用 AUTOINDEX，自动选择最优索引）
	index, err := entity.NewIndexAUTOINDEX(entity.MetricType(metric))
	if err != nil {
		return fmt.Errorf("create index failed: %w", err)
	}

	// 新 SDK 的 CreateIndex 方法使用 CreateIndexOption
	indexOption := client.NewCreateIndexOption(collectionName, "vector", index)
	err = a.client.CreateIndex(ctx, indexOption)
	if err != nil {
		return fmt.Errorf("milvus create index failed: %w", err)
	}

	return nil
}

// DropCollection 删除集合
func (a *MilvusSDKClientAdapter) DropCollection(ctx context.Context, collection string) error {
	// 新 SDK 的 DropCollection 方法使用 DropCollectionOption
	dropOption := client.NewDropCollectionOption(collection)
	err := a.client.DropCollection(ctx, dropOption)
	if err != nil {
		return fmt.Errorf("milvus drop collection failed: %w", err)
	}
	return nil
}

// HasCollection 检查集合是否存在
func (a *MilvusSDKClientAdapter) HasCollection(ctx context.Context, collection string) (bool, error) {
	// 新 SDK 的 HasCollection 方法使用 HasCollectionOption
	hasOption := client.NewHasCollectionOption(collection)
	exists, err := a.client.HasCollection(ctx, hasOption)
	if err != nil {
		return false, fmt.Errorf("milvus has collection failed: %w", err)
	}
	return exists, nil
}

// Close 关闭客户端连接
func (a *MilvusSDKClientAdapter) Close() error {
	return a.client.Close()
}
