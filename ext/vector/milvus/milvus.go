package milvus

import (
	"context"
	"fmt"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"

	"github.com/rushteam/reckit/core"
)

// MilvusService 是 Milvus 向量数据库的 VectorDatabaseService 实现。
//
// 注意：此实现位于扩展包中，需要单独引入：
//
//	go get github.com/rushteam/reckit/ext/vector/milvus
type MilvusService struct {
	Address  string
	Username string
	Password string
	Database string
	Timeout  int
	client   *milvusclient.Client
}

// NewMilvusService 创建一个新的 Milvus 服务实例。
func NewMilvusService(address string, opts ...MilvusOption) (*MilvusService, error) {
	service := &MilvusService{
		Address:  address,
		Database: "default",
		Timeout:  30,
	}

	for _, opt := range opts {
		opt(service)
	}

	// 初始化客户端
	ctx := context.Background()
	config := &milvusclient.ClientConfig{
		Address:  service.Address,
		Username: service.Username,
		Password: service.Password,
		DBName:   service.Database,
	}

	client, err := milvusclient.New(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("create milvus client: %w", err)
	}

	service.client = client
	return service, nil
}

type MilvusOption func(*MilvusService)

func WithMilvusAuth(username, password string) MilvusOption {
	return func(s *MilvusService) {
		s.Username = username
		s.Password = password
	}
}

func WithMilvusDatabase(database string) MilvusOption {
	return func(s *MilvusService) {
		s.Database = database
	}
}

func WithMilvusTimeout(timeout int) MilvusOption {
	return func(s *MilvusService) {
		s.Timeout = timeout
	}
}

// Search 实现 core.VectorService 接口
func (s *MilvusService) Search(ctx context.Context, req *core.VectorSearchRequest) (*core.VectorSearchResult, error) {
	if req.Collection == "" {
		return nil, fmt.Errorf("collection name is required")
	}
	if len(req.Vector) == 0 {
		return nil, fmt.Errorf("vector is required")
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}
	if !core.ValidateVectorMetric(req.Metric) {
		req.Metric = "cosine"
	}

	// 转换向量为 float32
	vectorFloat32 := convertToFloat32(req.Vector)

	// 构建搜索选项
	// 注意：v2.5.x SDK 中，metric type 在创建 collection 时已确定，搜索时自动使用
	// 不需要也不支持在搜索时指定 metric type
	searchOption := milvusclient.NewSearchOption(req.Collection, int(req.TopK), []entity.Vector{entity.FloatVector(vectorFloat32)}).
		WithOutputFields("id")

	// 添加过滤表达式 (使用模板参数方式，符合新版 SDK 规范)
	if len(req.Filter) > 0 {
		exprs := make([]string, 0, len(req.Filter))
		for k, v := range req.Filter {
			paramName := "f_" + k
			exprs = append(exprs, fmt.Sprintf("%s == $%s", k, paramName))
			searchOption = searchOption.WithTemplateParam(paramName, v)
		}
		searchOption = searchOption.WithFilter(strings.Join(exprs, " && "))
	}

	// 添加自定义搜索参数
	if len(req.Params) > 0 {
		annParam := index.NewCustomAnnParam()
		hasParams := false
		if nprobe, ok := req.Params["nprobe"].(int); ok {
			annParam.WithExtraParam("nprobe", nprobe)
			hasParams = true
		}
		if ef, ok := req.Params["ef"].(int); ok {
			annParam.WithExtraParam("ef", ef)
			hasParams = true
		}
		if radius, ok := req.Params["radius"].(float64); ok {
			annParam.WithExtraParam("radius", radius)
			hasParams = true
		}
		if rangeFilter, ok := req.Params["range_filter"].(float64); ok {
			annParam.WithExtraParam("range_filter", rangeFilter)
			hasParams = true
		}
		if hasParams {
			searchOption = searchOption.WithAnnParam(annParam)
		}
	}

	// 执行搜索
	searchResults, err := s.client.Search(ctx, searchOption)
	if err != nil {
		return nil, fmt.Errorf("milvus search failed: %w", err)
	}

	// 提取结果
	items := make([]core.VectorSearchItem, 0)

	for _, resultSet := range searchResults {
		if resultSet.Err != nil {
			continue
		}

		for i := 0; i < resultSet.Len(); i++ {
			// 提取 ID
			id, _ := resultSet.IDs.Get(i)
			var strID string
			switch v := id.(type) {
			case string:
				strID = v
			case int64:
				strID = fmt.Sprintf("%d", v)
			default:
				strID = fmt.Sprintf("%v", v)
			}

			item := core.VectorSearchItem{
				ID: strID,
			}

			// 提取分数 (v2.5.x 中 Scores 包含了距离信息)
			if i < len(resultSet.Scores) {
				score := float64(resultSet.Scores[i])
				item.Score = score
				item.Distance = score
			}
			items = append(items, item)
		}
	}

	return &core.VectorSearchResult{
		Items: items,
	}, nil
}

func (s *MilvusService) Insert(ctx context.Context, req *core.VectorInsertRequest) error {
	if req.Collection == "" {
		return fmt.Errorf("collection name is required")
	}
	if len(req.Vectors) == 0 {
		return fmt.Errorf("vectors are required")
	}
	if len(req.Vectors) != len(req.IDs) {
		return fmt.Errorf("vectors and ids length mismatch")
	}

	// 转换向量为 float32
	vectorsFloat32 := make([][]float32, len(req.Vectors))
	for i, v := range req.Vectors {
		vectorsFloat32[i] = convertToFloat32(v)
	}

	// 构建列数据
	idColumn := column.NewColumnVarChar("id", req.IDs)
	vectorColumn := column.NewColumnFloatVector("vector", len(vectorsFloat32[0]), vectorsFloat32)

	columns := []column.Column{idColumn, vectorColumn}

	// 添加元数据列（如果有）
	if len(req.Metadata) > 0 {
		for key, values := range extractMetadataColumns(req.Metadata) {
			if col := buildMetadataColumn(key, values); col != nil {
				columns = append(columns, col)
			}
		}
	}

	// 执行插入
	insertOption := milvusclient.NewColumnBasedInsertOption(req.Collection, columns...)
	_, err := s.client.Insert(ctx, insertOption)
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
func buildMetadataColumn(name string, values []interface{}) column.Column {
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
		return column.NewColumnVarChar(name, strValues)
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
		return column.NewColumnInt64(name, intValues)
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
		return column.NewColumnFloat(name, floatValues)
	case bool:
		boolValues := make([]bool, len(values))
		for i, v := range values {
			boolValues[i] = v.(bool)
		}
		return column.NewColumnBool(name, boolValues)
	default:
		// 默认转为字符串
		strValues := make([]string, len(values))
		for i, v := range values {
			strValues[i] = fmt.Sprintf("%v", v)
		}
		return column.NewColumnVarChar(name, strValues)
	}
}

func (s *MilvusService) Update(ctx context.Context, req *core.VectorUpdateRequest) error {
	deleteReq := &core.VectorDeleteRequest{
		Collection: req.Collection,
		IDs:        []string{req.ID},
	}
	if err := s.Delete(ctx, deleteReq); err != nil {
		return err
	}

	insertReq := &core.VectorInsertRequest{
		Collection: req.Collection,
		Vectors:    [][]float64{req.Vector},
		IDs:        []string{req.ID},
		Metadata:   []map[string]interface{}{req.Metadata},
	}
	return s.Insert(ctx, insertReq)
}

func (s *MilvusService) Delete(ctx context.Context, req *core.VectorDeleteRequest) error {
	if req.Collection == "" {
		return fmt.Errorf("collection name is required")
	}
	if len(req.IDs) == 0 {
		return fmt.Errorf("ids are required")
	}

	deleteOption := milvusclient.NewDeleteOption(req.Collection).WithStringIDs("id", req.IDs)
	_, err := s.client.Delete(ctx, deleteOption)
	if err != nil {
		return fmt.Errorf("milvus delete failed: %w", err)
	}

	return nil
}

func (s *MilvusService) CreateCollection(ctx context.Context, req *core.VectorCreateCollectionRequest) error {
	if req.Name == "" {
		return fmt.Errorf("collection name is required")
	}
	if req.Dimension <= 0 {
		return fmt.Errorf("dimension must be greater than 0")
	}
	if !core.ValidateVectorMetric(req.Metric) {
		req.Metric = string(core.MetricCosine)
	}

	// 转换 metric
	metric := s.convertMetricType(req.Metric)

	// 构建 Schema
	schemaDef := entity.NewSchema().
		WithName(req.Name).
		WithField(entity.NewField().
			WithName("id").
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(255).
			WithIsPrimaryKey(true)).
		WithField(entity.NewField().
			WithName("vector").
			WithDataType(entity.FieldTypeFloatVector).
			WithDim(int64(req.Dimension)))

	// 创建索引选项（使用 AUTOINDEX）
	indexOpt := milvusclient.NewCreateIndexOption(req.Name, "vector", index.NewAutoIndex(metric))

	// 创建集合
	createOption := milvusclient.NewCreateCollectionOption(req.Name, schemaDef).
		WithIndexOptions(indexOpt)

	err := s.client.CreateCollection(ctx, createOption)
	if err != nil {
		return fmt.Errorf("milvus create collection failed: %w", err)
	}

	return nil
}

func (s *MilvusService) DropCollection(ctx context.Context, collection string) error {
	dropOption := milvusclient.NewDropCollectionOption(collection)
	err := s.client.DropCollection(ctx, dropOption)
	if err != nil {
		return fmt.Errorf("milvus drop collection failed: %w", err)
	}
	return nil
}

func (s *MilvusService) HasCollection(ctx context.Context, collection string) (bool, error) {
	hasOption := milvusclient.NewHasCollectionOption(collection)
	exists, err := s.client.HasCollection(ctx, hasOption)
	if err != nil {
		return false, fmt.Errorf("milvus has collection failed: %w", err)
	}
	return exists, nil
}

func (s *MilvusService) Close() error {
	if s.client != nil {
		ctx := context.Background()
		return s.client.Close(ctx)
	}
	return nil
}

func (s *MilvusService) convertMetricType(metric string) entity.MetricType {
	switch metric {
	case "cosine":
		return entity.COSINE
	case "euclidean":
		return entity.L2
	case "inner_product":
		return entity.IP
	default:
		return entity.COSINE
	}
}

func convertToFloat32(vec []float64) []float32 {
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = float32(v)
	}
	return result
}

var (
	_ core.VectorService         = (*MilvusService)(nil)
	_ core.VectorDatabaseService = (*MilvusService)(nil)
)
