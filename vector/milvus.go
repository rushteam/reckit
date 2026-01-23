package vector

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/rushteam/reckit/core"
)

// MilvusService 是 Milvus 向量数据库的 ANNService 实现。
//
// 使用 Milvus 原生 VARCHAR 主键支持（Milvus 2.0+）：
//   - 直接使用 string IDs，无需转换为 int64
//   - 无哈希冲突风险
//   - 完全可逆（直接返回原始 ID）
//   - 性能更好（无转换开销）
//
// 创建集合时会自动使用 VARCHAR 主键类型。
type MilvusService struct {
	Address  string
	Username string
	Password string
	Database string
	Timeout  int

	client        MilvusClient
	clientFactory MilvusClientFactory
}

// NewMilvusService 创建一个新的 Milvus 服务实例。
func NewMilvusService(address string, opts ...MilvusOption) *MilvusService {
	service := &MilvusService{
		Address:       address,
		Database:      "default",
		Timeout:       30,
		clientFactory: &DefaultMilvusClientFactory{},
	}

	for _, opt := range opts {
		opt(service)
	}

	return service
}

// initClient 初始化 Milvus SDK 客户端
func (s *MilvusService) initClient(ctx context.Context) error {
	if s.client != nil {
		return nil
	}

	if s.clientFactory == nil {
		s.clientFactory = &DefaultMilvusClientFactory{}
	}

	timeout := time.Duration(s.Timeout) * time.Second
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	client, err := s.clientFactory.NewClient(ctx, s.Address, s.Username, s.Password, s.Database, timeout)
	if err != nil {
		return fmt.Errorf("init milvus client: %w", err)
	}

	s.client = client
	return nil
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

func WithMilvusClientFactory(factory MilvusClientFactory) MilvusOption {
	return func(s *MilvusService) {
		s.clientFactory = factory
	}
}

func WithMilvusClient(client MilvusClient) MilvusOption {
	return func(s *MilvusService) {
		s.client = client
	}
}

func (s *MilvusService) Search(ctx context.Context, req *SearchRequest) (*SearchResult, error) {
	if err := s.initClient(ctx); err != nil {
		return nil, err
	}

	if req.Collection == "" {
		return nil, fmt.Errorf("collection name is required")
	}
	if len(req.Vector) == 0 {
		return nil, fmt.Errorf("vector is required")
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}
	if !ValidateMetric(req.Metric) {
		req.Metric = string(MetricCosine)
	}

	milvusMetric := s.convertMetric(req.Metric)
	vectorFloat32 := convertToFloat32(req.Vector)

	searchParams := make(map[string]interface{})
	if req.Params != nil {
		searchParams = req.Params
	}

	filterExpr := ""
	if req.Filter != nil {
		filterExpr = s.buildFilterExpr(req.Filter)
	}

	// 直接返回 string IDs（使用 Milvus VARCHAR 主键，无需转换）
	ids, scores, distances, err := s.client.Search(
		ctx,
		req.Collection,
		[][]float32{vectorFloat32},
		int64(req.TopK),
		milvusMetric,
		searchParams,
		filterExpr,
	)
	if err != nil {
		return nil, fmt.Errorf("milvus search failed: %w", err)
	}

	return &SearchResult{
		IDs:       ids, // 直接使用 string IDs，无需转换
		Scores:    scores,
		Distances: distances,
	}, nil
}

func (s *MilvusService) buildFilterExpr(filter map[string]interface{}) string {
	exprs := make([]string, 0, len(filter))
	for k, v := range filter {
		switch val := v.(type) {
		case string:
			exprs = append(exprs, fmt.Sprintf("%s == '%s'", k, val))
		case int, int64:
			exprs = append(exprs, fmt.Sprintf("%s == %v", k, val))
		case float64:
			exprs = append(exprs, fmt.Sprintf("%s == %v", k, val))
		case bool:
			exprs = append(exprs, fmt.Sprintf("%s == %v", k, val))
		}
	}
	if len(exprs) == 0 {
		return ""
	}
	return strings.Join(exprs, " && ")
}

func (s *MilvusService) Insert(ctx context.Context, req *InsertRequest) error {
	if err := s.initClient(ctx); err != nil {
		return err
	}

	if req.Collection == "" {
		return fmt.Errorf("collection name is required")
	}
	if len(req.Vectors) == 0 {
		return fmt.Errorf("vectors are required")
	}
	if len(req.Vectors) != len(req.IDs) {
		return fmt.Errorf("vectors and ids length mismatch")
	}

	vectorsFloat32 := make([][]float32, len(req.Vectors))
	for i, v := range req.Vectors {
		vectorsFloat32[i] = convertToFloat32(v)
	}

	// 直接使用 string IDs（使用 Milvus VARCHAR 主键，无需转换）
	// Milvus 2.0+ 原生支持 VARCHAR 主键，可以直接使用 string ID
	data := []map[string]interface{}{
		{
			"id":     req.IDs, // 直接使用 string IDs
			"vector": vectorsFloat32,
		},
	}
	if len(req.Metadata) > 0 {
		data[0]["metadata"] = req.Metadata
	}

	err := s.client.Insert(ctx, req.Collection, data)
	if err != nil {
		return fmt.Errorf("milvus insert failed: %w", err)
	}

	return nil
}

func (s *MilvusService) Update(ctx context.Context, req *UpdateRequest) error {
	deleteReq := &DeleteRequest{
		Collection: req.Collection,
		IDs:        []string{req.ID},
	}
	if err := s.Delete(ctx, deleteReq); err != nil {
		return err
	}

	insertReq := &InsertRequest{
		Collection: req.Collection,
		Vectors:    [][]float64{req.Vector},
		IDs:        []string{req.ID},
		Metadata:   []map[string]interface{}{req.Metadata},
	}
	return s.Insert(ctx, insertReq)
}

func (s *MilvusService) Delete(ctx context.Context, req *DeleteRequest) error {
	if err := s.initClient(ctx); err != nil {
		return err
	}

	if req.Collection == "" {
		return fmt.Errorf("collection name is required")
	}
	if len(req.IDs) == 0 {
		return fmt.Errorf("ids are required")
	}

	expr := s.buildDeleteExpr(req.IDs)

	err := s.client.Delete(ctx, req.Collection, expr)
	if err != nil {
		return fmt.Errorf("milvus delete failed: %w", err)
	}

	return nil
}

func (s *MilvusService) buildDeleteExpr(ids []string) string {
	if len(ids) == 0 {
		return ""
	}
	if len(ids) == 1 {
		// VARCHAR 主键需要使用引号
		return fmt.Sprintf("id == '%s'", ids[0])
	}
	// VARCHAR 主键的批量删除，每个 ID 都需要引号
	quotedIDs := make([]string, len(ids))
	for i, id := range ids {
		quotedIDs[i] = fmt.Sprintf("'%s'", id)
	}
	return fmt.Sprintf("id in [%s]", strings.Join(quotedIDs, ", "))
}

func (s *MilvusService) CreateCollection(ctx context.Context, req *CreateCollectionRequest) error {
	if err := s.initClient(ctx); err != nil {
		return err
	}

	if req.Name == "" {
		return fmt.Errorf("collection name is required")
	}
	if req.Dimension <= 0 {
		return fmt.Errorf("dimension must be greater than 0")
	}
	if !ValidateMetric(req.Metric) {
		req.Metric = string(MetricCosine)
	}

	schema := s.buildSchema(req)

	err := s.client.CreateCollection(ctx, schema)
	if err != nil {
		return fmt.Errorf("milvus create collection failed: %w", err)
	}

	return nil
}

func (s *MilvusService) buildSchema(req *CreateCollectionRequest) interface{} {
	// 使用 Milvus VARCHAR 主键（支持原生 string ID）
	// Milvus 2.0+ 支持 VARCHAR 主键，无需转换为 int64
	schema := map[string]interface{}{
		"collection_name": req.Name,
		"dimension":       req.Dimension,
		"metric":          req.Metric,
		"primary_key": map[string]interface{}{
			"name":       "id",
			"data_type":  "VARCHAR",
			"max_length": 255, // VARCHAR 最大长度（1-65535）
			"is_primary": true,
		},
	}

	// 如果提供了额外参数，合并到 schema
	if req.Params != nil {
		for k, v := range req.Params {
			schema[k] = v
		}
	}

	return schema
}

func (s *MilvusService) DropCollection(ctx context.Context, collection string) error {
	if err := s.initClient(ctx); err != nil {
		return err
	}

	err := s.client.DropCollection(ctx, collection)
	if err != nil {
		return fmt.Errorf("milvus drop collection failed: %w", err)
	}

	return nil
}

func (s *MilvusService) HasCollection(ctx context.Context, collection string) (bool, error) {
	if err := s.initClient(ctx); err != nil {
		return false, err
	}

	exists, err := s.client.HasCollection(ctx, collection)
	if err != nil {
		return false, fmt.Errorf("milvus has collection failed: %w", err)
	}

	return exists, nil
}

func (s *MilvusService) Close() error {
	if s.client != nil {
		return s.client.Close()
	}
	return nil
}

func (s *MilvusService) convertMetric(metric string) string {
	switch metric {
	case string(MetricCosine):
		return "COSINE"
	case string(MetricEuclidean):
		return "L2"
	case string(MetricInnerProduct):
		return "IP"
	default:
		return "COSINE"
	}
}

func convertToFloat32(vec []float64) []float32 {
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = float32(v)
	}
	return result
}

var _ ANNService = (*MilvusService)(nil)

// SearchCore 实现 core.VectorService 接口
// 将 core.VectorSearchRequest 转换为内部 SearchRequest，然后调用内部实现
func (s *MilvusService) SearchCore(ctx context.Context, req *core.VectorSearchRequest) (*core.VectorSearchResult, error) {
	// 转换为内部 SearchRequest
	internalReq := &SearchRequest{
		Collection: req.Collection,
		Vector:     req.Vector,
		TopK:       req.TopK,
		Metric:     req.Metric,
		Filter:     req.Filter,
		Params:     req.Params,
	}

	// 调用内部实现（注意：这里调用的是 MilvusService.Search，接受 *SearchRequest）
	result, err := s.Search(ctx, internalReq)
	if err != nil {
		return nil, err
	}

	// 转换为 core.VectorSearchResult
	return &core.VectorSearchResult{
		IDs:       result.IDs,
		Scores:    result.Scores,
		Distances: result.Distances,
	}, nil
}

// 实现 core.VectorService 接口
// 注意：MilvusService 同时实现了两个接口：
// 1. ANNService（内部接口，用于完整的向量数据库操作，Search 方法接受 *SearchRequest）
// 2. core.VectorService（领域接口，用于召回场景，SearchCore 方法接受 *core.VectorSearchRequest）
//
// 为了满足 core.VectorService 接口，我们需要一个包装器
type milvusVectorServiceWrapper struct {
	*MilvusService
}

// Search 实现 core.VectorService 接口
func (w *milvusVectorServiceWrapper) Search(ctx context.Context, req *core.VectorSearchRequest) (*core.VectorSearchResult, error) {
	return w.MilvusService.SearchCore(ctx, req)
}

// NewMilvusVectorService 创建一个实现 core.VectorService 接口的包装器
func NewMilvusVectorService(service *MilvusService) core.VectorService {
	return &milvusVectorServiceWrapper{MilvusService: service}
}

var _ core.VectorService = (*milvusVectorServiceWrapper)(nil)
