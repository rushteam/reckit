package vector

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// MilvusService 是 Milvus 向量数据库的 ANNService 实现。
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

	// 转换 int64 IDs 为 string IDs
	strIDs := make([]string, len(ids))
	for i, id := range ids {
		strIDs[i] = strconv.FormatInt(id, 10)
	}

	return &SearchResult{
		IDs:       strIDs,
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

	// 转换 string IDs 为 int64 IDs (Milvus 内部通常使用 int64)
	intIDs := make([]int64, len(req.IDs))
	for i, id := range req.IDs {
		// 尝试解析为 int64，如果失败则使用哈希值
		if parsedID, err := strconv.ParseInt(id, 10, 64); err == nil {
			intIDs[i] = parsedID
		} else {
			// 对于非数字 ID，使用字符串哈希值（简化处理）
			// 实际生产环境可能需要更复杂的映射策略
			hash := int64(0)
			for _, c := range id {
				hash = hash*31 + int64(c)
			}
			intIDs[i] = hash
		}
	}

	data := []map[string]interface{}{
		{
			"id":     intIDs,
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
		return fmt.Sprintf("id == %s", ids[0])
	}
	return fmt.Sprintf("id in [%s]", strings.Join(ids, ", "))
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
	return map[string]interface{}{
		"collection_name": req.Name,
		"dimension":       req.Dimension,
		"metric":          req.Metric,
	}
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
