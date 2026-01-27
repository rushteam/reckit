package store

import (
	"context"
	"math"
	"sort"
	"sync"

	"github.com/rushteam/reckit/core"
)

// MemoryVectorService 是内存实现的向量服务，用于测试/开发/原型。
// 平替 Milvus 等第三方向量数据库 SDK，支持向量搜索、插入、删除等操作。
//
// 特点：
//   - 纯内存实现，进程重启后数据丢失
//   - 支持余弦相似度、欧氏距离、内积等距离度量
//   - 线程安全
//   - 适用于测试、开发、小规模原型场景
type MemoryVectorService struct {
	mu         sync.RWMutex
	collections map[string]*collection // collection name -> collection data
}

type collection struct {
	name      string
	dimension int
	metric    string
	vectors   map[string][]float64 // item ID -> vector
	metadata  map[string]map[string]interface{} // item ID -> metadata
}

// NewMemoryVectorService 创建内存向量服务实例。
func NewMemoryVectorService() *MemoryVectorService {
	return &MemoryVectorService{
		collections: make(map[string]*collection),
	}
}

func (m *MemoryVectorService) Name() string { return "memory_vector" }

// Search 实现 core.VectorService 接口
func (m *MemoryVectorService) Search(ctx context.Context, req *core.VectorSearchRequest) (*core.VectorSearchResult, error) {
	if req == nil {
		return nil, core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "vector search request is nil")
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	col, ok := m.collections[req.Collection]
	if !ok {
		return &core.VectorSearchResult{Items: []core.VectorSearchItem{}}, nil
	}

	if len(req.Vector) != col.dimension {
		return nil, core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "vector dimension mismatch")
	}

	topK := req.TopK
	if topK <= 0 {
		topK = 10
	}

	metric := req.Metric
	if metric == "" {
		metric = col.metric
	}
	if metric == "" {
		metric = "cosine"
	}

	// 计算所有向量的相似度
	type scoredItem struct {
		id    string
		score float64
	}
	scoredItems := make([]scoredItem, 0, len(col.vectors))

	for itemID, itemVector := range col.vectors {
		// 应用过滤条件
		if req.Filter != nil {
			if !m.matchFilter(req.Filter, col.metadata[itemID]) {
				continue
			}
		}

		var score float64
		switch metric {
		case "cosine":
			score = cosineSimilarity(req.Vector, itemVector)
		case "euclidean":
			// 欧氏距离转换为相似度分数（距离越小，分数越高）
			distance := euclideanDistance(req.Vector, itemVector)
			score = 1.0 / (1.0 + distance)
		case "inner_product":
			score = innerProduct(req.Vector, itemVector)
		default:
			score = cosineSimilarity(req.Vector, itemVector)
		}

		scoredItems = append(scoredItems, scoredItem{
			id:    itemID,
			score: score,
		})
	}

	// 按分数降序排序
	sort.Slice(scoredItems, func(i, j int) bool {
		return scoredItems[i].score > scoredItems[j].score
	})

	// 取 TopK
	if len(scoredItems) > topK {
		scoredItems = scoredItems[:topK]
	}

	// 转换为结果
	items := make([]core.VectorSearchItem, len(scoredItems))
	for i, item := range scoredItems {
		items[i] = core.VectorSearchItem{
			ID:       item.id,
			Score:    item.score,
			Distance: 1.0 - item.score, // 简化的距离计算
		}
	}

	return &core.VectorSearchResult{Items: items}, nil
}

// Close 实现 core.VectorService 接口
func (m *MemoryVectorService) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.collections = make(map[string]*collection)
	return nil
}

// Insert 实现 core.VectorDatabaseService 接口
func (m *MemoryVectorService) Insert(ctx context.Context, req *core.VectorInsertRequest) error {
	if req == nil {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "insert request is nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	col, ok := m.collections[req.Collection]
	if !ok {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeNotFound, "collection not found: "+req.Collection)
	}

	if len(req.Vectors) != len(req.IDs) {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "vectors and ids length mismatch")
	}

	for i, vector := range req.Vectors {
		if len(vector) != col.dimension {
			return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "vector dimension mismatch")
		}
		col.vectors[req.IDs[i]] = vector
		if len(req.Metadata) > i {
			col.metadata[req.IDs[i]] = req.Metadata[i]
		}
	}

	return nil
}

// Update 实现 core.VectorDatabaseService 接口
func (m *MemoryVectorService) Update(ctx context.Context, req *core.VectorUpdateRequest) error {
	if req == nil {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "update request is nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	col, ok := m.collections[req.Collection]
	if !ok {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeNotFound, "collection not found: "+req.Collection)
	}

	if len(req.Vector) != col.dimension {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "vector dimension mismatch")
	}

	col.vectors[req.ID] = req.Vector
	if req.Metadata != nil {
		col.metadata[req.ID] = req.Metadata
	}

	return nil
}

// Delete 实现 core.VectorDatabaseService 接口
func (m *MemoryVectorService) Delete(ctx context.Context, req *core.VectorDeleteRequest) error {
	if req == nil {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "delete request is nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	col, ok := m.collections[req.Collection]
	if !ok {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeNotFound, "collection not found: "+req.Collection)
	}

	for _, id := range req.IDs {
		delete(col.vectors, id)
		delete(col.metadata, id)
	}

	return nil
}

// CreateCollection 实现 core.VectorDatabaseService 接口
func (m *MemoryVectorService) CreateCollection(ctx context.Context, req *core.VectorCreateCollectionRequest) error {
	if req == nil {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "create collection request is nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if req.Name == "" {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "collection name is required")
	}

	if req.Dimension <= 0 {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "dimension must be greater than 0")
	}

	if _, exists := m.collections[req.Name]; exists {
		return core.NewDomainError(core.ModuleVector, core.ErrorCodeInvalidInput, "collection already exists: "+req.Name)
	}

	metric := req.Metric
	if !core.ValidateVectorMetric(metric) {
		metric = "cosine"
	}

	m.collections[req.Name] = &collection{
		name:      req.Name,
		dimension: req.Dimension,
		metric:    metric,
		vectors:   make(map[string][]float64),
		metadata:  make(map[string]map[string]interface{}),
	}

	return nil
}

// DropCollection 实现 core.VectorDatabaseService 接口
func (m *MemoryVectorService) DropCollection(ctx context.Context, collection string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.collections, collection)
	return nil
}

// HasCollection 实现 core.VectorDatabaseService 接口
func (m *MemoryVectorService) HasCollection(ctx context.Context, collection string) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	_, exists := m.collections[collection]
	return exists, nil
}

// matchFilter 检查元数据是否匹配过滤条件
func (m *MemoryVectorService) matchFilter(filter map[string]interface{}, metadata map[string]interface{}) bool {
	if metadata == nil {
		return false
	}

	for key, filterValue := range filter {
		metaValue, ok := metadata[key]
		if !ok {
			return false
		}

		// 简单的相等比较
		if metaValue != filterValue {
			return false
		}
	}

	return true
}

// 相似度计算函数

// cosineSimilarity 计算余弦相似度
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// euclideanDistance 计算欧氏距离
func euclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}

	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

// innerProduct 计算内积
func innerProduct(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}

// 确保实现了接口
var (
	_ core.VectorService = (*MemoryVectorService)(nil)
	_ core.VectorDatabaseService = (*MemoryVectorService)(nil)
)
