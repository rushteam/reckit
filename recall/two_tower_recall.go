package recall

import (
	"context"
	"fmt"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/pkg/conv"
	"github.com/rushteam/reckit/pkg/utils"
)

// TwoTowerRecall 是基于双塔模型的召回源实现。
//
// 核心流程：
//   1. 获取用户特征（通过 FeatureService）
//   2. 运行用户塔推理（通过 MLService，如 ONNX Runtime、TorchServe）
//   3. 向量检索（通过 ANNService，如 Milvus、Faiss）
//
// 设计原则：
//   - 高内聚：TwoTowerRecall 只负责协调流程，具体逻辑在各服务中
//   - 低耦合：通过接口依赖，可替换实现（FeatureService、MLService、ANNService）
//   - DDD 模式：每个服务都是独立的领域服务
//
// 使用示例：
//
//	// 1. 创建特征服务
//	featureService := feature.NewFeatureService(...)
//
//	// 2. 创建用户塔推理服务（ONNX Runtime 或 TorchServe）
//	userTowerService := service.NewTorchServeClient("http://localhost:8080", "user_tower")
//
//	// 3. 创建向量检索服务（Milvus）
//	vectorService := vector.NewMilvusService("localhost:19530")
//
//	// 4. 创建双塔召回源
//	twoTowerRecall := recall.NewTwoTowerRecall(
//		featureService,
//		userTowerService,
//		vectorService,
//		recall.WithTwoTowerTopK(100),
//		recall.WithTwoTowerCollection("item_embeddings"),
//	)
//
//	// 5. 在 Fanout 中使用
//	fanout := &recall.Fanout{
//		Sources: []recall.Source{
//			twoTowerRecall,
//			&recall.Hot{IDs: []string{"1", "2", "3"}},
//		},
//	}
type TwoTowerRecall struct {
	// FeatureService 特征服务，用于获取用户特征
	FeatureService feature.FeatureService

	// UserTowerService 用户塔推理服务（core.MLService 接口）
	// 支持 ONNX Runtime、TorchServe、TensorFlow Serving 等
	UserTowerService core.MLService

	// VectorService 向量检索服务（core.VectorService 接口）
	// 支持 Milvus、Faiss 等向量数据库
	// 注意：使用领域接口（core.VectorService），由基础设施层（vector）实现
	VectorService core.VectorService

	// TopK 返回 TopK 个物品
	TopK int

	// Collection 向量数据库集合名称（用于存储 Item Embeddings）
	Collection string

	// Metric 距离度量方式：cosine / euclidean / inner_product
	// 默认：inner_product（内积，适合双塔模型）
	Metric string

	// UserFeatureExtractor 自定义用户特征提取器（可选）
	// 如果为 nil，则使用 FeatureService.GetUserFeatures
	UserFeatureExtractor func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error)
}

// NewTwoTowerRecall 创建一个新的双塔召回源。
func NewTwoTowerRecall(
	featureService feature.FeatureService,
	userTowerService core.MLService,
	vectorService core.VectorService,
	opts ...TwoTowerRecallOption,
) *TwoTowerRecall {
	recall := &TwoTowerRecall{
		FeatureService:    featureService,
		UserTowerService:  userTowerService,
		VectorService:     vectorService,
		TopK:              100,
		Collection:        "item_embeddings",
		Metric:            "inner_product", // 默认使用内积
	}

	for _, opt := range opts {
		opt(recall)
	}

	return recall
}

// TwoTowerRecallOption 双塔召回配置选项
type TwoTowerRecallOption func(*TwoTowerRecall)

// WithTwoTowerTopK 设置 TopK
func WithTwoTowerTopK(topK int) TwoTowerRecallOption {
	return func(r *TwoTowerRecall) {
		r.TopK = topK
	}
}

// WithTwoTowerCollection 设置向量数据库集合名称
func WithTwoTowerCollection(collection string) TwoTowerRecallOption {
	return func(r *TwoTowerRecall) {
		r.Collection = collection
	}
}

// WithTwoTowerMetric 设置距离度量方式
func WithTwoTowerMetric(metric string) TwoTowerRecallOption {
	return func(r *TwoTowerRecall) {
		// 验证距离度量方式
		if core.ValidateVectorMetric(metric) {
			r.Metric = metric
		}
	}
}

// WithTwoTowerUserFeatureExtractor 设置自定义用户特征提取器
func WithTwoTowerUserFeatureExtractor(extractor func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error)) TwoTowerRecallOption {
	return func(r *TwoTowerRecall) {
		r.UserFeatureExtractor = extractor
	}
}

func (r *TwoTowerRecall) Name() string {
	return "recall.two_tower"
}

// Recall 实现 Source 接口，执行双塔召回流程
func (r *TwoTowerRecall) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	// 1. 获取用户特征
	userFeatures, err := r.getUserFeatures(ctx, rctx)
	if err != nil {
		return nil, fmt.Errorf("get user features: %w", err)
	}

	// 2. 运行用户塔推理，得到 User Embedding
	userEmbedding, err := r.runUserTower(ctx, userFeatures)
	if err != nil {
		return nil, fmt.Errorf("run user tower: %w", err)
	}

	// 3. 向量检索，找到相似的 Item Embeddings
	searchResult, err := r.searchVectors(ctx, userEmbedding)
	if err != nil {
		return nil, fmt.Errorf("search vectors: %w", err)
	}

	// 4. 转换为 Item 列表
	items := r.convertToItems(searchResult)

	return items, nil
}

// getUserFeatures 获取用户特征
func (r *TwoTowerRecall) getUserFeatures(
	ctx context.Context,
	rctx *core.RecommendContext,
) (map[string]float64, error) {
	// 使用自定义提取器（如果设置）
	if r.UserFeatureExtractor != nil {
		return r.UserFeatureExtractor(ctx, rctx)
	}

	// 使用 FeatureService（如果设置）
	if r.FeatureService != nil {
		return r.FeatureService.GetUserFeatures(ctx, rctx.UserID)
	}

	// 默认：从 RecommendContext 提取
	return r.extractUserFeaturesFromContext(rctx), nil
}

// extractUserFeaturesFromContext 从 RecommendContext 提取用户特征（默认实现）
func (r *TwoTowerRecall) extractUserFeaturesFromContext(rctx *core.RecommendContext) map[string]float64 {
	features := make(map[string]float64)

	if rctx == nil {
		return features
	}

	// 从 UserProfile 提取
	if rctx.User != nil {
		features["age"] = float64(rctx.User.Age)
		if rctx.User.Gender == "male" {
			features["gender"] = 1.0
		} else {
			features["gender"] = 0.0
		}
		for tag, score := range rctx.User.Interests {
			features["interest_"+tag] = score
		}
	}

	// 从 UserProfile map 提取
	if rctx.UserProfile != nil {
		for k, v := range rctx.UserProfile {
			if fv, ok := conv.ToFloat64(v); ok {
				features[k] = fv
			}
		}
	}

	// 从 Realtime 提取
	if rctx.Realtime != nil {
		for k, v := range rctx.Realtime {
			if fv, ok := conv.ToFloat64(v); ok {
				features["realtime_"+k] = fv
			}
		}
	}

	return features
}

// runUserTower 运行用户塔推理，返回 User Embedding
func (r *TwoTowerRecall) runUserTower(
	ctx context.Context,
	userFeatures map[string]float64,
) ([]float64, error) {
	if r.UserTowerService == nil {
		return nil, fmt.Errorf("user tower service is required")
	}

	// 调用 MLService 进行推理
	// 注意：优先使用 Features（字典格式），避免特征顺序问题
	req := &core.MLPredictRequest{
		Features: []map[string]float64{userFeatures}, // 字典格式（推荐，避免特征顺序问题）
	}

	resp, err := r.UserTowerService.Predict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("user tower predict: %w", err)
	}

	if len(resp.Predictions) == 0 {
		// 尝试从 Outputs 中提取 embedding 向量
		if resp.Outputs != nil {
			if embedding, ok := r.extractEmbeddingFromOutputs(resp.Outputs); ok {
				return embedding, nil
			}
		}
		return nil, fmt.Errorf("empty predictions from user tower")
	}

	// 检查返回的是 embedding 向量还是单个分数
	// 如果返回的是单个值，尝试从 Outputs 中提取 embedding
	if len(resp.Predictions) == 1 {
		// 尝试从 Outputs 中提取 embedding 向量
		if resp.Outputs != nil {
			if embedding, ok := r.extractEmbeddingFromOutputs(resp.Outputs); ok {
				return embedding, nil
			}
		}
		// 如果无法提取，返回错误
		return nil, fmt.Errorf("user tower returned single value (score: %f), expected embedding vector. "+
			"Please configure the model to return embedding vector from the last hidden layer", resp.Predictions[0])
	}

	// 返回 embedding 向量
	// 注意：Predictions 的长度应该等于 embedding 维度（例如：32、64、128）
	return resp.Predictions, nil
}

// extractEmbeddingFromOutputs 从 Outputs 中提取 embedding 向量
// 支持多种格式：数组、嵌套数组、对象等
func (r *TwoTowerRecall) extractEmbeddingFromOutputs(outputs interface{}) ([]float64, bool) {
	switch val := outputs.(type) {
	case []float64:
		// 直接是 float64 数组
		return val, true
	case []interface{}:
		// 是 interface{} 数组，需要转换
		embedding := make([]float64, 0, len(val))
		for _, item := range val {
			if fv, ok := conv.ToFloat64(item); ok {
				embedding = append(embedding, fv)
			}
		}
		if len(embedding) > 0 {
			return embedding, true
		}
	case map[string]interface{}:
		// 是对象，尝试查找常见的键名
		if embedding, ok := val["embedding"].([]interface{}); ok {
			return r.extractEmbeddingFromOutputs(embedding)
		}
		if embedding, ok := val["vector"].([]interface{}); ok {
			return r.extractEmbeddingFromOutputs(embedding)
		}
		if embedding, ok := val["output"].([]interface{}); ok {
			return r.extractEmbeddingFromOutputs(embedding)
		}
		// 尝试查找数值数组
		for _, v := range val {
			if embedding, ok := r.extractEmbeddingFromOutputs(v); ok {
				return embedding, true
			}
		}
	case string:
		// 如果是 JSON 字符串，尝试解析（简化处理，实际可能需要完整解析）
		// 这里不处理，由调用方处理
	}

	return nil, false
}

// searchVectors 向量检索，找到相似的 Item Embeddings
func (r *TwoTowerRecall) searchVectors(
	ctx context.Context,
	userEmbedding []float64,
) (*core.VectorSearchResult, error) {
	if r.VectorService == nil {
		return nil, fmt.Errorf("vector service is required")
	}

	req := &core.VectorSearchRequest{
		Collection: r.Collection,
		Vector:     userEmbedding,
		TopK:       r.TopK,
		Metric:     r.Metric,
	}

	result, err := r.VectorService.Search(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("vector search: %w", err)
	}

	return result, nil
}

// convertToItems 将搜索结果转换为 Item 列表
func (r *TwoTowerRecall) convertToItems(result *core.VectorSearchResult) []*core.Item {
	if result == nil || len(result.IDs) == 0 {
		return []*core.Item{}
	}

	items := make([]*core.Item, 0, len(result.IDs))
	for i, itemID := range result.IDs {
		item := core.NewItem(itemID)

		// 设置相似度分数
		if i < len(result.Scores) {
			item.Score = result.Scores[i]
		} else if i < len(result.Distances) {
			// 如果只有距离，转换为相似度分数（距离越小，相似度越高）
			// 对于内积，距离 = 1 - score，所以 score = 1 - distance
			item.Score = 1.0 - result.Distances[i]
		}

		// 添加召回来源标签
		item.PutLabel("recall_source", utils.Label{Value: "two_tower", Source: "recall"})
		item.PutLabel("recall_type", utils.Label{Value: "vector_search", Source: "recall"})
		item.PutLabel("recall_collection", utils.Label{Value: r.Collection, Source: "recall"})

		items = append(items, item)
	}

	return items
}

// 确保 TwoTowerRecall 实现了 Source 接口
var _ Source = (*TwoTowerRecall)(nil)
