package recall

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

// ANN 是 Embedding 向量检索召回源（Approximate Nearest Neighbor）。
// 支持余弦相似度、欧氏距离等计算方式。
//
// 设计原则：
//   - 直接使用 core.VectorService（领域层接口），符合 DDD 原则
//   - 消除过度抽象：不再需要 VectorStore 适配层
//   - 专注于高性能向量检索场景
type ANN struct {
	// VectorService 向量检索服务（领域层接口）
	// 可以是 Milvus、Faiss 等向量数据库的实现
	VectorService core.VectorService

	// Collection 集合名称（用于向量搜索）
	Collection string

	// UserEmbedding 用户向量（如果提供，优先使用；否则从 rctx 获取）
	UserEmbedding []float64

	// TopK 返回 TopK 相似物品
	TopK int

	// Metric 距离度量：cosine / euclidean / inner_product
	Metric string

	// UserEmbeddingExtractor 从 RecommendContext 提取用户向量（可选）
	UserEmbeddingExtractor func(rctx *core.RecommendContext) []float64
}

func (r *ANN) Name() string { return "recall.emb" } // 工业标准命名：emb (Embedding)

// EmbRecall 是 ANN 的类型别名，提供更符合工业习惯的命名。
type EmbRecall = ANN

func (r *ANN) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.VectorService == nil {
		return nil, nil
	}

	// 1. 获取用户向量
	userEmbedding := r.UserEmbedding
	if len(userEmbedding) == 0 {
		if r.UserEmbeddingExtractor != nil {
			userEmbedding = r.UserEmbeddingExtractor(rctx)
		} else if rctx != nil && rctx.UserProfile != nil {
			// 从 UserProfile 获取用户向量
			if uv, ok := rctx.UserProfile["user_embedding"]; ok {
				if vec, ok := uv.([]float64); ok {
					userEmbedding = vec
				} else if vec, ok := uv.([]interface{}); ok {
					// 转换为 []float64
					userEmbedding = make([]float64, 0, len(vec))
					for _, v := range vec {
						if fv, ok := v.(float64); ok {
							userEmbedding = append(userEmbedding, fv)
						}
					}
				}
			}
		}
	}

	if len(userEmbedding) == 0 {
		return nil, nil
	}

	// 2. 执行向量搜索
	topK := r.TopK
	if topK <= 0 {
		topK = 10
	}

	metric := r.Metric
	if metric == "" {
		metric = "cosine"
	}

	collection := r.Collection
	if collection == "" {
		collection = "items" // 默认集合名
	}

	// 使用 core.VectorService 进行向量搜索
	searchReq := &core.VectorSearchRequest{
		Collection: collection,
		Vector:     userEmbedding,
		TopK:       topK,
		Metric:     metric,
	}

	searchResult, err := r.VectorService.Search(ctx, searchReq)
	if err != nil {
		return nil, err
	}

	// 3. 封装结果
	out := make([]*core.Item, 0, len(searchResult.Items))
	for _, item := range searchResult.Items {
		it := core.NewItem(item.ID)
		it.Score = item.Score
		it.PutLabel("recall_source", utils.Label{Value: "ann", Source: "recall"})
		if metric != "" {
			it.PutLabel("recall_metric", utils.Label{Value: metric, Source: "recall"})
		}
		out = append(out, it)
	}

	return out, nil
}

