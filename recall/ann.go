package recall

import (
	"context"
	"math"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

// ANN 是 Embedding 向量检索召回源（Approximate Nearest Neighbor）。
// 支持余弦相似度、欧氏距离等计算方式。
type ANN struct {
	Store      VectorStore // 向量存储（可以是 Redis、内存、向量数据库等）
	Key        string      // 向量索引 key，例如 "embedding:items"
	UserVector []float64   // 用户向量（如果提供，优先使用；否则从 rctx 获取）
	TopK       int         // 返回 TopK 相似物品
	Metric     string      // 距离度量：cosine / euclidean

	// UserVectorExtractor 从 RecommendContext 提取用户向量（可选）
	UserVectorExtractor func(rctx *core.RecommendContext) []float64
}

// VectorStore 是向量存储接口（简化版）。
type VectorStore interface {
	// GetVector 获取物品向量
	GetVector(ctx context.Context, itemID string) ([]float64, error)

	// ListVectors 获取所有向量（用于暴力搜索）
	ListVectors(ctx context.Context) (map[string][]float64, error)

	// Search 使用向量搜索
	Search(ctx context.Context, vector []float64, topK int, metric string) ([]string, []float64, error)
}

func (r *ANN) Name() string { return "recall.emb" } // 工业标准命名：emb (Embedding)

// EmbRecall 是 ANN 的类型别名，提供更符合工业习惯的命名。
type EmbRecall = ANN

func (r *ANN) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Store == nil {
		return nil, nil
	}

	// 1. 获取用户向量
	userVector := r.UserVector
	if len(userVector) == 0 {
		if r.UserVectorExtractor != nil {
			userVector = r.UserVectorExtractor(rctx)
		} else if rctx != nil && rctx.User != nil {
			// 假设用户画像中存了向量（实际工程中常用）
			// 这里仅为示例
		} else if rctx != nil && rctx.UserProfile != nil {
			// 从 UserProfile 获取用户向量
			if uv, ok := rctx.UserProfile["user_vector"]; ok {
				if vec, ok := uv.([]float64); ok {
					userVector = vec
				} else if vec, ok := uv.([]interface{}); ok {
					// 转换为 []float64
					userVector = make([]float64, 0, len(vec))
					for _, v := range vec {
						if fv, ok := v.(float64); ok {
							userVector = append(userVector, fv)
						}
					}
				}
			}
		}
	}

	// 如果仍然没有，尝试通过 UserID 从 Store 获取
	if len(userVector) == 0 && rctx != nil && rctx.UserID != "" {
		var err error
		userVector, err = r.Store.GetVector(ctx, rctx.UserID)
		if err != nil {
			// 忽略错误，可能用户没有向量
		}
	}

	if len(userVector) == 0 {
		return nil, nil
	}

	// 2. 执行向量搜索
	topK := r.TopK
	if topK <= 0 {
		topK = 10
	}

	var ids []string
	var scores []float64
	var err error

	// 优先尝试 Search 方法（高性能）
	ids, scores, err = r.Store.Search(ctx, userVector, topK, r.Metric)
	if err != nil {
		// 如果 Search 不支持或报错，尝试暴力搜索（ListVectors）
		allVectors, err2 := r.Store.ListVectors(ctx)
		if err2 != nil {
			return nil, err
		}

		// 暴力搜索逻辑
		type scoredItem struct {
			itemID string
			score  float64
		}
		results := make([]scoredItem, 0, len(allVectors))

		for itemID, itemVec := range allVectors {
			var sim float64
			if r.Metric == "euclidean" {
				sim = 1.0 / (1.0 + euclideanDistanceVector(userVector, itemVec))
			} else {
				sim = cosineSimilarityVectorForANN(userVector, itemVec)
			}
			results = append(results, scoredItem{itemID: itemID, score: sim})
		}

		sort.Slice(results, func(i, j int) bool {
			return results[i].score > results[j].score
		})

		if len(results) > topK {
			results = results[:topK]
		}

		ids = make([]string, len(results))
		scores = make([]float64, len(results))
		for i, res := range results {
			ids[i] = res.itemID
			scores[i] = res.score
		}
	}

	// 3. 封装结果
	out := make([]*core.Item, 0, len(ids))
	for i, id := range ids {
		it := core.NewItem(id)
		it.Score = scores[i]
		it.PutLabel("recall_source", utils.Label{Value: "ann", Source: "recall"})
		it.PutLabel("ann_metric", utils.Label{Value: r.Metric, Source: "recall"})
		out = append(out, it)
	}

	return out, nil
}

func cosineSimilarityVectorForANN(a, b []float64) float64 {
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

func euclideanDistanceVector(a, b []float64) float64 {
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
