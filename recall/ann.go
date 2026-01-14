package recall

import (
	"context"
	"math"

	"reckit/core"
	"reckit/pkg/utils"
)

// ANN 是 Embedding 向量检索召回源（Approximate Nearest Neighbor）。
// 支持余弦相似度、欧氏距离等计算方式。
// 支持从 RecommendContext 获取用户向量，实现个性化召回。
type ANN struct {
	Store      VectorStore // 向量存储（可以是 Redis、内存、向量数据库等）
	Key        string      // 向量索引 key，例如 "embedding:items"
	UserVector []float64   // 用户向量（如果提供，优先使用；否则从 rctx 获取）
	TopK       int         // 返回 TopK 相似物品
	Metric     string      // 距离度量：cosine / euclidean

	// UserVectorExtractor 从 RecommendContext 提取用户向量（可选）
	// 如果未提供且 UserVector 为空，则从 rctx.UserProfile["user_vector"] 获取
	UserVectorExtractor func(rctx *core.RecommendContext) []float64
}

// VectorStore 是向量存储接口（简化版，生产环境可用专业向量数据库如 Faiss、Milvus）。
type VectorStore interface {
	GetVector(ctx context.Context, itemID int64) ([]float64, error)
	ListVectors(ctx context.Context) (map[int64][]float64, error)
}

func (r *ANN) Name() string { return "recall.ann" }

func (r *ANN) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Store == nil {
		return nil, nil
	}

	// 获取用户向量
	userVector := r.UserVector
	if len(userVector) == 0 {
		if r.UserVectorExtractor != nil {
			userVector = r.UserVectorExtractor(rctx)
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

	if len(userVector) == 0 {
		return nil, nil
	}

	// 获取所有物品向量（简化实现，生产环境应使用向量索引如 Faiss、Milvus）
	allVectors, err := r.Store.ListVectors(ctx)
	if err != nil {
		return nil, err
	}

	// 计算相似度并排序
	type scoredItem struct {
		itemID int64
		score  float64
	}
	scores := make([]scoredItem, 0, len(allVectors))

	for itemID, itemVec := range allVectors {
		var sim float64
		switch r.Metric {
		case "cosine":
			sim = cosineSimilarity(r.UserVector, itemVec)
		case "euclidean":
			sim = 1.0 / (1.0 + euclideanDistance(r.UserVector, itemVec))
		default:
			sim = cosineSimilarity(r.UserVector, itemVec)
		}
		scores = append(scores, scoredItem{itemID: itemID, score: sim})
	}

	// 排序取 TopK
	topK := r.TopK
	if topK <= 0 {
		topK = 10
	}
	if len(scores) > topK {
		// 简单选择排序取 TopK（生产环境可用堆）
		for i := 0; i < topK; i++ {
			maxIdx := i
			for j := i + 1; j < len(scores); j++ {
				if scores[j].score > scores[maxIdx].score {
					maxIdx = j
				}
			}
			scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
		}
		scores = scores[:topK]
	}

	// 构建结果
	out := make([]*core.Item, 0, len(scores))
	for _, s := range scores {
		it := core.NewItem(s.itemID)
		it.Score = s.score
		it.PutLabel("recall_source", utils.Label{Value: "ann", Source: "recall"})
		it.PutLabel("ann_metric", utils.Label{Value: r.Metric, Source: "recall"})
		out = append(out, it)
	}

	return out, nil
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
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
