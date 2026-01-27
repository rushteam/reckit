package recall

import (
	"context"
	"math"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/conv"
	"github.com/rushteam/reckit/pkg/utils"
)

// ContentStore 是基于内容的推荐的存储接口。
type ContentStore interface {
	// GetItemFeatures 获取物品的内容特征（类别、标签、关键词等）
	GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)

	// GetUserPreferences 获取用户的偏好特征（喜欢的类别、标签等）
	GetUserPreferences(ctx context.Context, userID string) (map[string]float64, error)

	// GetSimilarItems 根据物品特征获取相似物品（可选，用于优化）
	GetSimilarItems(ctx context.Context, itemFeatures map[string]float64, topK int) ([]string, error)

	// GetAllItems 获取所有物品 ID 列表
	GetAllItems(ctx context.Context) ([]string, error)
}

// ContentRecall 是基于内容的召回源（Content-Based Recommendation）。
//
// 核心思想："用户喜欢具有某些特征的物品，推荐具有相似特征的其他物品"
type ContentRecall struct {
	Store ContentStore

	// TopK 返回 TopK 个物品
	TopK int

	// Metric 距离度量方式：cosine / jaccard / tfidf
	Metric string

	// UserPreferencesKey 从 RecommendContext 获取用户偏好的 key
	// 如果为空，则从 Store 获取
	UserPreferencesKey string

	// UserPreferencesExtractor 从 RecommendContext 提取用户偏好（可选）
	UserPreferencesExtractor func(rctx *core.RecommendContext) map[string]float64
}

func (r *ContentRecall) Name() string {
	return "recall.content"
}

func (r *ContentRecall) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Store == nil || rctx == nil || rctx.UserID == "" {
		return nil, nil
	}

	// 1. 获取用户偏好特征
	var userPrefs map[string]float64
	var err error

	// 优先从 UserPreferencesExtractor 获取
	if r.UserPreferencesExtractor != nil {
		userPrefs = r.UserPreferencesExtractor(rctx)
	} else if r.UserPreferencesKey != "" && rctx.UserProfile != nil {
		// 从 Context 获取
		if prefs, ok := rctx.UserProfile[r.UserPreferencesKey]; ok {
			if p, ok := prefs.(map[string]float64); ok {
				userPrefs = p
			} else if p, ok := prefs.(map[string]interface{}); ok {
				userPrefs = conv.MapToFloat64(p)
			}
		}
	}

	// 如果从 Context 获取失败，从 Store 获取
	if userPrefs == nil {
		userPrefs, err = r.Store.GetUserPreferences(ctx, rctx.UserID)
		if err != nil {
			return nil, err
		}
	}

	if len(userPrefs) == 0 {
		return nil, nil
	}

	// 2. 获取所有物品并计算相似度
	allItems, err := r.Store.GetAllItems(ctx)
	if err != nil {
		return nil, err
	}

	type scoredItem struct {
		itemID string
		score  float64
	}
	scores := make([]scoredItem, 0)

	metric := r.Metric
	if metric == "" {
		metric = "cosine"
	}

	for _, itemID := range allItems {
		// 获取物品特征
		itemFeatures, err := r.Store.GetItemFeatures(ctx, itemID)
		if err != nil || len(itemFeatures) == 0 {
			continue
		}

		// 计算匹配度
		var score float64
		switch metric {
		case "jaccard":
			score = jaccardSimilarity(userPrefs, itemFeatures)
		case "cosine":
			fallthrough
		default:
			score = cosineSimilarityForMaps(userPrefs, itemFeatures)
		}

		if score > 0 {
			scores = append(scores, scoredItem{
				itemID: itemID,
				score:  score,
			})
		}
	}

	// 3. 排序取 TopK
	topK := r.TopK
	if topK <= 0 {
		topK = 20
	}
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})
	if len(scores) > topK {
		scores = scores[:topK]
	}

	// 4. 封装结果
	out := make([]*core.Item, 0, len(scores))
	for _, s := range scores {
		it := core.NewItem(s.itemID)
		it.Score = s.score
		it.PutLabel("recall_source", utils.Label{Value: "content", Source: "recall"})
		it.PutLabel("recall_metric", utils.Label{Value: metric, Source: "recall"})
		out = append(out, it)
	}

	return out, nil
}

// cosineSimilarityForMaps 计算两个特征向量的余弦相似度
func cosineSimilarityForMaps(a, b map[string]float64) float64 {
	// 获取所有特征键的并集
	allKeys := make(map[string]bool)
	for k := range a {
		allKeys[k] = true
	}
	for k := range b {
		allKeys[k] = true
	}

	var dot, normA, normB float64
	for k := range allKeys {
		valA := a[k]
		valB := b[k]
		dot += valA * valB
		normA += valA * valA
		normB += valB * valB
	}

	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// jaccardSimilarity 计算两个特征向量的 Jaccard 相似度
func jaccardSimilarity(a, b map[string]float64) float64 {
	var intersection, union float64
	allKeys := make(map[string]bool)
	for k := range a {
		allKeys[k] = true
		union += a[k]
	}
	for k := range b {
		allKeys[k] = true
		if _, ok := a[k]; ok {
			intersection += math.Min(a[k], b[k])
		} else {
			union += b[k]
		}
	}

	if union == 0 {
		return 0
	}
	return intersection / union
}
