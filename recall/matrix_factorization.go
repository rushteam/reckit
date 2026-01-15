package recall

import (
	"context"

	"reckit/core"
	"reckit/pkg/utils"
)

// MFStore 是矩阵分解的存储接口，用于获取用户和物品的隐向量。
type MFStore interface {
	// GetUserVector 获取用户的隐向量
	GetUserVector(ctx context.Context, userID int64) ([]float64, error)

	// GetItemVector 获取物品的隐向量
	GetItemVector(ctx context.Context, itemID int64) ([]float64, error)

	// GetAllItemVectors 获取所有物品的隐向量（用于在线召回）
	GetAllItemVectors(ctx context.Context) (map[int64][]float64, error)
}

// MFRecall 是基于矩阵分解（Matrix Factorization）的召回源。
//
// 核心思想：将用户-物品交互矩阵分解为用户隐向量和物品隐向量
// 预测分数 = 用户隐向量 · 物品隐向量
//
// 算法类型：
//  - MF (Matrix Factorization): 基础矩阵分解
//  - ALS (Alternating Least Squares): 交替最小二乘法
//  - SVD (Singular Value Decomposition): 奇异值分解
//
// 工程特征：
//  - 实时性：好（离线训练，在线查表）
//  - 计算复杂度：低（向量点积）
//  - 可解释性：中等
//  - 冷启动：中等
//
// 在 Reckit 中的位置：
//  - 核心 Recall Node（MFRecall）
//  - Label：recall.mf
//
// 使用场景：
//  - 输入：用户隐向量（离线训练得到）
//  - 输出：TopK 物品（通过向量点积计算）
type MFRecall struct {
	Store MFStore

	// TopK 返回 TopK 个物品
	TopK int

	// UserVectorKey 从 RecommendContext 获取用户隐向量的 key
	// 如果为空，则从 Store 获取
	UserVectorKey string

	// UserVectorExtractor 从 RecommendContext 提取用户隐向量（可选）
	UserVectorExtractor func(rctx *core.RecommendContext) []float64
}

func (r *MFRecall) Name() string {
	return "recall.mf"
}

func (r *MFRecall) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Store == nil || rctx == nil || rctx.UserID == 0 {
		return nil, nil
	}

	// 获取用户隐向量
	var userVector []float64
	var err error

	// 优先从 UserVectorExtractor 获取
	if r.UserVectorExtractor != nil {
		userVector = r.UserVectorExtractor(rctx)
	} else if r.UserVectorKey != "" && rctx.UserProfile != nil {
		// 从 Context 获取
		if uv, ok := rctx.UserProfile[r.UserVectorKey]; ok {
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

	// 如果从 Context 获取失败，从 Store 获取
	if len(userVector) == 0 {
		userVector, err = r.Store.GetUserVector(ctx, rctx.UserID)
		if err != nil {
			return nil, err
		}
	}

	if len(userVector) == 0 {
		return nil, nil
	}

	// 获取所有物品隐向量
	allItemVectors, err := r.Store.GetAllItemVectors(ctx)
	if err != nil {
		return nil, err
	}

	// 计算用户向量与所有物品向量的点积（预测分数）
	type scoredItem struct {
		itemID int64
		score  float64
	}
	scores := make([]scoredItem, 0, len(allItemVectors))

	for itemID, itemVector := range allItemVectors {
		// 计算点积：用户向量 · 物品向量
		score := dotProduct(userVector, itemVector)
		scores = append(scores, scoredItem{
			itemID: itemID,
			score:  score,
		})
	}

	// 排序取 TopK
	topK := r.TopK
	if topK <= 0 {
		topK = 20
	}
	if len(scores) > topK {
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
		it.PutLabel("recall_source", utils.Label{Value: "mf", Source: "recall"})
		out = append(out, it)
	}

	return out, nil
}

// dotProduct 计算两个向量的点积
func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
