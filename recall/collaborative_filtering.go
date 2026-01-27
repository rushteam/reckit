package recall

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

// SimilarityCalculator 是相似度计算接口，用于自定义相似度计算方法。
type SimilarityCalculator interface {
	// Calculate 计算两个向量的相似度
	// x, y: 两个向量（必须长度相同）
	// 返回: 相似度值（通常在 -1 到 1 之间）
	Calculate(x, y []float64) float64
}

// CosineSimilarity 是余弦相似度计算器。
type CosineSimilarity struct{}

func (c *CosineSimilarity) Calculate(x, y []float64) float64 {
	return cosineSimilarityVector(x, y)
}

// PearsonCorrelation 是皮尔逊相关系数计算器。
type PearsonCorrelation struct{}

func (p *PearsonCorrelation) Calculate(x, y []float64) float64 {
	return pearsonCorrelation(x, y)
}

// CFStore 是协同过滤的存储接口（已废弃，使用 core.RecallDataStore）。
// 为了向后兼容，保留为类型别名。
type CFStore = core.RecallDataStore

// UserBasedCF 是基于用户的协同过滤召回源（User-based Collaborative Filtering, User-CF）。
//
// 核心思想："兴趣相似的用户，喜欢相似的物品"
//
// 算法流程：
//  1. 用户 → 行为向量（点击/收藏/购买）
//  2. 计算用户相似度（Cosine / Pearson）
//  3. 找 TopK 相似用户
//  4. 推荐这些用户喜欢但目标用户未见过的物品
//
// 工程特征：
//  - 实时性：较差（用户变化快）
//  - 计算复杂度：高（用户数大）
//  - 可解释性：强
//  - 冷启动：差
//
// 工程使用现状：
//  - ❌ 几乎不直接在线用
//  - ✅ 离线分析 / 冷启动补充
//
// 在 Reckit 中的位置：
//  - 离线产出 u2u / u2i 结果
//  - 作为 Recall Node（u2u → u2i 工程拆分）
type UserBasedCF struct {
	Store CFStore

	// TopKSimilarUsers 计算相似度时考虑的 TopK 个相似用户
	// 如果 <= 0，则使用 Config 中的默认值
	TopKSimilarUsers int

	// TopKItems 最终返回的 TopK 个物品
	// 如果 <= 0，则使用 Config 中的默认值
	TopKItems int

	// SimilarityCalculator 相似度计算器（必需）
	// 使用内置计算器：CosineSimilarity、PearsonCorrelation
	// 或实现自定义计算器
	SimilarityCalculator SimilarityCalculator

	// MinCommonItems 两个用户至少需要有多少个共同交互物品才计算相似度
	// 如果 <= 0，则使用 Config 中的默认值
	MinCommonItems int
	
	// Config 召回配置（必需）
	// 提供默认值，不能为 nil
	Config core.RecallConfig
}

func (r *UserBasedCF) Name() string {
	return "recall.u2i" // 工业标准命名：u2i (User-to-Item)
}

func (r *UserBasedCF) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Store == nil || rctx == nil || rctx.UserID == "" {
		return nil, nil
	}

	// 获取目标用户的交互物品
	targetUserItems, err := r.Store.GetUserItems(ctx, rctx.UserID)
	if err != nil {
		return nil, err
	}
	if len(targetUserItems) == 0 {
		return nil, nil
	}

	// 获取所有用户
	allUsers, err := r.Store.GetAllUsers(ctx)
	if err != nil {
		return nil, err
	}

	// 计算与目标用户的相似度
	type userSimilarity struct {
		userID    string
		similarity float64
	}
	similarities := make([]userSimilarity, 0)

	topKSimilar := r.TopKSimilarUsers
	if topKSimilar <= 0 {
		topKSimilar = r.Config.DefaultTopKSimilarUsers()
	}

	minCommon := r.MinCommonItems
	if minCommon <= 0 {
		minCommon = r.Config.DefaultMinCommonItems()
	}

	// 验证必需字段
	if r.SimilarityCalculator == nil {
		return nil, fmt.Errorf("SimilarityCalculator is required")
	}
	if r.Config == nil {
		return nil, fmt.Errorf("Config is required")
	}

	// 计算每个用户与目标用户的相似度
	for _, userID := range allUsers {
		if userID == rctx.UserID {
			continue // 跳过自己
		}

		userItems, err := r.Store.GetUserItems(ctx, userID)
		if err != nil || len(userItems) == 0 {
			continue
		}

		// 计算共同物品
		commonItems := make(map[string]struct{})
		targetScores := make([]float64, 0)
		userScores := make([]float64, 0)

		for itemID, targetScore := range targetUserItems {
			if userScore, ok := userItems[itemID]; ok {
				commonItems[itemID] = struct{}{}
				targetScores = append(targetScores, targetScore)
				userScores = append(userScores, userScore)
			}
		}

		// 如果共同物品太少，跳过
		if len(commonItems) < minCommon {
			continue
		}

		// 计算相似度
		sim := r.SimilarityCalculator.Calculate(targetScores, userScores)

		if sim > 0 { // 只保留正相似度
			similarities = append(similarities, userSimilarity{
				userID:    userID,
				similarity: sim,
			})
		}
	}

	// 排序取 TopK 相似用户
	if len(similarities) > topKSimilar {
		sort.Slice(similarities, func(i, j int) bool {
			return similarities[i].similarity > similarities[j].similarity
		})
		similarities = similarities[:topKSimilar]
	} else {
		sort.Slice(similarities, func(i, j int) bool {
			return similarities[i].similarity > similarities[j].similarity
		})
	}

	// 收集相似用户喜欢的物品（加权）
	// score[itemID] = Σ(similarity * userScore)
	itemScores := make(map[string]float64)
	for _, sim := range similarities {
		userItems, err := r.Store.GetUserItems(ctx, sim.userID)
		if err != nil {
			continue
		}

		for itemID, score := range userItems {
			// 跳过目标用户已经交互过的物品
			if _, ok := targetUserItems[itemID]; ok {
				continue
			}
			// 加权累加：相似度 * 用户评分
			itemScores[itemID] += sim.similarity * score
		}
	}

	// 转换为排序列表
	type scoredItem struct {
		itemID string
		score  float64
	}
	scoredItems := make([]scoredItem, 0, len(itemScores))
	for itemID, score := range itemScores {
		scoredItems = append(scoredItems, scoredItem{
			itemID: itemID,
			score:  score,
		})
	}

	// 排序取 TopK
	topK := r.TopKItems
	if topK <= 0 {
		topK = 20
	}
	sort.Slice(scoredItems, func(i, j int) bool {
		return scoredItems[i].score > scoredItems[j].score
	})
	if len(scoredItems) > topK {
		scoredItems = scoredItems[:topK]
	}

	// 构建结果
	out := make([]*core.Item, 0, len(scoredItems))
	for _, s := range scoredItems {
		it := core.NewItem(s.itemID)
		it.Score = s.score
		it.PutLabel("recall_source", utils.Label{Value: "u2i", Source: "recall"}) // u2u → u2i
		out = append(out, it)
	}

	return out, nil
}

// ItemBasedCF 是基于物品的协同过滤召回源（Item-based Collaborative Filtering, Item-CF）。
//
// 核心思想："被同一批用户喜欢的物品，相互相似"
//
// 算法流程：
//  1. 构建物品 → 用户倒排表
//  2. 计算物品相似度
//  3. 对用户历史行为物品，取相似物品集合
//
// 工程特征：
//  - 实时性：好
//  - 计算复杂度：可控
//  - 可解释性：强
//  - 稳定性：高
//
// 工业地位：
//  - 工业级召回的"常青树"
//  - 电商、内容流、短视频都在用
//  - 可直接线上使用
//
// 在 Reckit 中的位置：
//  - 核心 Recall Node（i2iRecall）
//  - Label：recall.i2i
//
// 使用场景：
//  - 输入：用户最近点击 items
//  - 输出：相似 items
//  - "我看了这个，还可能看什么"
type ItemBasedCF struct {
	Store CFStore

	// TopKSimilarItems 计算相似度时考虑的 TopK 个相似物品
	// 如果 <= 0，则使用 Config 中的默认值
	TopKSimilarItems int

	// TopKItems 最终返回的 TopK 个物品
	// 如果 <= 0，则使用 Config 中的默认值
	TopKItems int

	// SimilarityCalculator 相似度计算器（必需）
	// 使用内置计算器：CosineSimilarity、PearsonCorrelation
	// 或实现自定义计算器
	SimilarityCalculator SimilarityCalculator

	// MinCommonUsers 两个物品至少需要有多少个共同交互用户才计算相似度
	// 如果 <= 0，则使用 Config 中的默认值
	MinCommonUsers int

	// UserHistoryKey 从 RecommendContext 获取用户历史物品的 key
	// 如果为空，则从 Store 获取用户的所有交互物品
	UserHistoryKey string
	
	// Config 召回配置（必需）
	// 提供默认值，不能为 nil
	Config core.RecallConfig
}

func (r *ItemBasedCF) Name() string {
	return "recall.i2i" // 工业标准命名：i2i (Item-to-Item)
}

func (r *ItemBasedCF) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Store == nil || rctx == nil || rctx.UserID == "" {
		return nil, nil
	}

	// 获取用户的历史交互物品
	var userItems map[string]float64
	var err error

	if r.UserHistoryKey != "" && rctx.UserProfile != nil {
		// 从 Context 获取用户历史
		if history, ok := rctx.UserProfile[r.UserHistoryKey]; ok {
			if items, ok := history.(map[string]float64); ok {
				userItems = items
			} else if items, ok := history.(map[string]interface{}); ok {
				userItems = make(map[string]float64)
				for k, v := range items {
					if score, ok := v.(float64); ok {
						userItems[k] = score
					}
				}
			}
		}
	}

	// 如果从 Context 获取失败，从 Store 获取
	if userItems == nil {
		userItems, err = r.Store.GetUserItems(ctx, rctx.UserID)
		if err != nil {
			return nil, err
		}
	}

	if len(userItems) == 0 {
		return nil, nil
	}

	// 获取所有物品
	allItems, err := r.Store.GetAllItems(ctx)
	if err != nil {
		return nil, err
	}

	// 计算用户历史物品与其他物品的相似度
	type itemSimilarity struct {
		itemID    string
		similarity float64
	}

	topKSimilar := r.TopKSimilarItems
	if topKSimilar <= 0 {
		topKSimilar = r.Config.DefaultTopKItems() // 使用 TopKItems 作为默认值
	}

	minCommon := r.MinCommonUsers
	if minCommon <= 0 {
		minCommon = r.Config.DefaultMinCommonUsers()
	}

	// 验证必需字段
	if r.SimilarityCalculator == nil {
		return nil, fmt.Errorf("SimilarityCalculator is required")
	}
	if r.Config == nil {
		return nil, fmt.Errorf("Config is required")
	}

	// 为每个用户历史物品，找到相似物品
	itemScores := make(map[string]float64)

	for historyItemID, historyScore := range userItems {
		// 获取历史物品的交互用户
		historyItemUsers, err := r.Store.GetItemUsers(ctx, historyItemID)
		if err != nil {
			continue
		}

		similarities := make([]itemSimilarity, 0)

		// 计算与其他物品的相似度
		for _, candidateItemID := range allItems {
			// 跳过用户已经交互过的物品
			if _, ok := userItems[candidateItemID]; ok {
				continue
			}

			// 获取候选物品的交互用户
			candidateItemUsers, err := r.Store.GetItemUsers(ctx, candidateItemID)
			if err != nil {
				continue
			}

			// 计算共同用户
			commonUsers := make(map[string]struct{})
			historyScores := make([]float64, 0)
			candidateScores := make([]float64, 0)

			for userID, historyUserScore := range historyItemUsers {
				if candidateUserScore, ok := candidateItemUsers[userID]; ok {
					commonUsers[userID] = struct{}{}
					historyScores = append(historyScores, historyUserScore)
					candidateScores = append(candidateScores, candidateUserScore)
				}
			}

			// 如果共同用户太少，跳过
			if len(commonUsers) < minCommon {
				continue
			}

			// 计算相似度
			sim := r.SimilarityCalculator.Calculate(historyScores, candidateScores)

			if sim > 0 { // 只保留正相似度
				similarities = append(similarities, itemSimilarity{
					itemID:    candidateItemID,
					similarity: sim,
				})
			}
		}

		// 排序取 TopK 相似物品
		sort.Slice(similarities, func(i, j int) bool {
			return similarities[i].similarity > similarities[j].similarity
		})
		if len(similarities) > topKSimilar {
			similarities = similarities[:topKSimilar]
		}

		// 加权累加：用户对历史物品的评分 * 物品相似度
		for _, sim := range similarities {
			itemScores[sim.itemID] += historyScore * sim.similarity
		}
	}

	// 转换为排序列表
	type scoredItem struct {
		itemID string
		score  float64
	}
	scoredItems := make([]scoredItem, 0, len(itemScores))
	for itemID, score := range itemScores {
		scoredItems = append(scoredItems, scoredItem{
			itemID: itemID,
			score:  score,
		})
	}

	// 排序取 TopK
	topK := r.TopKItems
	if topK <= 0 {
		topK = r.Config.DefaultTopKItems()
	}
	sort.Slice(scoredItems, func(i, j int) bool {
		return scoredItems[i].score > scoredItems[j].score
	})
	if len(scoredItems) > topK {
		scoredItems = scoredItems[:topK]
	}

	// 构建结果
	out := make([]*core.Item, 0, len(scoredItems))
	for _, s := range scoredItems {
		it := core.NewItem(s.itemID)
		it.Score = s.score
		it.PutLabel("recall_source", utils.Label{Value: "i2i", Source: "recall"}) // 工业标准：i2i
		out = append(out, it)
	}

	return out, nil
}

// U2IRecall 是 UserBasedCF 的类型别名，提供更符合工业习惯的命名。
// u2i (User-to-Item) 表示"直接给用户算候选物品集合"的召回方向。
type U2IRecall = UserBasedCF

// I2IRecall 是 ItemBasedCF 的类型别名，提供更符合工业习惯的命名。
// i2i (Item-to-Item) 是工业级召回的"常青树"，电商、内容流、短视频都在用。
type I2IRecall = ItemBasedCF

// pearsonCorrelation 计算皮尔逊相关系数
func pearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}

	// 计算均值
	var meanX, meanY float64
	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= float64(len(x))
	meanY /= float64(len(y))

	// 计算协方差和方差
	var cov, varX, varY float64
	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY
		cov += dx * dy
		varX += dx * dx
		varY += dy * dy
	}

	if varX == 0 || varY == 0 {
		return 0
	}

	return cov / math.Sqrt(varX*varY)
}

// cosineSimilarityVector 计算两个向量的余弦相似度
func cosineSimilarityVector(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}
	var dotProduct, normX, normY float64
	for i := range x {
		dotProduct += x[i] * y[i]
		normX += x[i] * x[i]
		normY += y[i] * y[i]
	}
	if normX == 0 || normY == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(normX) * math.Sqrt(normY))
}
