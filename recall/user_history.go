package recall

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// UserHistory 是基于用户 historical 行为的个性化召回源。
// 支持从 Store 读取用户的浏览、点击、购买等历史，推荐相似物品。
type UserHistory struct {
	Store UserHistoryStore

	// KeyPrefix 是 Store 中的 key 前缀，实际 key 为 {KeyPrefix}:{UserID}
	KeyPrefix string

	// BehaviorType 行为类型：view / click / purchase / favorite 等
	BehaviorType string

	// TopK 返回 TopK 个物品
	TopK int

	// TimeWindow 时间窗口（秒），只考虑该时间窗口内的历史
	// 0 表示考虑所有历史
	TimeWindow int64

	// EnableSimilarExtend 是否启用相似物品扩展（I2I 召回）。
	//
	// 开启后，会通过 SimilarItemStore.GetSimilarItems() 将用户历史物品扩展为相似物品，
	// 实现 I2I (Item-to-Item) 召回，推荐用户未交互过但可能感兴趣的物品。
	//
	// 与 I2IRecall 的对比：
	//   - UserHistory + EnableSimilarExtend：
	//     * 本质：I2I 接口模式，依赖外部实现 SimilarItemStore
	//     * 输入：用户历史 item 列表
	//     * 输出：相似 item 列表
	//     * 实现方式：外部提供 I2I 数据源（预计算索引、向量检索等）
	//     * 性能：依赖外部实现，可使用预计算索引（性能更好）
	//     * 灵活性：高，可切换不同 I2I 数据源
	//     * 适用场景：生产环境推荐，使用预计算的 I2I 索引
	//
	//   - I2IRecall (ItemBasedCF)：
	//     * 本质：I2I 完整实现，内部通过协同过滤计算相似度
	//     * 输入：用户历史 item
	//     * 输出：相似 item
	//     * 实现方式：内部实时计算物品相似度（Cosine/Pearson）
	//     * 性能：实时计算，可能较慢
	//     * 灵活性：低，固定协同过滤算法
	//     * 适用场景：小规模数据或需要实时计算的场景
	//
	// 使用示例：
	//   // 方式 1：使用预计算的 I2I 索引（推荐）
	//   userHistoryRecall := &recall.UserHistory{
	//       Store:               &PrecomputedI2IStore{store: redisStore},
	//       EnableSimilarExtend: true,  // 开启 I2I
	//   }
	//
	//   // 方式 2：直接使用 I2IRecall（实时计算）
	//   i2iRecall := &recall.I2IRecall{
	//       Store:                cfStore,
	//       SimilarityCalculator: &recall.CosineSimilarity{},
	//   }
	EnableSimilarExtend bool
}

// UserHistoryStore 是用户历史存储接口。
type UserHistoryStore interface {
	// GetUserHistory 获取用户的历史行为物品列表
	GetUserHistory(ctx context.Context, userID string, keyPrefix, behaviorType string, timeWindow int64) ([]string, error)
}

// SimilarItemStore 获取相似物品的存储接口
type SimilarItemStore interface {
	// GetSimilarItems 获取给定物品列表的相似物品
	GetSimilarItems(ctx context.Context, itemIDs []string, topK int) ([]string, error)
}

func (r *UserHistory) Name() string {
	return "recall.user_history"
}

func (r *UserHistory) Kind() pipeline.Kind {
	return pipeline.KindRecall
}

func (r *UserHistory) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	_ []*core.Item,
) ([]*core.Item, error) {
	return r.Recall(ctx, rctx)
}

func (r *UserHistory) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Store == nil || rctx == nil || rctx.UserID == "" {
		return nil, nil
	}

	keyPrefix := r.KeyPrefix
	if keyPrefix == "" {
		keyPrefix = "user:history"
	}

	behaviorType := r.BehaviorType
	if behaviorType == "" {
		behaviorType = "click" // 默认使用点击历史
	}

	timeWindow := r.TimeWindow
	if timeWindow == 0 {
		// 默认考虑最近 30 天的历史
		timeWindow = 30 * 24 * 3600
	}

	// 获取用户历史物品
	itemIDs, err := r.Store.GetUserHistory(ctx, rctx.UserID, keyPrefix, behaviorType, timeWindow)
	if err != nil {
		return nil, err
	}

	if len(itemIDs) == 0 {
		return nil, nil
	}

	// 如果支持相似物品推荐，获取相似物品
	if r.EnableSimilarExtend {
		if similarStore, ok := r.Store.(SimilarItemStore); ok {
			topK := r.TopK
			if topK <= 0 {
				topK = 20
			}
			similarIDs, err := similarStore.GetSimilarItems(ctx, itemIDs, topK)
			if err == nil && len(similarIDs) > 0 {
				itemIDs = similarIDs
			}
		}
	}

	// 限制返回数量
	topK := r.TopK
	if topK > 0 && len(itemIDs) > topK {
		itemIDs = itemIDs[:topK]
	}

	// 构建结果
	out := make([]*core.Item, 0, len(itemIDs))
	for _, id := range itemIDs {
		it := core.NewItem(id)
		it.PutLabel("recall_source", utils.Label{Value: "user_history", Source: "recall"})
		it.PutLabel("behavior_type", utils.Label{Value: behaviorType, Source: "recall"})
		out = append(out, it)
	}

	return out, nil
}
