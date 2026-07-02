package recall

import (
	"context"
	"fmt"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// HistoryWeightFunc 历史位置衰减函数。
// index 为行为序列中的位置（0-based，越小越近期），返回该位置的权重。
type HistoryWeightFunc func(index int) float64

// LinearDecay 线性衰减：weight = 1/(index+1)。
// 适用于长窗口离线 I2I / Swing 召回。
func LinearDecay(index int) float64 {
	return 1.0 / float64(index+1)
}

// QuadraticDecay 二次衰减：weight = 1/(index+1)²。
// 适用于短窗口会话级召回，更强调最近行为。
func QuadraticDecay(index int) float64 {
	d := float64(index + 1)
	return 1.0 / (d * d)
}

// PrecomputedI2IRecall 基于离线预计算的 item-to-item 相似度召回源。
//
// 工作流程：
//  1. 获取用户行为历史（优先从 rctx.Params / Attributes 读取，降级到 HistoryStore）
//  2. 对每个历史物品，通过 SimilarStore 批量读取预计算的相似物品
//  3. 按历史位置衰减权重 × 相似度分数 聚合候选分数
//  4. TopK 截断返回
//
// 通过配置不同的 SimilarKeyPrefix 和 WeightFunc，可实现多种变体：
//   - CF I2I：prefix = "cf:similar"，WeightFunc = LinearDecay
//   - Swing I2I：prefix = "swing:similar"，WeightFunc = LinearDecay
//   - Session 短兴趣：prefix = "cf:similar"，WeightFunc = QuadraticDecay
type PrecomputedI2IRecall struct {
	// SimilarStore 预计算相似度数据源（必需）。
	// 若实现 core.VersionedSimilarStore，支持自动版本化 key。
	SimilarStore core.PrecomputedSimilarStore

	// HistoryStore 用户行为历史存储（可选）。
	// 当 rctx.Params / Attributes 中无历史数据时降级使用。
	HistoryStore UserHistoryStore

	// NodeName 自定义节点名称。为空时默认 "recall.precomputed_i2i"。
	NodeName string

	// SimilarKeyPrefix 相似度数据 key 前缀。
	// 拼接规则：{prefix}:{item_id} 或 {prefix}:v{version}:{item_id}。
	// 默认 "cf:similar"。
	SimilarKeyPrefix string

	// VersionKey 版本号 key（可选）。
	// 当 SimilarStore 实现 VersionedSimilarStore 且此字段非空时，
	// 读取活跃版本号拼接为 {prefix}:v{version}:{item_id}。
	// 为空时不启用版本化，直接使用 SimilarKeyPrefix。
	VersionKey string

	// HistoryKey 从 rctx.Params / Attributes 读取用户历史的 key。
	// 期望值类型为 []string（物品 ID 列表，按时间降序）。
	// 默认 "user_history"。
	HistoryKey string

	// HistoryKeyPrefix 从 HistoryStore 读取历史时的 key 前缀。
	// 默认 "user:history"。
	HistoryKeyPrefix string

	// HistoryBehaviorType 从 HistoryStore 读取历史时的行为类型。
	// 默认 "click"。
	HistoryBehaviorType string

	// HistoryTimeWindow 从 HistoryStore 读取历史的时间窗口（秒）。
	// 默认 86400（1 天）。
	HistoryTimeWindow int64

	// MaxHistoryItems 参与扩展的最大历史条数。
	// 历史按时间降序排列，靠后的权重极低，截断可减少批量查询开销。
	// 默认 50。
	MaxHistoryItems int

	// TopKPerItem 每个历史物品最多取多少个相似物品。
	// 默认 10。
	TopKPerItem int

	// TopK 最终返回条数。
	// 默认 30。
	TopK int

	// WeightFunc 历史位置衰减函数（可选）。
	// 默认 LinearDecay。
	WeightFunc HistoryWeightFunc
}

func (r *PrecomputedI2IRecall) Name() string {
	if r.NodeName != "" {
		return r.NodeName
	}
	return "recall.precomputed_i2i"
}

func (r *PrecomputedI2IRecall) Kind() pipeline.Kind { return pipeline.KindRecall }

func (r *PrecomputedI2IRecall) Process(ctx context.Context, rctx *core.RecommendContext, _ []*core.Item) ([]*core.Item, error) {
	return r.Recall(ctx, rctx)
}

func (r *PrecomputedI2IRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
	if r == nil || r.SimilarStore == nil || rctx == nil || rctx.UserID == "" {
		return []*core.Item{}, nil
	}

	history, err := r.getUserHistory(ctx, rctx)
	if err != nil {
		return []*core.Item{}, err
	}
	if len(history) == 0 {
		return []*core.Item{}, nil
	}

	maxHistory := r.maxHistoryItems()
	if len(history) > maxHistory {
		history = history[:maxHistory]
	}

	prefix, ok := r.resolvePrefix(ctx)
	if !ok {
		return []*core.Item{}, nil
	}

	historySet := make(map[string]struct{}, len(history))
	keys := make([]string, 0, len(history))
	for _, itemID := range history {
		if itemID == "" {
			continue
		}
		historySet[itemID] = struct{}{}
		keys = append(keys, fmt.Sprintf("%s:%s", prefix, itemID))
	}
	if len(keys) == 0 {
		return []*core.Item{}, nil
	}

	topKPerItem := r.topKPerItem()
	results, err := r.SimilarStore.BatchGetSimilar(ctx, keys, topKPerItem)
	if err != nil {
		return []*core.Item{}, fmt.Errorf("precomputed_i2i: batch get similar failed: %w", err)
	}

	weightFn := r.WeightFunc
	if weightFn == nil {
		weightFn = LinearDecay
	}

	topK := r.topK()
	scores := make(map[string]float64, topK*3)
	for i, similar := range results {
		if len(similar) == 0 {
			continue
		}
		w := weightFn(i)
		for _, sm := range similar {
			if sm.Member == "" || sm.Score <= 0 {
				continue
			}
			if _, interacted := historySet[sm.Member]; interacted {
				continue
			}
			scores[sm.Member] += w * sm.Score
		}
	}

	if len(scores) == 0 {
		return []*core.Item{}, nil
	}

	type scored struct {
		id    string
		score float64
	}
	candidates := make([]scored, 0, len(scores))
	for id, s := range scores {
		candidates = append(candidates, scored{id: id, score: s})
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}

	out := make([]*core.Item, 0, len(candidates))
	for _, c := range candidates {
		it := core.NewItem(c.id)
		it.Score = c.score
		it.PutLabel("recall_source", utils.Label{Value: r.Name(), Source: "recall"})
		out = append(out, it)
	}
	return out, nil
}

func (r *PrecomputedI2IRecall) getUserHistory(ctx context.Context, rctx *core.RecommendContext) ([]string, error) {
	key := r.HistoryKey
	if key == "" {
		key = "user_history"
	}

	if rctx.Params != nil {
		if h, ok := rctx.Params[key].([]string); ok && len(h) > 0 {
			return h, nil
		}
	}
	if rctx.Attributes != nil {
		if h, ok := rctx.Attributes[key].([]string); ok && len(h) > 0 {
			return h, nil
		}
	}

	if r.HistoryStore == nil {
		return nil, nil
	}

	prefix := r.HistoryKeyPrefix
	if prefix == "" {
		prefix = "user:history"
	}
	behavior := r.HistoryBehaviorType
	if behavior == "" {
		behavior = "click"
	}
	window := r.HistoryTimeWindow
	if window <= 0 {
		window = 86400
	}

	items, err := r.HistoryStore.GetUserHistory(ctx, rctx.UserID, prefix, behavior, window)
	if err != nil {
		return nil, err
	}
	ids := make([]string, len(items))
	for i, h := range items {
		ids[i] = h.ItemID
	}
	return ids, nil
}

func (r *PrecomputedI2IRecall) resolvePrefix(ctx context.Context) (string, bool) {
	prefix := r.SimilarKeyPrefix
	if prefix == "" {
		prefix = "cf:similar"
	}

	if r.VersionKey == "" {
		return prefix, true
	}

	vs, ok := r.SimilarStore.(core.VersionedSimilarStore)
	if !ok {
		return prefix, true
	}

	version, err := vs.GetActiveVersion(ctx, r.VersionKey)
	if err != nil || version == "" {
		return "", false
	}
	return fmt.Sprintf("%s:v%s", prefix, version), true
}

func (r *PrecomputedI2IRecall) maxHistoryItems() int {
	if r.MaxHistoryItems > 0 {
		return r.MaxHistoryItems
	}
	return 50
}

func (r *PrecomputedI2IRecall) topKPerItem() int {
	if r.TopKPerItem > 0 {
		return r.TopKPerItem
	}
	return 10
}

func (r *PrecomputedI2IRecall) topK() int {
	if r.TopK > 0 {
		return r.TopK
	}
	return 30
}
