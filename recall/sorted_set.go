package recall

import (
	"context"
	"encoding/json"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// SortOrder 定义有序集合的排序方向。
type SortOrder string

const (
	// OrderDesc 降序（默认）：适用于热门（按热度）、评分最高（按评分）、最新（按时间戳）等。
	OrderDesc SortOrder = "desc"
	// OrderAsc 升序：适用于价格最低、距离最近等场景。
	OrderAsc SortOrder = "asc"
)

// SortedSetRecall 是基于外部有序集合的通用召回源。
//
// 它是 Hot / Trending / Latest / TopRated 等具体业务召回的底层抽象：
// 从 Store 的有序集合中按指定方向拉取 TopK 个 item ID 及分数。
//
// 数据读取策略（按优先级）：
//  1. SortedSetRangeStore → ZRevRangeWithScores / ZRangeWithScores（带分数 + 双向）
//  2. KeyValueStore → ZRange（仅降序、不含分数）
//  3. Store.Get → JSON 数组（纯 ID 列表）
//  4. IDs 字段 → fallback 静态列表
//
// Key 解析优先级：Key > KeyPrefix+Scene > KeyPrefix。
//
// 常见业务场景与对应构造器：
//
//	NewHotRecall      → 热门召回（按热度降序）
//	NewTrendingRecall → 趋势召回（按趋势分降序）
//	NewLatestRecall   → 最新召回（按发布时间降序）
//	NewTopRatedRecall → 高分召回（按评分降序）
type SortedSetRecall struct {
	Store core.Store

	// Key 完整的存储 key（优先使用）。
	Key string

	// KeyPrefix key 前缀。当 Key 为空时：
	//   - 若 rctx.Scene 非空，实际 key = KeyPrefix:Scene
	//   - 否则 key = KeyPrefix
	KeyPrefix string

	// TopK 返回数量上限，默认 100。
	TopK int

	// Order 排序方向，默认 OrderDesc。
	Order SortOrder

	// IDs fallback 静态 ID 列表（当 Store 不可用或为空时使用）。
	IDs []string

	// NodeName 自定义节点名称，影响 Name() 返回值和 recall_source 标签。
	// 默认 "recall.sorted_set"。
	NodeName string
}

func (r *SortedSetRecall) Name() string {
	if r.NodeName != "" {
		return r.NodeName
	}
	return "recall.sorted_set"
}

func (r *SortedSetRecall) Kind() pipeline.Kind { return pipeline.KindRecall }

func (r *SortedSetRecall) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	_ []*core.Item,
) ([]*core.Item, error) {
	return r.Recall(ctx, rctx)
}

func (r *SortedSetRecall) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	topK := r.TopK
	if topK <= 0 {
		topK = 100
	}
	stop := int64(topK - 1)

	key := r.resolveKey(rctx)

	// 策略 1: SortedSetRangeStore（带分数 + 双向）
	if key != "" && r.Store != nil {
		if ssStore, ok := r.Store.(core.SortedSetRangeStore); ok {
			members, err := r.rangeWithScores(ctx, ssStore, key, stop)
			if err == nil && len(members) > 0 {
				return r.buildItemsFromScored(members), nil
			}
		}
	}

	// 策略 2: KeyValueStore.ZRange（仅降序、不含分数）
	if key != "" && r.Store != nil {
		if kvStore, ok := r.Store.(core.KeyValueStore); ok {
			ids, err := kvStore.ZRange(ctx, key, 0, stop)
			if err == nil && len(ids) > 0 {
				if r.Order == OrderAsc {
					reverseStrings(ids)
				}
				return r.buildItemsFromIDs(ids), nil
			}
		}
	}

	// 策略 3: Store.Get + JSON
	if key != "" && r.Store != nil {
		data, err := r.Store.Get(ctx, key)
		if err == nil {
			var ids []string
			if json.Unmarshal(data, &ids) == nil && len(ids) > 0 {
				if topK > 0 && len(ids) > topK {
					ids = ids[:topK]
				}
				return r.buildItemsFromIDs(ids), nil
			}
		}
	}

	// 策略 4: fallback 静态列表
	if len(r.IDs) > 0 {
		ids := r.IDs
		if topK > 0 && len(ids) > topK {
			ids = ids[:topK]
		}
		return r.buildItemsFromIDs(ids), nil
	}

	return nil, nil
}

func (r *SortedSetRecall) resolveKey(rctx *core.RecommendContext) string {
	if r.Key != "" {
		return r.Key
	}
	if r.KeyPrefix == "" {
		return ""
	}
	if rctx != nil && rctx.Scene != "" {
		return r.KeyPrefix + ":" + rctx.Scene
	}
	return r.KeyPrefix
}

func (r *SortedSetRecall) rangeWithScores(
	ctx context.Context,
	store core.SortedSetRangeStore,
	key string,
	stop int64,
) ([]core.ScoredMember, error) {
	if r.Order == OrderAsc {
		return store.ZRangeWithScores(ctx, key, 0, stop)
	}
	return store.ZRevRangeWithScores(ctx, key, 0, stop)
}

func (r *SortedSetRecall) buildItemsFromScored(members []core.ScoredMember) []*core.Item {
	out := make([]*core.Item, 0, len(members))
	for _, m := range members {
		it := core.NewItem(m.Member)
		it.Score = m.Score
		it.PutLabel("recall_source", utils.Label{Value: r.Name(), Source: "recall"})
		out = append(out, it)
	}
	return out
}

func (r *SortedSetRecall) buildItemsFromIDs(ids []string) []*core.Item {
	out := make([]*core.Item, 0, len(ids))
	for _, id := range ids {
		it := core.NewItem(id)
		it.PutLabel("recall_source", utils.Label{Value: r.Name(), Source: "recall"})
		out = append(out, it)
	}
	return out
}

func reverseStrings(s []string) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}

// ---------------------------------------------------------------------------
// 便捷构造器 —— 常见业务场景
// ---------------------------------------------------------------------------

// NewHotRecall 创建热门召回（按热度降序）。
//
//	recall := NewHotRecall(store, "hot:feed", 100)
//	// 或使用 key_prefix，key 按 scene 自动拼接：
//	recall := NewHotRecall(store, "", 100)
//	recall.KeyPrefix = "hot"  // key = hot:{scene}
func NewHotRecall(store core.Store, key string, topK int) *SortedSetRecall {
	return &SortedSetRecall{
		Store: store, Key: key, TopK: topK,
		Order: OrderDesc, NodeName: "recall.hot",
	}
}

// NewTrendingRecall 创建趋势召回（按趋势分降序）。
func NewTrendingRecall(store core.Store, key string, topK int) *SortedSetRecall {
	return &SortedSetRecall{
		Store: store, Key: key, TopK: topK,
		Order: OrderDesc, NodeName: "recall.trending",
	}
}

// NewLatestRecall 创建最新召回（按发布时间降序，最新优先）。
func NewLatestRecall(store core.Store, key string, topK int) *SortedSetRecall {
	return &SortedSetRecall{
		Store: store, Key: key, TopK: topK,
		Order: OrderDesc, NodeName: "recall.latest",
	}
}

// NewTopRatedRecall 创建高分召回（按评分降序）。
func NewTopRatedRecall(store core.Store, key string, topK int) *SortedSetRecall {
	return &SortedSetRecall{
		Store: store, Key: key, TopK: topK,
		Order: OrderDesc, NodeName: "recall.top_rated",
	}
}

// NewEditorPickRecall 创建编辑推荐召回（按运营权重降序）。
func NewEditorPickRecall(store core.Store, key string, topK int) *SortedSetRecall {
	return &SortedSetRecall{
		Store: store, Key: key, TopK: topK,
		Order: OrderDesc, NodeName: "recall.editor_pick",
	}
}
