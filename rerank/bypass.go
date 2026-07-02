package rerank

import (
	"context"
	"strconv"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

const bypassParamsPrefix = "__bypass:"

// BypassExtractNode 在 Rank 之前将特定渠道的物品提取出来跳过后续处理，
// 配合 BypassBoostNode 在 Rank/Diversity 之后前置回结果，保持原始顺序。
//
// 典型场景：运营精排池（L0）、编辑推荐位等需要保持固定排序的物品。
type BypassExtractNode struct {
	// GroupName 旁路组名称（必需），用于 Extract/Boost 配对。
	GroupName string

	// MatchFunc 判断 item 是否应被提取。
	// 与 LabelKey/LabelValue 二选一；MatchFunc 优先。
	MatchFunc func(item *core.Item) bool

	// LabelKey + LabelValue 通过 label 匹配（MatchFunc 为 nil 时使用）。
	LabelKey   string
	LabelValue string

	// SortByLabel 提取后按此 label 的数值排序（可选，如 "pool_rank"）。
	// 为空则保持原始顺序。
	SortByLabel string
	SortAsc     bool // true = 升序（默认），false = 降序
}

func (n *BypassExtractNode) Name() string        { return "extract.bypass." + n.GroupName }
func (n *BypassExtractNode) Kind() pipeline.Kind { return pipeline.KindPostProcess }

func (n *BypassExtractNode) Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	if n == nil || n.GroupName == "" || len(items) == 0 {
		return items, nil
	}

	matchFn := n.MatchFunc
	if matchFn == nil && n.LabelKey != "" {
		lk, lv := n.LabelKey, n.LabelValue
		matchFn = func(item *core.Item) bool {
			if item.Labels == nil {
				return false
			}
			lbl, ok := item.Labels[lk]
			return ok && lbl.Value == lv
		}
	}
	if matchFn == nil {
		return items, nil
	}

	var extracted, rest []*core.Item
	for _, it := range items {
		if it != nil && matchFn(it) {
			extracted = append(extracted, it)
		} else {
			rest = append(rest, it)
		}
	}
	if len(extracted) == 0 {
		return items, nil
	}

	if n.SortByLabel != "" {
		sortByLabelValue(extracted, n.SortByLabel, n.SortAsc)
	}

	if rctx.Params == nil {
		rctx.Params = make(map[string]any)
	}
	rctx.Params[bypassParamsPrefix+n.GroupName] = extracted
	return rest, nil
}

// BypassBoostNode 将之前提取的物品前置到结果列表。
type BypassBoostNode struct {
	// GroupName 旁路组名称（必需），须与 BypassExtractNode.GroupName 一致。
	GroupName string

	// MaxItems 最多前置的物品数。0 = 不限制。
	MaxItems int
}

func (n *BypassBoostNode) Name() string        { return "rerank.bypass_boost." + n.GroupName }
func (n *BypassBoostNode) Kind() pipeline.Kind { return pipeline.KindReRank }

func (n *BypassBoostNode) Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	if n == nil || n.GroupName == "" {
		return items, nil
	}
	key := bypassParamsPrefix + n.GroupName
	if rctx == nil || rctx.Params == nil {
		return items, nil
	}
	saved, ok := rctx.Params[key].([]*core.Item)
	if !ok || len(saved) == 0 {
		return items, nil
	}
	delete(rctx.Params, key)

	if n.MaxItems > 0 && len(saved) > n.MaxItems {
		saved = saved[:n.MaxItems]
	}
	return append(saved, items...), nil
}

func sortByLabelValue(items []*core.Item, labelKey string, asc bool) {
	for i := 1; i < len(items); i++ {
		for j := i; j > 0; j-- {
			a := labelInt(items[j-1], labelKey)
			b := labelInt(items[j], labelKey)
			swap := (asc && a > b) || (!asc && a < b)
			if swap {
				items[j-1], items[j] = items[j], items[j-1]
			} else {
				break
			}
		}
	}
}

func labelInt(it *core.Item, key string) int {
	if it == nil || it.Labels == nil {
		return 0
	}
	if lbl, ok := it.Labels[key]; ok {
		n, _ := strconv.Atoi(lbl.Value)
		return n
	}
	return 0
}
